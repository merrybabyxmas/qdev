import asyncio
import time
import numpy as np
from src.ingestion.websocket_client import MultiSymbolHFTStreamManager
from src.models.state_detector import MarketStateDetector
from src.models.ranker_engine import RealTimeCrossSectionalRanker
from src.signals.router import PipelineRouter, ExecutionAction
from src.execution.policy import ExecutionTracker
from src.risk.manager import RiskManager
from src.brokers.base import BrokerInterface
from src.models.champion_registry import ChampionRegistry
from src.monitoring.control_plane import HFTControlPlane
from src.utils.logger import logger

class LiveTradingEngine:
    """
    HFT Execution Plane: 스트림을 실시간 처리하며,
    Control Plane(스케줄러)이 발행한 hft_policy.json을 주기적으로 읽어 실행 여부를 상위에서 통제받습니다.
    """
    def __init__(self, symbols, broker_engine: BrokerInterface, is_simulation=False):
        self.symbols = symbols
        self.broker = broker_engine

        self.stream_manager = MultiSymbolHFTStreamManager(symbols=symbols, enable_live_stream=not is_simulation)
        self.stream_manager.on_feature_update = self._on_feature_event

        self.detector = MarketStateDetector()

        # Load the HFT champion model from the specialized leaderboard
        self.registry = ChampionRegistry()
        champ = self.registry.get_champion("hft_microstructure")
        logger.info(f"Loaded HFT Champion Model: {champ.get('model_id', 'Default')} (Sharpe: {champ.get('sharpe', 0.0)})")

        # Control Plane Integration
        self.control_plane = HFTControlPlane()
        self.current_policy = self.control_plane.read_policy()
        self.last_policy_sync_time = time.time()
        self.policy_sync_interval = 10.0 # 스케줄러 정책을 10초마다 확인

        self.ranker = RealTimeCrossSectionalRanker(symbols=symbols)
        self.router = PipelineRouter()

        self.tracker = ExecutionTracker(broker_or_engine=self.broker, cancel_threshold_bps=1.0)
        self.risk_manager = RiskManager(max_position_cap=0.40)

        self.is_simulation = is_simulation
        self.tick_counter = 0

        self.cached_equity = 100000.0
        self.cached_positions = {sym: 0.0 for sym in symbols}
        self.last_account_sync_time = 0.0
        self.account_sync_interval = 60.0

    def _sync_account_state(self):
        now = time.time()
        if now - self.last_account_sync_time > self.account_sync_interval:
            try:
                acct = self.broker.get_account()
                self.cached_equity = acct.get("equity", self.cached_equity)

                positions = self.broker.get_positions()
                for sym in self.symbols:
                    self.cached_positions[sym] = positions.get(sym, 0.0)

                self.risk_manager.evaluate_account_risk(self.cached_equity)
                self.last_account_sync_time = now
            except Exception as e:
                logger.error(f"Failed to sync account info: {e}")

    def _sync_macro_policy(self):
        """10초마다 상위 스케줄러가 작성한 hft_policy.json 정책을 로드하여 엔진 동기화"""
        now = time.time()
        if now - self.last_policy_sync_time > self.policy_sync_interval:
            new_policy = self.control_plane.read_policy()
            if self.current_policy.get("allow_hft") != new_policy.get("allow_hft"):
                logger.info(f"Control Plane Policy Changed! allow_hft: {new_policy.get('allow_hft')}")
            self.current_policy = new_policy
            self.last_policy_sync_time = now

    def _on_feature_event(self, symbol: str, feature_event: dict):
        self.tick_counter += 1
        t = feature_event["timestamp"]
        b = feature_event["bid"]
        a = feature_event["ask"]

        self._sync_account_state()
        self._sync_macro_policy()

        equity = self.cached_equity
        current_qty = self.cached_positions.get(symbol, 0.0)

        current_state = self.detector.detect_state(feature_event)

        preds, raw_targets = self.ranker.update_and_predict(symbol, feature_event)
        target_weights = self.risk_manager.apply_position_caps(raw_targets)

        target_weight_for_sym = target_weights.get(symbol, 0.0)
        pred_for_sym = preds.get(symbol, 0.0)

        # Context-Aware Pipeline Execution Routing
        action: ExecutionAction = self.router.route_execution(
            state=current_state,
            prediction=pred_for_sym,
            policy=self.current_policy,
            symbol=symbol
        )

        if action.action == "HALT":
            # Override target weight to 0 to safely exit any open positions when halted
            target_weight_for_sym = 0.0

        self.tracker.evaluate_cancel_replace(t, symbol, b, a)

        mid_price = feature_event["mid_price"]
        delta_qty = self.risk_manager.calculate_order_qty(symbol, target_weight_for_sym, current_qty, mid_price, equity)

        active_orders_for_sym = {k: v for k, v in self.tracker.active_orders.items() if v["symbol"] == symbol}

        if action.action == "HALT" or delta_qty == 0.0:
            for oid in list(active_orders_for_sym.keys()):
                if self.is_simulation:
                    self.broker.cancel_order(oid, t)
                else:
                    self.broker.cancel_order(oid)
                self.tracker.untrack_order(oid)

        elif action.action == "PASSIVE_MAKE" and not active_orders_for_sym and delta_qty != 0.0:
            side = "buy" if delta_qty > 0 else "sell"
            qty = abs(delta_qty)
            price = b if side == "buy" else a

            if self.is_simulation:
                oid = self.broker.place_limit_order(symbol, side, price, qty, t)
            else:
                oid = self.broker.place_limit_order(symbol, side, price, qty)
            if oid:
                self.tracker.track_order(oid, symbol, side, price, qty)

        elif action.action == "AGGRESSIVE_TAKE" and not active_orders_for_sym and delta_qty != 0.0:
            side = "buy" if delta_qty > 0 else "sell"
            qty = abs(delta_qty)
            price = a if side == "buy" else b

            if self.is_simulation:
                oid = self.broker.place_limit_order(symbol, side, price, qty, t)
            else:
                oid = self.broker.place_limit_order(symbol, side, price, qty)
            if oid:
                self.tracker.track_order(oid, symbol, side, price, qty)

        if self.is_simulation:
            self.broker.process_quote_update(t, b, a)
            for oid in list(self.tracker.active_orders.keys()):
                if oid not in getattr(self.broker, "active_orders", {}):
                    self.tracker.untrack_order(oid)

            self.cached_positions[symbol] = self.broker.inventory

    def start(self):
        logger.info("=== Starting Continuous Live Trading Engine (Execution Plane) ===")
        initial_policy = self.control_plane.read_policy()
        logger.info(f"Initial Policy Status - HFT Allowed: {initial_policy.get('allow_hft')}")

        if not self.is_simulation:
            self._sync_account_state()

        self.stream_manager.run()
