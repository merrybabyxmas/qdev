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
from src.utils.logger import logger

class LiveTradingEngine:
    """
    모든 HFT 컴포넌트(스트림 수신 -> 마이크로스트럭처 계산 -> 챔피언 모델 -> 랭킹 -> 라우팅 -> 실행)를
    무한 루프 안에서 엮는 실제 트레이딩 엔진.
    """
    def __init__(self, symbols, broker_engine: BrokerInterface, is_simulation=False):
        self.symbols = symbols
        self.broker = broker_engine

        self.stream_manager = MultiSymbolHFTStreamManager(symbols=symbols, enable_live_stream=not is_simulation)
        self.stream_manager.on_feature_update = self._on_feature_event

        self.detector = MarketStateDetector()

        # Load the champion model
        self.registry = ChampionRegistry()
        champ = self.registry.get_champion()
        logger.info(f"Loaded Champion Model: {champ.get('model_id', 'Default')} (Sharpe: {champ.get('sharpe', 0.0)})")

        self.ranker = RealTimeCrossSectionalRanker(symbols=symbols)
        self.router = PipelineRouter()

        self.tracker = ExecutionTracker(broker_or_engine=self.broker, cancel_threshold_bps=1.0)
        self.risk_manager = RiskManager(max_position_cap=0.40)

        self.is_simulation = is_simulation
        self.tick_counter = 0

        # Local state caches to prevent API rate limits (HTTP 429)
        self.cached_equity = 100000.0
        self.cached_positions = {sym: 0.0 for sym in symbols}
        self.last_account_sync_time = 0.0
        self.account_sync_interval = 60.0 # sync every 60 seconds

    def _sync_account_state(self):
        """Throttled REST API calls to fetch real account state."""
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

    def _on_feature_event(self, symbol: str, feature_event: dict):
        """Websocket 스트림에서 틱마다 콜백됨."""
        self.tick_counter += 1
        t = feature_event["timestamp"]
        b = feature_event["bid"]
        a = feature_event["ask"]

        # 1. Sync Account State (Throttled to avoid rate limits)
        self._sync_account_state()

        equity = self.cached_equity
        current_qty = self.cached_positions.get(symbol, 0.0)

        # 2. State Detection
        current_state = self.detector.detect_state(feature_event)

        # 3. Continuous Cross-sectional Ranking & Online Update
        preds, raw_targets = self.ranker.update_and_predict(symbol, feature_event)

        # 4. Apply Portfolio Caps
        target_weights = self.risk_manager.apply_position_caps(raw_targets)

        target_weight_for_sym = target_weights.get(symbol, 0.0)
        pred_for_sym = preds.get(symbol, 0.0)

        if self.tick_counter % 50 == 0:
            logger.info(f"[{symbol}] State: {current_state.name} | Target W: {target_weight_for_sym:.2%} | Equity: {equity:.2f} | Pos: {current_qty}")

        # 5. Pipeline Execution Routing
        action: ExecutionAction = self.router.route_execution(current_state, pred_for_sym)

        # 6. Cancel/Replace Tracking
        self.tracker.evaluate_cancel_replace(t, symbol, b, a)

        # 7. Calculate Exact USD Trade Delta based on Actual Equity
        mid_price = feature_event["mid_price"]
        delta_qty = self.risk_manager.calculate_order_qty(symbol, target_weight_for_sym, current_qty, mid_price, equity)

        active_orders_for_sym = {k: v for k, v in self.tracker.active_orders.items() if v["symbol"] == symbol}

        # 8. Execute newly proposed actions
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
            # Make passive order at the favorable side of the book
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
            # Cross the spread to take liquidity aggressively
            price = a if side == "buy" else b

            if self.is_simulation:
                oid = self.broker.place_limit_order(symbol, side, price, qty, t)
            else:
                oid = self.broker.place_limit_order(symbol, side, price, qty)
            if oid:
                self.tracker.track_order(oid, symbol, side, price, qty)

        # Simulate Matching Process if running offline/simulation broker
        if self.is_simulation:
            self.broker.process_quote_update(t, b, a)

            # Clean up filled orders from tracker
            for oid in list(self.tracker.active_orders.keys()):
                if oid not in getattr(self.broker, "active_orders", {}):
                    self.tracker.untrack_order(oid)

            # In simulation, immediately update local cache assuming fast fills
            self.cached_positions[symbol] = self.broker.inventory

    def start(self):
        """실시간(Live) 트레이딩 루프 시작 (블로킹)"""
        logger.info("=== Starting Continuous Live Trading Engine ===")
        # In live mode, we do an initial sync
        if not self.is_simulation:
            self._sync_account_state()

        self.stream_manager.run()
