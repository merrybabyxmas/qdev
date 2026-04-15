import asyncio
import numpy as np
from src.ingestion.websocket_client import MultiSymbolHFTStreamManager
from src.models.state_detector import MarketStateDetector
from src.models.ranker_engine import RealTimeCrossSectionalRanker
from src.signals.router import PipelineRouter, ExecutionAction
from src.execution.policy import ExecutionTracker
from src.risk.manager import RiskManager
from src.utils.logger import logger

class LiveTradingEngine:
    """
    모든 HFT 컴포넌트(스트림 수신 -> 마이크로스트럭처 계산 -> 온라인 학습 -> 랭킹 -> 상태 기반 파이프라인 라우팅 -> 취소/재주문 실행)를
    무한 루프(Event-driven Loop) 안에서 엮는 실제 트레이딩 엔진.
    """
    def __init__(self, symbols, broker_engine, is_simulation=False):
        self.symbols = symbols
        self.broker = broker_engine

        self.stream_manager = MultiSymbolHFTStreamManager(symbols=symbols)
        self.stream_manager.on_feature_update = self._on_feature_event

        self.detector = MarketStateDetector(high_vol_threshold=0.5, toxic_vpin_threshold=0.8, trend_threshold=0.001)
        self.ranker = RealTimeCrossSectionalRanker(symbols=symbols, target_lookahead=5)
        self.router = PipelineRouter()

        self.tracker = ExecutionTracker(broker_or_engine=self.broker, cancel_threshold_bps=1.0)
        self.risk_manager = RiskManager(max_position_cap=0.40)

        self.is_simulation = is_simulation
        self.tick_counter = 0

    def _on_feature_event(self, symbol: str, feature_event: dict):
        """Websocket 스트림에서 틱마다 콜백됨."""
        self.tick_counter += 1
        t = feature_event["timestamp"]
        b = feature_event["bid"]
        a = feature_event["ask"]

        # 1. State Detection
        current_state = self.detector.detect_state(feature_event)

        # 2. Continuous Cross-sectional Ranking & Online Update
        preds, raw_targets = self.ranker.update_and_predict(symbol, feature_event)

        # 3. Apply Portfolio Caps
        target_weights = self.risk_manager.apply_position_caps(raw_targets)

        target_weight_for_sym = target_weights.get(symbol, 0.0)
        pred_for_sym = preds.get(symbol, 0.0)

        if self.tick_counter % 50 == 0:
            logger.info(f"[{symbol}] State: {current_state.name} | Target W: {target_weight_for_sym:.2%} | Raw Pred: {pred_for_sym:+.2f} bps")

        # 4. Pipeline Execution Routing
        action: ExecutionAction = self.router.route_execution(current_state, pred_for_sym)

        # 5. Execution Policy (Cancel/Replace tracking)
        # Check all active orders for this symbol, replace if lagging behind real-time quote
        self.tracker.evaluate_cancel_replace(t, symbol, b, a)

        # 6. Execute newly proposed actions
        active_orders_for_sym = {k: v for k, v in self.tracker.active_orders.items() if v["symbol"] == symbol}

        if action.action == "HALT":
            for oid in list(active_orders_for_sym.keys()):
                self.broker.cancel_order(oid, t) if self.is_simulation else self.broker.cancel_order(oid)
                self.tracker.untrack_order(oid)

        elif action.action == "PASSIVE_MAKE" and not active_orders_for_sym and target_weight_for_sym > 0:
            # Place buy order at Bid (Provide Liquidity)
            if self.is_simulation:
                oid = self.broker.place_limit_order(symbol, "buy", b, 1.0 * action.size_multiplier, t)
            else:
                oid = self.broker.place_order({"symbol": symbol, "side": "buy", "price": b, "qty": 1.0})
            self.tracker.track_order(oid, symbol, "buy", b, 1.0 * action.size_multiplier)

        elif action.action == "AGGRESSIVE_TAKE" and not active_orders_for_sym and target_weight_for_sym > 0:
            # Place buy order at Ask (Cross the Spread / Take Liquidity)
            if self.is_simulation:
                oid = self.broker.place_limit_order(symbol, "buy", a, 1.0 * action.size_multiplier, t)
            else:
                oid = self.broker.place_order({"symbol": symbol, "side": "buy", "price": a, "qty": 1.0})
            self.tracker.track_order(oid, symbol, "buy", a, 1.0 * action.size_multiplier)

        # Simulate Matching Process if running offline/simulation broker
        if self.is_simulation:
            self.broker.process_quote_update(t, b, a)

            # Clean up filled orders from tracker
            for oid in list(self.tracker.active_orders.keys()):
                if oid not in self.broker.active_orders:
                    self.tracker.untrack_order(oid)

    def start(self):
        """실시간(Live) 트레이딩 루프 시작 (블로킹)"""
        logger.info("=== Starting Continuous Live Trading Engine ===")
        self.stream_manager.run()
