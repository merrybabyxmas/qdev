import asyncio
import logging
import numpy as np
from src.state.ring_buffers import TickRingBuffer, QuoteRingBuffer
from src.features.microstructure.imbalance import (compute_order_book_imbalance,
                                                  compute_microprice,
                                                  compute_spread,
                                                  compute_trade_intensity,
                                                  compute_toxicity_vpin_proxy,
                                                  compute_volatility_burst)
from src.backtest.matching_engine import HFTMatchingEngine
from src.models.sgd_online import OnlineSGDRegressor
from src.execution.policy import ExecutionTracker
from src.models.state_detector import MarketStateDetector, MarketState
from src.signals.router import PipelineRouter, ExecutionAction
from src.utils.logger import logger

# Configure simple logging for fast console output
logging.getLogger("src").setLevel(logging.INFO)

class AdvancedOnlineHFTSimulator:
    """
    5축 기반 시장 상태 감지기(State Detector)와 전략 라우터(Pipeline Router)를 포함한
    최종 통합 HFT 시뮬레이터. 실시간으로 모델 학습과 주문 수정이 일어남.
    """
    def __init__(self):
        self.trade_buf = TickRingBuffer(capacity=100)
        self.engine = HFTMatchingEngine(latency_ms=5.0, fee_bps=0.0001)
        self.tracker = ExecutionTracker(broker_or_engine=self.engine, cancel_threshold_bps=1.0)
        self.model = OnlineSGDRegressor(learning_rate='constant', eta0=0.01)

        self.detector = MarketStateDetector(high_vol_threshold=0.5, toxic_vpin_threshold=0.8, trend_threshold=0.001)
        self.router = PipelineRouter()

        self.history_features = []
        self.history_targets = []
        self.last_mid = 100.0
        self.tick_count = 0
        self.active_order_id = None
        self.target_lookahead = 5

    def feed_trade(self, t: float, p: float, s: float, side: int):
        self.trade_buf.append(t, p, s, side)

    def feed_quote(self, t: float, b: float, bs: float, a: float, as_: float):
        self.tick_count += 1
        self.last_mid = (b + a) / 2.0

        # 1. 5축 Feature Computation
        obi = compute_order_book_imbalance(bs, as_)
        mprice = compute_microprice(b, bs, a, as_)
        spread = compute_spread(b, a)

        recent_trades = self.trade_buf.get_recent(50)
        intensity = compute_trade_intensity(recent_trades, window_ms=1000.0)
        toxicity = compute_toxicity_vpin_proxy(recent_trades, window_ms=1000.0)
        vol_burst = compute_volatility_burst(recent_trades, window_ms=1000.0)

        mprice_drift = (mprice - self.last_mid)/self.last_mid if self.last_mid > 0 else 0.0

        state_features = {
            "volatility_burst": vol_burst,
            "spread": spread,
            "toxicity_vpin": toxicity,
            "microprice_drift": mprice_drift,
            "is_event_window": False,
            "is_anomaly_time": False
        }

        # 2. Market State Detection
        current_state = self.detector.detect_state(state_features)

        # ML 모델에 들어갈 피처 벡터 (단순화)
        ml_features = np.array([obi, mprice_drift, spread, toxicity, vol_burst])
        self.history_features.append(ml_features)
        self.history_targets.append(self.last_mid)

        # 3. Online Learning Update
        if self.tick_count > self.target_lookahead:
            idx = self.tick_count - self.target_lookahead - 1
            past_features = self.history_features[idx].reshape(1, -1)
            past_mid = self.history_targets[idx]

            target_return = np.array([(self.last_mid - past_mid) / past_mid * 10000.0]) # in bps
            self.model.update(past_features, target_return)

        # 4. Model Prediction
        pred_return = self.model.predict(ml_features.reshape(1, -1))[0]

        if self.tick_count % 10 == 0:
            logger.info(f"[t={t:.1f}] Q: {bs:.1f}@{b:.2f} | {as_:.1f}@{a:.2f} | State: {current_state.name} | Pred: {pred_return:+.2f} bps")

        # 5. Pipeline Routing (Dynamic Execution)
        action: ExecutionAction = self.router.route_execution(current_state, pred_return)

        # 6. Cancel/Replace Execution Logic
        self.tracker.evaluate_cancel_replace(t, "BTC", b, a)

        # Execute based on routed action
        if action.action == "HALT":
            if self.active_order_id in self.engine.active_orders:
                self.engine.cancel_order(self.active_order_id, t)
                self.tracker.untrack_order(self.active_order_id)
                self.active_order_id = None

        elif action.action == "PASSIVE_MAKE" and not self.active_order_id:
            # Passive: 걸어두기만 함 (현재 Bid에 Limit 매수)
            if pred_return > 0.1:
                oid = self.engine.place_limit_order("BTC", "buy", b, 1.0 * action.size_multiplier, t)
                self.tracker.track_order(oid, "BTC", "buy", b, 1.0 * action.size_multiplier)
                self.active_order_id = oid

        elif action.action == "AGGRESSIVE_TAKE" and not self.active_order_id:
            # Aggressive: 상대 호가(Ask)를 직접 타격
            if pred_return > 0.5:
                oid = self.engine.place_limit_order("BTC", "buy", a, 1.0 * action.size_multiplier, t)
                self.tracker.track_order(oid, "BTC", "buy", a, 1.0 * action.size_multiplier)
                self.active_order_id = oid

        # 7. Process Market Matching
        self.engine.process_quote_update(t, b, a)

        # Clean up filled/canceled orders
        if self.active_order_id and self.active_order_id not in self.engine.active_orders:
            self.tracker.untrack_order(self.active_order_id)
            self.active_order_id = None

    def print_performance(self):
        val = self.engine.get_portfolio_value(self.last_mid)
        logger.info(f"Final Portfolio Value: {val:.2f} (Inventory: {self.engine.inventory})")
        logger.info(f"Total Limit Orders Placed: {self.engine.order_counter}")

def run_test():
    sim = AdvancedOnlineHFTSimulator()
    t = 1000.0

    logger.info("=== Starting Advanced Multi-Axis HFT Simulation ===")

    bid = 100.0
    for i in range(50):
        t += 10.0

        # 1. Trade 발생 시뮬레이션 (최근 체결)
        # i가 20 근처일 때 Toxic Flow (일방적 매도 체결 발생)를 인위적으로 생성
        if 20 <= i <= 25:
            sim.feed_trade(t - 1.0, bid - 0.1, 5.0, -1) # Sell trade
            spread_modifier = 0.2
        else:
            sim.feed_trade(t - 1.0, bid, 1.0, 1) # Normal Buy trade
            spread_modifier = 0.05

        # 2. Quote 업데이트 시뮬레이션
        if i % 5 == 0:
            bid += 0.1
        ask = bid + spread_modifier

        bs = 10.0 + np.random.rand() * 5.0
        as_ = 2.0 + np.random.rand() * 2.0

        sim.feed_quote(t, bid, bs, ask, as_)

    sim.print_performance()

if __name__ == "__main__":
    run_test()
