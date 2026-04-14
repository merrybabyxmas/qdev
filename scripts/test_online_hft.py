import asyncio
import logging
import numpy as np
from src.state.ring_buffers import TickRingBuffer, QuoteRingBuffer
from src.features.microstructure.imbalance import compute_order_book_imbalance, compute_microprice, compute_spread
from src.backtest.matching_engine import HFTMatchingEngine
from src.models.sgd_online import OnlineSGDRegressor
from src.execution.policy import ExecutionTracker
from src.utils.logger import logger

# Configure simple logging for fast console output
logging.getLogger("src").setLevel(logging.INFO)

class OnlineHFTSimulator:
    """
    온라인 학습(SGDRegressor)과 취소/대체(Cancel/Replace) 로직이 포함된 HFT 파이프라인 검증 시뮬레이터.
    실시간으로 수백 틱이 들어올 때마다 모델이 즉시 학습하며, 호가창을 따라다니면서 주문을 수정함.
    """
    def __init__(self):
        self.engine = HFTMatchingEngine(latency_ms=5.0, fee_bps=0.0001)
        self.tracker = ExecutionTracker(broker_or_engine=self.engine, cancel_threshold_bps=1.0)
        self.model = OnlineSGDRegressor(learning_rate='constant', eta0=0.01)

        self.history_features = []
        self.history_targets = []
        self.last_mid = 100.0
        self.tick_count = 0
        self.active_order_id = None
        self.target_lookahead = 5 # 5틱 이후의 수익률을 예측 목표로 삼음

    def feed_quote(self, t: float, b: float, bs: float, a: float, as_: float):
        self.tick_count += 1
        self.last_mid = (b + a) / 2.0

        # 1. Feature Computation
        obi = compute_order_book_imbalance(bs, as_)
        mprice = compute_microprice(b, bs, a, as_)
        spread = compute_spread(b, a)

        # 간단한 피처 벡터 [OBI, Microprice 괴리율, Spread]
        features = np.array([obi, (mprice - self.last_mid)/self.last_mid, spread])
        self.history_features.append(features)
        self.history_targets.append(self.last_mid) # 현재 Mid 저장 (이후 수익률 계산용)

        # 2. Online Learning Update (과거 데이터의 타깃이 완성되었으면 학습)
        if self.tick_count > self.target_lookahead:
            idx = self.tick_count - self.target_lookahead - 1
            past_features = self.history_features[idx].reshape(1, -1)
            past_mid = self.history_targets[idx]
            current_mid = self.last_mid

            # Target: 5틱 동안의 미래 중간가 수익률
            target_return = np.array([(current_mid - past_mid) / past_mid * 10000.0]) # in bps
            self.model.update(past_features, target_return)

        # 3. Model Prediction
        pred_return = self.model.predict(features.reshape(1, -1))[0]

        # 4. Cancel/Replace Execution Logic
        # 모델이 단기적으로 강한 상승을 예측하면 매수, 강한 하락을 예측하면 매도
        # 만약 기존 포지션과 반대되거나, 시장 호가가 변했다면 Tracker가 알아서 취소/재주문(Cancel/Replace)을 처리함

        if self.tick_count % 10 == 0:
            logger.info(f"[t={t:.1f}] Q: {bs}@{b} | {as_}@{a} | OBI: {obi:+.2f} | Pred: {pred_return:+.2f} bps")

        # 추적 중인 모든 주문에 대해 호가 변동(Drift)을 모니터링하여 Replace 수행
        self.tracker.evaluate_cancel_replace(t, "BTC", b, a)

        if pred_return > 1.0 and not self.active_order_id:
            # 강한 매수 시그널 발생 시 현재 최우선 매수호가(Best Bid)에 지정가(Passive) 오더 배치
            oid = self.engine.place_limit_order("BTC", "buy", b, 1.0, t)
            self.tracker.track_order(oid, "BTC", "buy", b, 1.0)
            self.active_order_id = oid

        # 5. Process Market Matching
        self.engine.process_quote_update(t, b, a)

        # 체결 여부 확인 후 State 관리
        if self.active_order_id and self.active_order_id not in self.engine.active_orders:
            self.tracker.untrack_order(self.active_order_id)
            self.active_order_id = None # 체결되었거나 취소됨

    def print_performance(self):
        val = self.engine.get_portfolio_value(self.last_mid)
        logger.info(f"Final Portfolio Value: {val:.2f} (Inventory: {self.engine.inventory})")
        logger.info(f"Total Limit Orders Placed: {self.engine.order_counter}")

def run_test():
    sim = OnlineHFTSimulator()
    t = 1000.0

    logger.info("=== Starting Online HFT Simulation ===")

    # 모의 틱 생성: 가격이 서서히 오르면서 호가창이 상승(매수세 우위 OBI>0)하는 상황 시뮬레이션
    bid = 100.0
    for i in range(50):
        t += 10.0

        # 임의의 시장 노이즈 + 추세
        if i % 5 == 0:
            bid += 0.1
        ask = bid + 0.1

        # 추세가 있을 때 OBI가 매수에 치우치도록 시뮬레이션
        bs = 10.0 + np.random.rand() * 5.0
        as_ = 2.0 + np.random.rand() * 2.0

        sim.feed_quote(t, bid, bs, ask, as_)

    sim.print_performance()

if __name__ == "__main__":
    run_test()
