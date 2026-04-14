from _bootstrap import ensure_project_root

ensure_project_root()

import asyncio
import logging
from src.state.ring_buffers import TickRingBuffer, QuoteRingBuffer
from src.features.microstructure.imbalance import compute_order_book_imbalance, compute_microprice, compute_spread
from src.backtest.matching_engine import HFTMatchingEngine
from src.utils.logger import logger

# Configure simple logging for fast console output
logging.getLogger("src").setLevel(logging.INFO)

class HFTSimulator:
    """
    모의 틱/호가 데이터를 이용한 HFT 파이프라인 검증용 스크립트.
    """
    def __init__(self):
        self.trade_buf = TickRingBuffer(capacity=100)
        self.quote_buf = QuoteRingBuffer(capacity=100)
        self.engine = HFTMatchingEngine(latency_ms=10.0, fee_bps=0.0001)
        self.last_mid = 100.0

    def feed_quote(self, t: float, b: float, bs: float, a: float, as_: float):
        self.quote_buf.append(t, b, bs, a, as_)
        self.last_mid = (b + a) / 2.0

        # 1. Feature Computation
        obi = compute_order_book_imbalance(bs, as_)
        mprice = compute_microprice(b, bs, a, as_)
        spread = compute_spread(b, a)

        logger.info(f"[t={t:.1f}] Q: {bs}@{b} | {as_}@{a} | OBI: {obi:+.2f} | μPrice: {mprice:.2f}")

        # 2. Execute Strategy (E001 Logic: LOB Imbalance Baseline)
        # 만약 OBI가 매우 강한 매수(>0.8)이고 스프레드가 좁다면 Market Making Passive Quoting
        if obi > 0.8 and spread <= 0.1:
            # 예상 적정가가 상승 중이므로 Bid 쪽에 패시브 오더
            self.engine.place_limit_order("BTC", "buy", b, 1.0, t)
        elif obi < -0.8 and spread <= 0.1:
            # 하락 중이므로 Ask 쪽에 패시브 매도
            self.engine.place_limit_order("BTC", "sell", a, 1.0, t)

        # 3. Process Execution
        self.engine.process_quote_update(t, b, a)

    def print_performance(self):
        val = self.engine.get_portfolio_value(self.last_mid)
        logger.info(f"Final Portfolio Value: {val:.2f} (Inventory: {self.engine.inventory})")

def run_test():
    sim = HFTSimulator()

    # 모의 시간 틱 (ms)
    t = 1000.0

    # 1. 초기 상태
    sim.feed_quote(t, 100.0, 1.0, 100.1, 1.0)

    # 2. 강한 매수세 발생 (Ask 물량 소진 중)
    t += 50.0
    sim.feed_quote(t, 100.0, 10.0, 100.1, 1.0) # OBI = (10-1)/(10+1) = 0.81 -> Buy limit at 100.0 placed

    # 3. 시장 호가 하락 -> 내 매수 오더가 체결될 수 있음
    t += 50.0
    sim.feed_quote(t, 99.9, 5.0, 100.0, 2.0) # Ask가 100.0으로 내려옴 -> 내가 100.0에 건 매수 오더 체결

    # 4. 강한 매도세 발생 (Bid 물량 소진 중)
    t += 50.0
    sim.feed_quote(t, 99.9, 1.0, 100.0, 10.0) # OBI = (1-10)/(1+10) = -0.81 -> Sell limit at 100.0 placed

    # 5. 시장 호가 상승 -> 내 매도 오더가 체결될 수 있음
    t += 50.0
    sim.feed_quote(t, 100.0, 5.0, 100.1, 5.0) # Bid가 100.0으로 올라옴 -> 내가 100.0에 건 매도 오더 체결

    sim.print_performance()

if __name__ == "__main__":
    run_test()
