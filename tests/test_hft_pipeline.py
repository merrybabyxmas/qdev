import unittest
import numpy as np
from src.state.ring_buffers import TickRingBuffer, QuoteRingBuffer
from src.features.microstructure.imbalance import compute_order_book_imbalance, compute_microprice, compute_spread, compute_trade_intensity
from src.backtest.matching_engine import HFTMatchingEngine

class TestHFTPipeline(unittest.TestCase):
    def test_ring_buffers(self):
        buf = TickRingBuffer(capacity=5)
        for i in range(10):
            buf.append(float(i), 100.0, 1.0, 1)

        recent = buf.get_recent(3)
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[-1][0], 9.0)

    def test_microstructure_features(self):
        obi = compute_order_book_imbalance(10.0, 2.0)
        self.assertAlmostEqual(obi, 8/12)

        mprice = compute_microprice(100.0, 10.0, 100.5, 2.0)
        self.assertAlmostEqual(mprice, (100.0 * 2.0 + 100.5 * 10.0) / 12.0)

        spread = compute_spread(100.0, 100.5)
        self.assertAlmostEqual(spread, 0.5)

        ticks = np.array([
            [1000.0, 100.0, 5.0, 1],
            [1500.0, 100.1, 3.0, -1],
            [2500.0, 100.2, 2.0, 1]
        ])
        # Window of 1000ms from 2500 -> keeps only 1500 and 2500
        intensity = compute_trade_intensity(ticks, window_ms=1000.0)
        self.assertEqual(intensity, 5.0)

    def test_matching_engine(self):
        engine = HFTMatchingEngine(latency_ms=10.0, fee_bps=0.0)

        # Place buy order at t=100
        engine.place_limit_order("BTC", "buy", price=100.0, size=1.0, current_time_ms=100.0)

        # Order shouldn't be active yet at t=105 due to latency
        engine.process_quote_update(105.0, bid=99.9, ask=100.0)
        self.assertEqual(engine.inventory, 0.0)

        # Order should be active and filled at t=115 when Ask hits 100.0
        engine.process_quote_update(115.0, bid=99.9, ask=100.0)
        self.assertEqual(engine.inventory, 1.0)
        self.assertEqual(engine.cash, 100000.0 - 100.0)

if __name__ == '__main__':
    unittest.main()
