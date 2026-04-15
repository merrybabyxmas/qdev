import unittest
import numpy as np
from src.models.sgd_online import OnlineSGDRegressor
from src.execution.policy import ExecutionTracker
from src.backtest.matching_engine import HFTMatchingEngine

class TestOnlineHFTPipeline(unittest.TestCase):
    def test_sgd_online_learning(self):
        model = OnlineSGDRegressor()

        # Mock features: [OBI, Microprice_diff, Spread]
        X1 = np.array([[0.8, 0.01, 0.05]])
        y1 = np.array([5.0]) # 5 bps return

        model.update(X1, y1)
        self.assertTrue(model.is_fitted)

        pred1 = model.predict(X1)

        X2 = np.array([[-0.5, -0.01, 0.05]])
        y2 = np.array([-3.0])
        model.update(X2, y2)

        pred2 = model.predict(X2)
        # Should predict differently for X2
        self.assertNotEqual(pred1[0], pred2[0])

    def test_cancel_replace_policy(self):
        engine = HFTMatchingEngine(latency_ms=10.0, fee_bps=0.0)
        # Threshold: 100 bps = 1%
        tracker = ExecutionTracker(broker_or_engine=engine, cancel_threshold_bps=100.0)

        # Place buy limit order at 100.0
        oid = engine.place_limit_order("BTC", "buy", 100.0, 1.0, current_time_ms=100.0)
        tracker.track_order(oid, "BTC", "buy", 100.0, 1.0)

        self.assertIn(oid, engine.active_orders)
        self.assertIn(oid, tracker.active_orders)

        # Market bid moves up slightly to 100.5 (Drift = 0.5% < 1.0%)
        # Should NOT cancel
        tracker.evaluate_cancel_replace(150.0, "BTC", current_bid=100.5, current_ask=101.0)
        self.assertIn(oid, engine.active_orders)

        # Market bid moves up significantly to 101.5 (Drift = 1.5% > 1.0%)
        # Should CANCEL and REPLACE
        tracker.evaluate_cancel_replace(200.0, "BTC", current_bid=101.5, current_ask=102.0)

        self.assertNotIn(oid, engine.active_orders)
        self.assertNotIn(oid, tracker.active_orders)

        # There should be exactly one new order active now at the new price 101.5
        self.assertEqual(len(engine.active_orders), 1)
        new_oid = list(engine.active_orders.keys())[0]
        self.assertEqual(engine.active_orders[new_oid]["price"], 101.5)

if __name__ == '__main__':
    unittest.main()
