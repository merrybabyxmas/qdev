import unittest
from src.models.state_detector import MarketStateDetector, MarketState
from src.signals.router import PipelineRouter, ExecutionAction

class TestStateRouter(unittest.TestCase):
    def setUp(self):
        # wide_spread_threshold_bps=30: 30 bps 초과 시 WIDE_SPREAD_ILLIQUID
        self.detector = MarketStateDetector(high_vol_threshold=0.5, wide_spread_threshold_bps=30.0, toxic_vpin_threshold=0.8, trend_threshold=0.001)
        self.router = PipelineRouter()
        self.default_policy = {
            "allow_hft": True,
            "symbols": {"BTC": {"enabled": True}}
        }
        # 정상 장 features: mid=$100, spread=$0.10 → 10 bps (30 bps 미만 → 유동성 정상)
        self._normal_spread = {"spread": 0.10, "mid_price": 100.0}

    def test_hard_rules_event_shock(self):
        features = {"is_event_window": True}
        state = self.detector.detect_state(features)
        self.assertEqual(state, MarketState.EVENT_SHOCK)

        action = self.router.route_execution(state, prediction=10.0, policy=self.default_policy, symbol="BTC")
        self.assertEqual(action.action, "HALT")

    def test_high_vol_toxic(self):
        features = {
            "volatility_burst": 0.8,
            "toxicity_vpin": 0.9,
            "microprice_drift": 0.0,
            **self._normal_spread,  # spread 10 bps → 비유동성 아님
        }
        state = self.detector.detect_state(features)
        self.assertEqual(state, MarketState.HIGH_VOL_TOXIC)

        action = self.router.route_execution(state, prediction=10.0, policy=self.default_policy, symbol="BTC")
        self.assertEqual(action.action, "HALT")

    def test_stable_trend_aggressive(self):
        features = {
            "volatility_burst": 0.2,
            "toxicity_vpin": 0.2,
            "microprice_drift": 0.005,  # > 0.001
            **self._normal_spread,
        }
        state = self.detector.detect_state(features)
        self.assertEqual(state, MarketState.STABLE_TREND)

        action = self.router.route_execution(state, prediction=1.0, policy=self.default_policy, symbol="BTC")
        self.assertEqual(action.action, "AGGRESSIVE_TAKE")

    def test_stable_mean_reversion_passive(self):
        features = {
            "volatility_burst": 0.2,
            "toxicity_vpin": 0.2,
            "microprice_drift": 0.0005,  # <= 0.001
            **self._normal_spread,
        }
        state = self.detector.detect_state(features)
        self.assertEqual(state, MarketState.STABLE_MEAN_REVERSION)

        action = self.router.route_execution(state, prediction=0.5, policy=self.default_policy, symbol="BTC")
        self.assertEqual(action.action, "PASSIVE_MAKE")

if __name__ == '__main__':
    unittest.main()
