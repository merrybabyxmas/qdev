import unittest
import os
import json
from src.monitoring.control_plane import HFTControlPlane
from src.models.champion_registry import ChampionRegistry
from src.signals.router import PipelineRouter
from src.models.state_detector import MarketState

class TestControlPlaneAndRouting(unittest.TestCase):
    def setUp(self):
        self.policy_path = "artifacts/control_plane/test_policy.json"
        self.registry_path = "data/models/test_champion_registry.json"
        if os.path.exists(self.policy_path):
            os.remove(self.policy_path)
        if os.path.exists(self.registry_path):
            os.remove(self.registry_path)

    def tearDown(self):
        if os.path.exists(self.policy_path):
            os.remove(self.policy_path)
        if os.path.exists(self.registry_path):
            os.remove(self.registry_path)

    def test_control_plane_generation(self):
        cp = HFTControlPlane(policy_path=self.policy_path)

        symbols_config = {
            "BTC": {"enabled": True, "reason": "normal"},
            "ETH": {"enabled": False, "reason": "high_spread"}
        }
        thresholds = {"prediction_bps_min": 1.0}

        cp.generate_policy(
            regime="NORMAL",
            allow_hft=True,
            symbol_configs=symbols_config,
            global_thresholds=thresholds
        )

        policy = cp.read_policy()
        self.assertEqual(policy["regime"], "NORMAL")
        self.assertTrue(policy["allow_hft"])
        self.assertEqual(policy["symbols"]["ETH"]["enabled"], False)

    def test_champion_registry_multi_horizon(self):
        reg = ChampionRegistry(self.registry_path)

        reg.register_new_champion("hft_microstructure", {"model_id": "NEW-HFT", "sharpe": 3.0})
        champ_hft = reg.get_champion("hft_microstructure")
        self.assertEqual(champ_hft["model_id"], "NEW-HFT")

        champ_macro = reg.get_champion("macro_daily")
        self.assertEqual(champ_macro["model_id"], "M-BASE-LGBM-001") # Default macro

    def test_context_aware_routing(self):
        router = PipelineRouter()

        policy = {
            "allow_hft": True,
            "symbols": {
                "BTC": {"enabled": True},
                "ETH": {"enabled": False}
            },
            "thresholds": {"prediction_bps_min": 1.0}
        }

        # 1. BTC with allow_hft=True, enabled=True, and Trending State -> AGGRESSIVE_TAKE
        action1 = router.route_execution(MarketState.STABLE_TREND, prediction=1.5, policy=policy, symbol="BTC")
        self.assertEqual(action1.action, "AGGRESSIVE_TAKE")

        # 2. ETH with enabled=False -> HALT
        action2 = router.route_execution(MarketState.STABLE_TREND, prediction=1.5, policy=policy, symbol="ETH")
        self.assertEqual(action2.action, "HALT")

        # 3. BTC but Microstructure is TOXIC -> HALT
        action3 = router.route_execution(MarketState.HIGH_VOL_TOXIC, prediction=10.0, policy=policy, symbol="BTC")
        self.assertEqual(action3.action, "HALT")

if __name__ == '__main__':
    unittest.main()
