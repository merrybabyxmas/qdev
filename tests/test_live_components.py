import unittest
import os
from unittest.mock import MagicMock, patch
from src.brokers.alpaca_broker import AlpacaBroker
from src.models.champion_registry import ChampionRegistry
from src.risk.manager import RiskManager

class TestLiveComponents(unittest.TestCase):
    def setUp(self):
        self.test_reg_path = "data/models/test_champion_registry.json"
        if os.path.exists(self.test_reg_path):
            os.remove(self.test_reg_path)

    def tearDown(self):
        if os.path.exists(self.test_reg_path):
            os.remove(self.test_reg_path)

    @patch('src.brokers.alpaca_broker.TradingClient')
    def test_alpaca_broker_connect(self, MockClient):
        mock_instance = MockClient.return_value
        mock_instance.get_account.return_value.buying_power = "100000.0"

        broker = AlpacaBroker("key", "sec", paper=True)
        broker.connect()

        self.assertTrue(broker.connected)
        mock_instance.get_account.assert_called_once()

    def test_champion_registry(self):
        reg = ChampionRegistry(self.test_reg_path)
        champ = reg.get_champion()

        self.assertEqual(champ["model_id"], "M-BASE-LGBM-001")

        new_champ = {
            "model_id": "M-PRO-LGBM-002",
            "sharpe": 2.5
        }
        reg.register_new_champion(new_champ)

        updated_champ = reg.get_champion()
        self.assertEqual(updated_champ["model_id"], "M-PRO-LGBM-002")
        self.assertEqual(updated_champ["sharpe"], 2.5)

    def test_usd_position_sizing(self):
        manager = RiskManager(max_position_cap=0.50, max_drawdown=0.10)

        equity = 100000.0
        target_weight = 0.20
        current_qty = 0.0
        btc_price = 50000.0

        delta = manager.calculate_order_qty("BTC", target_weight, current_qty, btc_price, equity)
        self.assertAlmostEqual(delta, 0.4, places=4)

        current_qty = 0.5
        delta2 = manager.calculate_order_qty("BTC", target_weight, current_qty, btc_price, equity)
        self.assertAlmostEqual(delta2, -0.1, places=4)

        manager.evaluate_account_risk(100000.0)
        manager.evaluate_account_risk(80000.0)

        delta3 = manager.calculate_order_qty("BTC", target_weight, current_qty, btc_price, 80000.0)
        self.assertAlmostEqual(delta3, -0.5, places=4)

if __name__ == '__main__':
    unittest.main()
