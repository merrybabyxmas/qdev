import unittest
from unittest.mock import MagicMock
from src.ingestion.websocket_client import MultiSymbolHFTStreamManager
from src.models.ranker_engine import RealTimeCrossSectionalRanker

class TestMultiStreamAndRanker(unittest.TestCase):
    def test_ranker_engine(self):
        symbols = ["BTC", "ETH"]
        ranker = RealTimeCrossSectionalRanker(symbols, target_lookahead=2)

        feature_btc = {
            "timestamp": 1000.0, "bid": 100.0, "bid_size": 10.0, "ask": 100.1, "ask_size": 2.0,
            "microprice": 100.05, "obi": 0.8, "spread": 0.1, "intensity": 5.0,
            "toxicity_vpin": 0.9, "volatility_burst": 0.5, "mid_price": 100.05
        }

        feature_eth = {
            "timestamp": 1000.0, "bid": 50.0, "bid_size": 2.0, "ask": 50.1, "ask_size": 10.0,
            "microprice": 50.05, "obi": -0.8, "spread": 0.1, "intensity": 1.0,
            "toxicity_vpin": 0.1, "volatility_burst": 0.1, "mid_price": 50.05
        }

        preds_btc, lgbm_btc, w_btc = ranker.update_and_predict("BTC", feature_btc)
        self.assertIn("BTC", preds_btc)
        self.assertIn("BTC", lgbm_btc)
        self.assertEqual(len(w_btc), 2)

        preds_eth, lgbm_eth, w_eth = ranker.update_and_predict("ETH", feature_eth)
        self.assertIn("ETH", preds_eth)
        self.assertIn("ETH", lgbm_eth)

if __name__ == '__main__':
    unittest.main()
