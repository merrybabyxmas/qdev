import unittest
import pandas as pd
import numpy as np
from src.features.builder import build_technical_features
from src.models.hmm import SimpleHMMRegimeDetector
from src.models.lgbm import LightGBMRanker
from src.risk.manager import RiskManager
from src.strategies.ml_strategy import MLStrategy

class TestTradingSystem(unittest.TestCase):
    def setUp(self):
        # Create dummy data for testing
        dates = pd.date_range(start='2023-01-01', periods=50)
        self.df = pd.DataFrame({
            'open': np.random.rand(50) * 100 + 100,
            'high': np.random.rand(50) * 100 + 110,
            'low': np.random.rand(50) * 100 + 90,
            'close': np.random.rand(50) * 100 + 100,
            'volume': np.random.rand(50) * 1000
        }, index=dates)

    def test_features(self):
        feat_df = build_technical_features(self.df)
        self.assertTrue('SMA_20' in feat_df.columns)
        self.assertTrue('EMA_20' in feat_df.columns)
        self.assertTrue('return_1d' in feat_df.columns)
        self.assertTrue('return_5d' in feat_df.columns)
        self.assertTrue('volatility_20d' in feat_df.columns)

    def test_hmm(self):
        feat_df = build_technical_features(self.df)
        hmm = SimpleHMMRegimeDetector(n_components=2)
        hmm.fit(feat_df)
        preds = hmm.predict(feat_df)
        self.assertEqual(len(preds), len(feat_df))

    def test_lgbm(self):
        feat_df = build_technical_features(self.df)
        feat_df['target_return'] = feat_df['return_1d'].shift(-1)
        feat_df = feat_df.dropna()

        lgbm = LightGBMRanker()
        lgbm.fit(feat_df)
        preds = lgbm.predict(feat_df)
        self.assertEqual(len(preds), len(feat_df))

    def test_risk_manager(self):
        manager = RiskManager(max_position_cap=0.30)
        weights = {"AAPL": 0.50, "MSFT": 0.20}
        capped = manager.apply_position_caps(weights)
        self.assertEqual(capped["AAPL"], 0.30)
        self.assertEqual(capped["MSFT"], 0.20)

    def test_ml_strategy(self):
        strategy = MLStrategy(symbols=["AAPL", "MSFT", "GOOG"])
        preds = {"AAPL": 0.05, "MSFT": -0.01, "GOOG": 0.02}
        weights = strategy.generate_weights(preds)

        # Only AAPL and GOOG are positive
        self.assertEqual(weights["AAPL"], 0.5)
        self.assertEqual(weights["GOOG"], 0.5)
        self.assertEqual(weights["MSFT"], 0.0)

if __name__ == '__main__':
    unittest.main()
