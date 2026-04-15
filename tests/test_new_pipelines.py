import unittest
import pandas as pd
import numpy as np

from src.envs.trading_env import TradingEnv
from src.models.rl import PPOModel, SACModel, DQNModel
from src.models.sde import OUProcess, HestonVolatility, NeuralSDEModel
from src.models.dl import DeepLearningModel

class TestTradingEnv(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'f1': np.random.randn(100),
            'f2': np.random.randn(100),
            'target_return': np.random.randn(100) * 0.01
        })
        self.features = ['f1', 'f2']

    def test_env_step_and_reset(self):
        env = TradingEnv(self.df, self.features, 'target_return')
        obs, info = env.reset()
        self.assertEqual(obs.shape, (2,))

        obs, reward, done, trunc, info = env.step(np.array([0.5]))
        self.assertIsInstance(reward, float)
        self.assertFalse(done)

class TestRLModels(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'f1': np.random.randn(10),
            'f2': np.random.randn(10),
            'target_return': np.random.randn(10) * 0.01
        })
        self.features = ['f1', 'f2']

    def test_ppo(self):
        model = PPOModel(self.features)
        model.fit(self.df, total_timesteps=10) # very short for testing
        preds = model.predict(self.df)
        self.assertEqual(len(preds), len(self.df))

    def test_sac(self):
        model = SACModel(self.features)
        model.fit(self.df, total_timesteps=10)
        preds = model.predict(self.df)
        self.assertEqual(len(preds), len(self.df))

    def test_dqn(self):
        model = DQNModel(self.features)
        model.fit(self.df, total_timesteps=10)
        preds = model.predict(self.df)
        self.assertEqual(len(preds), len(self.df))


class TestSDEModels(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'spread': np.random.randn(100) * 0.5,
            'close': np.cumprod(1 + np.random.randn(100) * 0.01) * 100
        })

    def test_ou_process(self):
        model = OUProcess()
        model.fit(self.df, feature='spread')
        preds = model.predict(self.df, feature='spread')
        self.assertEqual(len(preds), len(self.df))

    def test_heston_volatility(self):
        model = HestonVolatility()
        model.fit(self.df, price_col='close')
        preds = model.predict(self.df, price_col='close')
        self.assertEqual(len(preds), len(self.df))

    def test_neural_sde(self):
        model = NeuralSDEModel(state_size=1)
        model.fit(self.df, feature='close', epochs=2)
        preds = model.predict(self.df, feature='close')
        self.assertEqual(len(preds), len(self.df))


class TestDLModels(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'f1': np.random.randn(50),
            'f2': np.random.randn(50),
            'target_return': np.random.randn(50) * 0.01
        })
        self.features = ['f1', 'f2']

    def test_lstm(self):
        model = DeepLearningModel('LSTM', self.features, seq_len=5)
        model.fit(self.df, epochs=2)
        preds = model.predict(self.df)
        self.assertEqual(len(preds), len(self.df))

    def test_mlp(self):
        model = DeepLearningModel('MLP', self.features, seq_len=1)
        model.fit(self.df, epochs=2)
        preds = model.predict(self.df)
        self.assertEqual(len(preds), len(self.df))

    def test_transformer(self):
        model = DeepLearningModel('Transformer', self.features, seq_len=5)
        model.fit(self.df, epochs=2)
        preds = model.predict(self.df)
        self.assertEqual(len(preds), len(self.df))

    def test_autoencoder(self):
        model = DeepLearningModel('Autoencoder', self.features, seq_len=1)
        model.fit(self.df, epochs=2)
        preds = model.predict(self.df)
        self.assertEqual(len(preds), len(self.df))

    def test_tft(self):
        model = DeepLearningModel('TFT', self.features, seq_len=5)
        model.fit(self.df, epochs=2)
        preds = model.predict(self.df)
        self.assertEqual(len(preds), len(self.df))

    def test_patch_tst(self):
        model = DeepLearningModel('PatchTST', self.features, seq_len=4) # Multiple of patch_len=2
        model.fit(self.df, epochs=2)
        preds = model.predict(self.df)
        self.assertEqual(len(preds), len(self.df))

    def test_gnn(self):
        model = DeepLearningModel('GNN', self.features, seq_len=1)
        model.fit(self.df, epochs=2)
        preds = model.predict(self.df)
        self.assertEqual(len(preds), len(self.df))

    def test_multimodal(self):
        model = DeepLearningModel('Multimodal', self.features, seq_len=1)
        model.fit(self.df, epochs=2)
        preds = model.predict(self.df)
        self.assertEqual(len(preds), len(self.df))

if __name__ == '__main__':
    unittest.main()
