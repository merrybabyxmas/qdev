import numpy as np
import pandas as pd
from typing import Optional
from pathlib import Path
import joblib
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from src.envs.trading_env import TradingEnv
from src.utils.logger import logger

class BaseModel:
    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path):
        raise NotImplementedError


class PPOModel(BaseModel):
    """
    R002: PPO Allocation Agent.
    """
    def __init__(self, features: list[str], target: str = 'target_return'):
        self.features = features
        self.target = target
        self.model = None
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, total_timesteps: int = 10000):
        logger.info("Fitting PPO Model...")
        env = DummyVecEnv([lambda: TradingEnv(df, self.features, self.target)])
        self.model = PPO("MlpPolicy", env, verbose=0)
        self.model.learn(total_timesteps=total_timesteps)
        self.is_fitted = True
        logger.info("PPO Model fitted.")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or df.empty:
            return np.zeros(len(df), dtype=float)

        env = DummyVecEnv([lambda: TradingEnv(df, self.features, self.target)])
        obs = env.reset()
        predictions = []

        for _ in range(len(df)):
            action, _states = self.model.predict(obs, deterministic=True)
            predictions.append(action[0][0])
            obs, _, done, _ = env.step(action)
            if done[0] and len(predictions) < len(df):
                # If environment finishes early, pad the rest with zeros
                predictions.extend([0.0] * (len(df) - len(predictions)))
                break

        return np.array(predictions)

class SACModel(BaseModel):
    """
    R003: SAC Portfolio Allocation Agent.
    """
    def __init__(self, features: list[str], target: str = 'target_return'):
        self.features = features
        self.target = target
        self.model = None
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, total_timesteps: int = 10000):
        logger.info("Fitting SAC Model...")
        env = DummyVecEnv([lambda: TradingEnv(df, self.features, self.target)])
        self.model = SAC("MlpPolicy", env, verbose=0)
        self.model.learn(total_timesteps=total_timesteps)
        self.is_fitted = True
        logger.info("SAC Model fitted.")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or df.empty:
            return np.zeros(len(df), dtype=float)

        env = DummyVecEnv([lambda: TradingEnv(df, self.features, self.target)])
        obs = env.reset()
        predictions = []

        for _ in range(len(df)):
            action, _states = self.model.predict(obs, deterministic=True)
            predictions.append(action[0][0])
            obs, _, done, _ = env.step(action)
            if done[0] and len(predictions) < len(df):
                # If environment finishes early, pad the rest with zeros
                predictions.extend([0.0] * (len(df) - len(predictions)))
                break

        return np.array(predictions)

class DQNModel(BaseModel):
    """
    R001: DQN Directional Trading Agent.
    DQN requires discrete action space, so we use a discrete wrapper over our environment.
    """
    def __init__(self, features: list[str], target: str = 'target_return'):
        self.features = features
        self.target = target
        self.model = None
        self.is_fitted = False

    def _make_discrete_env(self, df: pd.DataFrame):
        import gymnasium as gym
        from gymnasium import spaces
        from src.envs.trading_env import TradingEnv

        class DiscreteActionEnv(gym.ActionWrapper):
            def __init__(self, env):
                super().__init__(env)
                # 3 actions: Short (-1), Neutral (0), Long (1)
                self.action_space = spaces.Discrete(3)

            def action(self, act):
                mapping = {0: -1.0, 1: 0.0, 2: 1.0}
                return np.array([mapping[act]], dtype=np.float32)

        env = TradingEnv(df, self.features, self.target)
        return DiscreteActionEnv(env)

    def fit(self, df: pd.DataFrame, total_timesteps: int = 10000):
        logger.info("Fitting DQN Model...")
        env = DummyVecEnv([lambda: self._make_discrete_env(df)])
        self.model = DQN("MlpPolicy", env, verbose=0)
        self.model.learn(total_timesteps=total_timesteps)
        self.is_fitted = True
        logger.info("DQN Model fitted.")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or df.empty:
            return np.zeros(len(df), dtype=float)

        env = DummyVecEnv([lambda: self._make_discrete_env(df)])
        obs = env.reset()
        predictions = []

        mapping = {0: -1.0, 1: 0.0, 2: 1.0}

        for _ in range(len(df)):
            action, _states = self.model.predict(obs, deterministic=True)
            predictions.append(mapping[action[0]])
            obs, _, done, _ = env.step(action)
            if done[0] and len(predictions) < len(df):
                # If environment finishes early, pad the rest with zeros
                predictions.extend([0.0] * (len(df) - len(predictions)))
                break

        return np.array(predictions)
