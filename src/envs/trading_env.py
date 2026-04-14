import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional

class TradingEnv(gym.Env):
    """
    A minimal Trading Environment for RL models.
    Observation space: The features available at the current time step.
    Action space: A continuous value between -1 and 1 representing the portfolio weight or position size.
    """
    def __init__(self, df: pd.DataFrame, features: list[str], target: str = 'target_return'):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.features = features
        self.target = target

        self.current_step = 0
        self.max_step = len(self.df) - 1

        # Action space: Continuous allocation weight [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation space: the available features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.features),), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        if self.current_step >= self.max_step:
            return self._get_observation(), 0.0, True, False, {}

        # The reward is the allocation weight multiplied by the next step's return
        current_return = self.df.loc[self.current_step, self.target]
        weight = action[0]
        reward = float(weight * current_return)

        self.current_step += 1

        done = self.current_step >= self.max_step
        obs = self._get_observation()

        return obs, reward, done, False, {}

    def _get_observation(self) -> np.ndarray:
        obs = self.df.loc[self.current_step, self.features].values
        return obs.astype(np.float32)
