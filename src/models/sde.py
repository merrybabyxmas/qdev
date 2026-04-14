import numpy as np
import pandas as pd
import torch
from torch import nn
import torchsde
from typing import Optional
from src.utils.logger import logger

class OUProcess:
    """
    S001: OU Mean Reversion Signal.
    Simulates and predicts using an Ornstein-Uhlenbeck process.
    dX_t = theta * (mu - X_t) dt + sigma * dW_t
    """
    def __init__(self, theta: float = 0.5, mu: float = 0.0, sigma: float = 0.1):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = 1.0 / 252.0  # Assumes daily frequency

    def fit(self, df: pd.DataFrame, feature: str = 'spread'):
        """Simple least-squares calibration for OU parameters."""
        if feature not in df.columns:
            logger.warning(f"Feature {feature} not found for OUProcess fit.")
            return

        x = df[feature].values[:-1]
        y = df[feature].values[1:]

        # Linear regression: X_{t+1} = a * X_t + b + epsilon
        # where a = exp(-theta*dt), b = mu*(1 - exp(-theta*dt))
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]

        self.theta = -np.log(m) / self.dt if m > 0 and m < 1 else self.theta
        self.mu = c / (1 - m) if m != 1 else self.mu

        residuals = y - (m * x + c)
        self.sigma = np.std(residuals) / np.sqrt(self.dt)
        logger.info(f"OUProcess calibrated: theta={self.theta:.3f}, mu={self.mu:.3f}, sigma={self.sigma:.3f}")

    def predict(self, df: pd.DataFrame, feature: str = 'spread') -> np.ndarray:
        if feature not in df.columns:
            return np.zeros(len(df))

        # Predict the next step deviation from mean as a simple mean reversion signal
        # Expected change in dt: E[X_{t+dt} - X_t] = (mu - X_t) * (1 - exp(-theta*dt))
        current_vals = df[feature].values
        expected_change = (self.mu - current_vals) * (1 - np.exp(-self.theta * self.dt))
        return expected_change


class HestonVolatility:
    """
    S003: Stochastic Volatility Timing Model (Heston-like proxy).
    Uses a basic localized variance estimator or simulated paths to project vol.
    """
    def __init__(self, kappa: float = 1.0, theta: float = 0.04, xi: float = 0.2, rho: float = -0.5):
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.dt = 1.0 / 252.0

    def fit(self, df: pd.DataFrame, price_col: str = 'close'):
        logger.info("Skipping rigorous Heston MLE for brevity. Using defaults.")
        pass

    def predict(self, df: pd.DataFrame, price_col: str = 'close') -> np.ndarray:
        if price_col not in df.columns:
            return np.zeros(len(df))

        returns = df[price_col].pct_change().fillna(0).values
        var = np.zeros_like(returns)
        var[0] = self.theta

        for t in range(1, len(returns)):
            # Simple discrete approximation for variance state
            var[t] = var[t-1] + self.kappa * (self.theta - var[t-1]) * self.dt
            # add a random shock to simulate the diffusion part of variance
            var[t] += self.xi * np.sqrt(max(var[t-1], 0)) * np.random.normal() * np.sqrt(self.dt)
            var[t] = max(var[t], 1e-6)

        return np.sqrt(var)


class NeuralSDEDriftDiffusion(nn.Module):
    """
    Core PyTorch module defining the drift and diffusion functions for torchsde.
    """
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, state_size: int = 1):
        super().__init__()
        self.drift_net = nn.Sequential(
            nn.Linear(state_size, 16),
            nn.Tanh(),
            nn.Linear(16, state_size)
        )
        self.diffusion_net = nn.Sequential(
            nn.Linear(state_size, 16),
            nn.Tanh(),
            nn.Linear(16, state_size),
            nn.Softplus() # Diffusion must be non-negative
        )

    def f(self, t, y):
        return self.drift_net(y)

    def g(self, t, y):
        return self.diffusion_net(y)

class NeuralSDEModel:
    """
    S007/S008: Neural SDE Return/Volatility Forecaster.
    Wraps the torchsde forward simulation.
    """
    def __init__(self, state_size: int = 1):
        self.state_size = state_size
        self.model = NeuralSDEDriftDiffusion(state_size=state_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, feature: str = 'close', epochs: int = 5):
        if feature not in df.columns:
            return

        logger.info("Fitting Neural SDE Model (simplified MSE loss)...")
        data = torch.tensor(df[feature].values, dtype=torch.float32).unsqueeze(1)
        t_size = len(data)
        ts = torch.linspace(0, 1, t_size)

        # Batched processing to prevent OOM
        batch_size = min(32, t_size)

        self.model.train()
        for epoch in range(epochs):
            for i in range(0, t_size - batch_size, batch_size):
                self.optimizer.zero_grad()

                batch_data = data[i:i+batch_size]
                batch_ts = torch.linspace(0, 1, batch_size)
                y0 = batch_data[0].unsqueeze(0)

                # Predict path using SDE solver
                preds = torchsde.sdeint(self.model, y0, batch_ts, dt=1.0/batch_size)
                preds = preds.squeeze(1)

                loss = nn.MSELoss()(preds, batch_data)
                loss.backward()
                self.optimizer.step()

        self.is_fitted = True
        logger.info("Neural SDE Model fitted.")

    def predict(self, df: pd.DataFrame, feature: str = 'close') -> np.ndarray:
        if not self.is_fitted or feature not in df.columns:
            return np.zeros(len(df))

        self.model.eval()
        data = torch.tensor(df[feature].values, dtype=torch.float32).unsqueeze(1)
        t_size = len(data)

        preds_all = []
        batch_size = min(32, t_size)

        with torch.no_grad():
            for i in range(0, t_size, batch_size):
                curr_batch_size = min(batch_size, t_size - i)
                if curr_batch_size <= 1:
                    if len(preds_all) > 0:
                        preds_all.append(preds_all[-1])
                    else:
                        preds_all.append(data[i].numpy())
                    break

                batch_data = data[i:i+curr_batch_size]
                batch_ts = torch.linspace(0, 1, curr_batch_size)
                y0 = batch_data[0].unsqueeze(0)

                preds = torchsde.sdeint(self.model, y0, batch_ts, dt=1.0/curr_batch_size)
                preds_all.extend(preds.squeeze(1).squeeze().numpy().tolist())

        # Ensure length matches
        preds_out = np.array(preds_all)
        if len(preds_out) < t_size:
            preds_out = np.pad(preds_out, (0, t_size - len(preds_out)), 'edge')
        elif len(preds_out) > t_size:
            preds_out = preds_out[:t_size]

        return preds_out
