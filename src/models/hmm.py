import numpy as np
import pandas as pd
from pathlib import Path

import joblib
from hmmlearn import hmm
from src.utils.logger import logger

class SimpleHMMRegimeDetector:
    """
    Simple Hidden Markov Model for Regime Detection (B007 / F023).
    Typically classifies regimes into Low Volatility / High Volatility based on returns and volatility.
    """
    def __init__(self, n_components: int = 2, n_iter: int = 25):
        self.n_components = n_components
        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=n_iter,
            tol=1e-2,
            random_state=42,
        )
        self.is_fitted = False
        self.features = ['return_1d', 'volatility_20d']

    def fit(self, df: pd.DataFrame):
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            logger.warning(f"Missing required features for HMM: {missing}")
            return

        features = df[self.features].values
        self.model.fit(features)
        self.is_fitted = True
        logger.info("HMM Regime Detector fitted.")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if df.empty:
            return np.empty(0, dtype=int)

        if not self.is_fitted:
            logger.warning("Model is not fitted yet.")
            return np.zeros(len(df), dtype=int)

        missing = [f for f in self.features if f not in df.columns]
        if missing:
            logger.warning(f"Missing required features for HMM prediction: {missing}")
            return np.zeros(len(df), dtype=int)

        features = df[self.features].values
        regimes = self.model.predict(features)
        return regimes

    def save(self, path) -> None:
        payload = {
            "n_components": self.n_components,
            "n_iter": self.model.n_iter,
            "model": self.model,
            "is_fitted": self.is_fitted,
            "features": self.features,
        }
        joblib.dump(payload, Path(path))

    @classmethod
    def load(cls, path) -> "SimpleHMMRegimeDetector":
        payload = joblib.load(Path(path))
        obj = cls(n_components=payload.get("n_components", 2), n_iter=payload.get("n_iter", 25))
        obj.model = payload["model"]
        obj.is_fitted = payload.get("is_fitted", False)
        obj.features = payload.get("features", obj.features)
        return obj
