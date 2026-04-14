import numpy as np
import pandas as pd
from hmmlearn import hmm
from src.utils.logger import logger

class SimpleHMMRegimeDetector:
    """
    Simple Hidden Markov Model for Regime Detection (B007 / F023).
    Typically classifies regimes into Low Volatility / High Volatility based on returns and volatility.
    """
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", random_state=42)
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        if 'return_1d' not in df.columns or 'volatility_20d' not in df.columns:
            logger.warning("Missing required features for HMM (return_1d, volatility_20d)")
            return

        features = df[['return_1d', 'volatility_20d']].values
        self.model.fit(features)
        self.is_fitted = True
        logger.info("HMM Regime Detector fitted.")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            logger.warning("Model is not fitted yet.")
            return np.zeros(len(df))

        features = df[['return_1d', 'volatility_20d']].values
        regimes = self.model.predict(features)
        return regimes
