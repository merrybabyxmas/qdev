import lightgbm as lgb
import pandas as pd
import numpy as np
from src.utils.logger import logger

class LightGBMRanker:
    """
    F001: Technical Indicator + LightGBM Ranker.
    Predicts cross-sectional ranking scores.
    """
    def __init__(self):
        self.model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.is_fitted = False
        self.features = ['SMA_20', 'EMA_20', 'return_1d', 'return_5d', 'volatility_20d']

    def fit(self, df: pd.DataFrame, target: str = 'target_return'):
        """Fits the model using standard features to predict the target."""
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            logger.error(f"Missing features for LGBM: {missing}")
            return

        if target not in df.columns:
            logger.error(f"Target column {target} missing.")
            return

        X = df[self.features]
        y = df[target]

        logger.info("Fitting LightGBM Ranker...")
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("LightGBM Ranker fitted.")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            logger.warning("LightGBM not fitted. Returning zeros.")
            return np.zeros(len(df))

        X = df[self.features]
        return self.model.predict(X)
