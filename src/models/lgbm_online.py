import lightgbm as lgb
import pandas as pd
import numpy as np
from src.utils.logger import logger

class OnlineLightGBMRanker:
    """
    HFT 환경에 적합한 점진적 학습(Incremental Learning) LightGBM 모델.
    기존 학습된 트리를 유지하면서 새로운 데이터가 들어올 때 트리를 추가하여(online update)
    모델이 시장의 최신 단기 패턴에 적응하게 함.
    """
    def __init__(self, learning_rate: float = 0.05, n_estimators_per_update: int = 5):
        self.learning_rate = learning_rate
        self.n_estimators_per_update = n_estimators_per_update
        # Store the raw booster for incremental training
        self.booster = None
        self.is_fitted = False
        self.features = ['obi', 'microprice_diff', 'spread', 'trade_intensity']

    def update(self, X_new: np.ndarray, y_new: np.ndarray):
        """
        새로운 (X, y) 배치 데이터가 들어오면 기존 모델에 트리를 이어서 학습 (Online Learning).
        """
        if len(X_new) == 0:
            return

        train_data = lgb.Dataset(X_new, label=y_new)

        params = {
            'objective': 'regression',
            'learning_rate': self.learning_rate,
            'num_leaves': 15,
            'max_depth': 4,
            'min_data_in_leaf': 10,
            'verbose': -1,
            'n_jobs': 1 # Single thread is often better for very small online updates
        }

        if not self.is_fitted:
            logger.info("Initializing Online LightGBM Model with first batch...")
            self.booster = lgb.train(params, train_data, num_boost_round=self.n_estimators_per_update)
            self.is_fitted = True
        else:
            logger.debug("Incrementally updating LightGBM Model with new batch...")
            # Keep_training_booster=True allows continuing training from the existing trees
            self.booster = lgb.train(
                params,
                train_data,
                num_boost_round=self.n_estimators_per_update,
                init_model=self.booster,
                keep_training_booster=True
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(len(X))
        return self.booster.predict(X)
