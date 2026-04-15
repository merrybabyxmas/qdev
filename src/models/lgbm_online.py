import joblib
import lightgbm as lgb
import numpy as np
from pathlib import Path
from src.utils.logger import logger

class OnlineLightGBMRanker:
    """
    HFT 환경에 적합한 점진적 학습(Incremental Learning) LightGBM 모델.
    기존 학습된 트리를 유지하면서 새로운 데이터가 들어올 때 트리를 추가하여(online update)
    모델이 시장의 최신 단기 패턴에 적응하게 함.
    """
    def __init__(self, learning_rate: float = 0.05, n_estimators_per_update: int = 5,
                 warmup_ticks: int = 50):
        self.learning_rate = learning_rate
        self.n_estimators_per_update = n_estimators_per_update
        self.warmup_ticks = warmup_ticks  # 첫 학습 전 최소 tick 수
        # Store the raw booster for incremental training
        self.booster = None
        self.is_fitted = False
        self.total_updates = 0
        self.features = ['obi', 'microprice_diff', 'spread', 'toxicity_vpin', 'volatility_burst']

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
                keep_training_booster=True,
            )
        self.total_updates += 1

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(len(X))
        return self.booster.predict(X)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "learning_rate": self.learning_rate,
            "n_estimators_per_update": self.n_estimators_per_update,
            "warmup_ticks": self.warmup_ticks,
            "is_fitted": self.is_fitted,
            "total_updates": self.total_updates,
            "features": self.features,
        }
        if self.is_fitted and self.booster is not None:
            booster_path = str(path) + ".lgb"
            self.booster.save_model(booster_path)
            payload["booster_path"] = booster_path
        joblib.dump(payload, path)
        logger.info(f"OnlineLightGBMRanker saved to {path}")

    @classmethod
    def load(cls, path: str) -> "OnlineLightGBMRanker":
        payload = joblib.load(path)
        obj = cls(
            learning_rate=payload.get("learning_rate", 0.05),
            n_estimators_per_update=payload.get("n_estimators_per_update", 5),
            warmup_ticks=payload.get("warmup_ticks", 50),
        )
        obj.is_fitted = payload.get("is_fitted", False)
        obj.total_updates = payload.get("total_updates", 0)
        obj.features = payload.get("features", obj.features)
        booster_path = payload.get("booster_path")
        if booster_path and Path(booster_path).exists():
            obj.booster = lgb.Booster(model_file=booster_path)
        logger.info(f"OnlineLightGBMRanker loaded from {path} (updates={obj.total_updates})")
        return obj
