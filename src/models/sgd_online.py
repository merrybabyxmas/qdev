import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from src.utils.logger import logger

class OnlineSGDRegressor:
    """
    초당 수천 번의 틱에서도 1개씩 즉시(partial_fit) 학습 가능한 초경량 선형 회귀 모델.
    O(1) 시간에 즉시 가중치를 업데이트하므로 진정한 실시간 스트림 처리에 적합함.
    """
    def __init__(self, learning_rate: str = 'adaptive', eta0: float = 0.01):
        self.model = SGDRegressor(learning_rate=learning_rate, eta0=eta0, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def update(self, X_new: np.ndarray, y_new: np.ndarray):
        """1개 혹은 미니배치의 X, y를 받아 즉시 가중치 업데이트 (Online Learning)"""
        if len(X_new) == 0:
            return

        if not self.is_fitted:
            # 첫 배치로 스케일러 초기화 및 첫 학습
            X_scaled = self.scaler.fit_transform(X_new)
            self.model.partial_fit(X_scaled, y_new)
            self.is_fitted = True
            logger.info("Initialized Online SGD Model with first batch.")
        else:
            # 온라인 스케일러 업데이트 방식 (매우 간소화된 버전)
            self.scaler.partial_fit(X_new)
            X_scaled = self.scaler.transform(X_new)
            self.model.partial_fit(X_scaled, y_new)
            logger.debug(f"Incrementally updated SGD Model with {len(X_new)} samples.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(len(X))
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
