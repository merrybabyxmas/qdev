import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from src.utils.logger import logger

class OnlineLogisticDirectionClassifier:
    """
    HFT_BASE_002: OBI + Logistic Direction Classifier
    호가 불균형과 Microprice 기반으로 Next 1-tick 방향성(UP/FLAT/DOWN) 분류.
    SGDClassifier(log_loss)를 사용하여 실시간(partial_fit) 업데이트.
    """
    def __init__(self, learning_rate: str = 'adaptive', eta0: float = 0.01):
        # 3 classes: -1 (Down), 0 (Flat), 1 (Up)
        self.model = SGDClassifier(loss='log_loss', learning_rate=learning_rate, eta0=eta0, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.classes = np.array([-1, 0, 1])

    def update(self, X_new: np.ndarray, y_new: np.ndarray):
        """
        y_new: 방향성 레이블 배열 (-1, 0, 1)
        """
        if len(X_new) == 0:
            return

        if not self.is_fitted:
            X_scaled = self.scaler.fit_transform(X_new)
            # classes 매개변수는 partial_fit 첫 호출 시 반드시 제공되어야 함
            self.model.partial_fit(X_scaled, y_new, classes=self.classes)
            self.is_fitted = True
            logger.info("Initialized Online Logistic Classifier with first batch.")
        else:
            self.scaler.partial_fit(X_new)
            X_scaled = self.scaler.transform(X_new)
            self.model.partial_fit(X_scaled, y_new)
            logger.debug(f"Incrementally updated Logistic Classifier with {len(X_new)} samples.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        각 클래스(-1, 0, 1)에 대한 확률을 반환.
        """
        if not self.is_fitted:
            return np.ones((len(X), 3)) / 3.0 # Uniform distribution if not fitted
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.zeros(len(X))
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
