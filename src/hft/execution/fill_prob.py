import numpy as np
from sklearn.linear_model import LogisticRegression
from src.utils.logger import logger

class FillProbabilityGate:
    """
    HFT_EXEC_001: Fill Probability Gate.
    주문 제출 시 지정가 주문이 실제 체결될 확률(Fill Probability)을 로지스틱 회귀를 통해 추정합니다.
    특정 임계치(Threshold) 미만의 체결 확률을 가진 주문은 제출하지 않거나,
    스팸(Quote spam)을 줄이기 위해 조기에 취소합니다.
    """
    def __init__(self, min_fill_prob: float = 0.6):
        self.min_fill_prob = min_fill_prob
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        과거 Order Book Replay(시뮬레이터)에서 나온 (체결 됨=1, 미체결=0) 데이터를 통해 훈련.
        X 특성 예시: [spread_bps, queue_size_proxy, top_of_book_size, order_age_ms]
        """
        if len(np.unique(y)) < 2:
            logger.warning("Fill probability fitting requires both classes (filled=1, missed=0). Skipping fit.")
            return

        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("Fill Probability Gate model fitted successfully.")

    def estimate_probability(self, features: np.ndarray) -> float:
        """단일 주문의 체결 확률 반환"""
        if not self.is_fitted:
            # 보수적 기본값: 모델이 없으면 일단 통과시킴 (또는 환경에 따라 0.0)
            return 1.0

        features = features.reshape(1, -1)
        prob = self.model.predict_proba(features)[0, 1] # Class '1' probability
        logger.debug(f"Estimated Fill Probability: {prob:.2%}")
        return prob

    def is_executable(self, features: np.ndarray) -> bool:
        """임계치보다 체결 확률이 높으면 실행 가능(True)으로 판단"""
        prob = self.estimate_probability(features)
        if prob >= self.min_fill_prob:
            return True
        else:
            logger.info(f"[EXEC_GATE] Order rejected due to low fill probability ({prob:.2%} < {self.min_fill_prob:.2%}).")
            return False
