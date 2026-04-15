import numpy as np
from typing import Dict, Any, List
from src.models.sgd_online import OnlineSGDRegressor
from src.utils.logger import logger

class RealTimeCrossSectionalRanker:
    """
    다수 종목의 실시간 피처를 받아들이고,
    Online SGD 모델을 통해 모든 종목의 단기 상승률(Alpha)을 연속 예측한 뒤,
    이를 바탕으로 크로스섹셔널(Cross-Sectional) 랭킹 및 타깃 포트폴리오 비중을 지속적으로 산출함.
    """
    def __init__(self, symbols: List[str], target_lookahead: int = 5):
        self.symbols = symbols
        self.model = OnlineSGDRegressor(learning_rate='constant', eta0=0.01)
        self.target_lookahead = target_lookahead

        # State Tracking
        self.latest_features = {sym: None for sym in symbols}
        self.tick_counts = {sym: 0 for sym in symbols}

        # Online Learning buffers (per symbol)
        self.history_features = {sym: [] for sym in symbols}
        self.history_mids = {sym: [] for sym in symbols}

    def update_and_predict(self, symbol: str, features_dict: Dict[str, Any]) -> Dict[str, float]:
        """
        1틱이 들어올 때마다:
        1. 해당 종목의 과거 피처로 타깃이 생성가능해졌다면 모델을 점진적 학습(Update)
        2. 최신 피처로 해당 종목의 신호 강도 예측
        3. 전체 종목의 최신 예측치 기반 랭킹 업데이트 및 Target Weights 반환
        """
        self.tick_counts[symbol] += 1
        current_mid = features_dict["mid_price"]

        # 1. Feature Extraction
        # 특성 벡터 구성 (간소화: OBI, Microprice Drift, Spread, Toxicity, Vol Burst)
        last_mid = self.history_mids[symbol][-1] if len(self.history_mids[symbol]) > 0 else current_mid
        mprice_drift = (features_dict["microprice"] - last_mid) / last_mid if last_mid > 0 else 0.0

        ml_features = np.array([
            features_dict["obi"],
            mprice_drift,
            features_dict["spread"],
            features_dict["toxicity_vpin"],
            features_dict["volatility_burst"]
        ])

        self.history_features[symbol].append(ml_features)
        self.history_mids[symbol].append(current_mid)
        self.latest_features[symbol] = features_dict

        # 2. Online Learning (SGD Partial Fit)
        tc = self.tick_counts[symbol]
        if tc > self.target_lookahead:
            idx = tc - self.target_lookahead - 1
            past_features = self.history_features[symbol][idx].reshape(1, -1)
            past_mid = self.history_mids[symbol][idx]

            target_return = np.array([(current_mid - past_mid) / past_mid * 10000.0]) # bps
            self.model.update(past_features, target_return)

        # 3. Prediction & Ranking
        predictions = {}
        for sym in self.symbols:
            if len(self.history_features[sym]) > 0:
                feat_vec = self.history_features[sym][-1].reshape(1, -1)
                predictions[sym] = self.model.predict(feat_vec)[0]
            else:
                predictions[sym] = 0.0

        # 4. Convert Predictions to Target Weights (Positive Only for simplicity)
        positive_preds = {k: v for k, v in predictions.items() if v > 0.0}
        total_pred = sum(positive_preds.values())

        target_weights = {sym: 0.0 for sym in self.symbols}
        if total_pred > 0:
            for sym, pred in positive_preds.items():
                target_weights[sym] = pred / total_pred # Weight by confidence

        return predictions, target_weights
