import numpy as np
from typing import Dict, Any, List, Tuple
from src.models.sgd_online import OnlineSGDRegressor
from src.models.lgbm_online import OnlineLightGBMRanker
from src.utils.logger import logger


class RealTimeCrossSectionalRanker:
    """
    다수 종목의 실시간 피처를 받아들이고, SGD / LGBM 두 모델을 병렬로 학습·예측.
    SGD 예측값을 주 트레이딩 신호(target_weight)로 사용하고,
    LGBM 예측값은 비교·리더보드용으로 별도 반환.
    """
    def __init__(self, symbols: List[str], target_lookahead: int = 5):
        self.symbols = symbols
        self.model = OnlineSGDRegressor(learning_rate='constant', eta0=0.01)
        self.lgbm = OnlineLightGBMRanker(learning_rate=0.05, n_estimators_per_update=5, warmup_ticks=50)
        self.target_lookahead = target_lookahead

        # State Tracking
        self.latest_features = {sym: None for sym in symbols}
        self.tick_counts = {sym: 0 for sym in symbols}

        # Online Learning buffers (per symbol)
        self.history_features = {sym: [] for sym in symbols}
        self.history_mids = {sym: [] for sym in symbols}

    def update_and_predict(
        self, symbol: str, features_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Returns:
            sgd_predictions  — dict[symbol → bps]  (primary trading signal)
            lgbm_predictions — dict[symbol → bps]  (secondary comparison signal)
            target_weights   — dict[symbol → weight] (from SGD)
        """
        self.tick_counts[symbol] += 1
        current_mid = features_dict["mid_price"]

        # Feature vector (5-axis microstructure)
        last_mid = self.history_mids[symbol][-1] if self.history_mids[symbol] else current_mid
        mprice_drift = (features_dict["microprice"] - last_mid) / last_mid if last_mid > 0 else 0.0

        ml_features = np.array([
            features_dict["obi"],
            mprice_drift,
            features_dict["spread"],
            features_dict["toxicity_vpin"],
            features_dict["volatility_burst"],
        ])

        self.history_features[symbol].append(ml_features)
        self.history_mids[symbol].append(current_mid)
        self.latest_features[symbol] = features_dict

        # Online learning — both models update on the same lagged target
        tc = self.tick_counts[symbol]
        if tc > self.target_lookahead:
            idx = tc - self.target_lookahead - 1
            past_features = self.history_features[symbol][idx].reshape(1, -1)
            past_mid = self.history_mids[symbol][idx]
            target_return = np.array([(current_mid - past_mid) / past_mid * 10000.0])  # bps

            self.model.update(past_features, target_return)

            # LGBM requires a small warmup before first training
            if tc >= self.lgbm.warmup_ticks:
                self.lgbm.update(past_features, target_return)

        # SGD predictions (all symbols)
        sgd_predictions: Dict[str, float] = {}
        lgbm_predictions: Dict[str, float] = {}
        for sym in self.symbols:
            if self.history_features[sym]:
                feat_vec = self.history_features[sym][-1].reshape(1, -1)
                sgd_predictions[sym] = float(self.model.predict(feat_vec)[0])
                lgbm_predictions[sym] = float(self.lgbm.predict(feat_vec)[0])
            else:
                sgd_predictions[sym] = 0.0
                lgbm_predictions[sym] = 0.0

        # Target weights from SGD (positive-only, proportional)
        positive_preds = {k: v for k, v in sgd_predictions.items() if v > 0.0}
        total_pred = sum(positive_preds.values())
        target_weights = {sym: 0.0 for sym in self.symbols}
        if total_pred > 0:
            for sym, pred in positive_preds.items():
                target_weights[sym] = pred / total_pred

        return sgd_predictions, lgbm_predictions, target_weights
