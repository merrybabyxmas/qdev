import numpy as np
from typing import Dict, Any
from enum import Enum
from src.utils.logger import logger

class MarketState(Enum):
    """
    HFT 실무 및 미시구조 문헌을 반영한 8가지 핵심 운영 상황
    """
    STABLE_MEAN_REVERSION = 1     # 안정적 저변동성 평균회귀
    STABLE_TREND = 2              # 안정적 저변동성 추세
    NORMAL_BALANCED = 3           # 정상 변동성 균형장
    HIGH_VOL_TREND = 4            # 고변동성 추세 확장
    HIGH_VOL_TOXIC = 5            # 고변동성 독성 흐름 (위험)
    WIDE_SPREAD_ILLIQUID = 6      # 넓은 스프레드·얕은 호가 비유동성 장
    EVENT_SHOCK = 7               # 이벤트/뉴스 전후 불안정 장
    TIME_ANOMALY = 8              # 오픈/클로즈 시간대 특수 장

class MarketStateDetector:
    """
    5축(변동성, 유동성, 독성, 가격동학, 시간/이벤트) 데이터를 바탕으로
    3층 구조(Hard rule -> Probabilistic Smoothing -> 8 Classes)를 거쳐
    최종 시장 레짐을 결정하는 탐지기.
    """
    def __init__(self,
                 high_vol_threshold: float = 0.5,
                 wide_spread_threshold_bps: float = 30.0,
                 toxic_vpin_threshold: float = 0.8,
                 trend_threshold: float = 0.001):

        self.high_vol_threshold = high_vol_threshold
        self.wide_spread_threshold_bps = wide_spread_threshold_bps  # bps 기준 (예: 30 bps)
        self.toxic_vpin_threshold = toxic_vpin_threshold
        self.trend_threshold = trend_threshold

    def detect_state(self, features: Dict[str, Any]) -> MarketState:
        """
        1층: 하드 룰 기반 필터링 (Feature thresholds)
        2층: 논리적/확률적 분류 (간소화된 Rule Tree로 HMM 스무딩을 대체)
        3층: 8가지 운영 클래스 라우팅
        """
        vol = features.get("volatility_burst", 0.0)
        spread = features.get("spread", 0.0)
        mid = features.get("mid_price", 0.0)
        toxicity = features.get("toxicity_vpin", 0.0)
        price_drift = features.get("microprice_drift", 0.0)
        is_event = features.get("is_event_window", False)
        is_anomaly_time = features.get("is_anomaly_time", False)

        # spread를 bps로 변환 (달러 절대값이 아닌 상대적 비율 기준)
        spread_bps = (spread / mid * 10000.0) if mid > 0 else 0.0

        # 1. 시간대 / 이벤트 오버라이드 (가장 강력한 상태)
        if is_event:
            return MarketState.EVENT_SHOCK
        if is_anomaly_time:
            return MarketState.TIME_ANOMALY

        # 2. 비유동성 / 얕은 호가 (bps 기준: 30 bps 초과 시 비유동성)
        if spread_bps > self.wide_spread_threshold_bps:
            return MarketState.WIDE_SPREAD_ILLIQUID

        # 3. 고변동성 영역
        if vol > self.high_vol_threshold:
            if toxicity > self.toxic_vpin_threshold:
                return MarketState.HIGH_VOL_TOXIC
            elif abs(price_drift) > self.trend_threshold:
                return MarketState.HIGH_VOL_TREND
            else:
                return MarketState.NORMAL_BALANCED # 기본 정상 상태로 fallback

        # 4. 저/중 변동성 영역 (정상장)
        else:
            if abs(price_drift) > self.trend_threshold:
                return MarketState.STABLE_TREND
            elif abs(price_drift) <= self.trend_threshold and toxicity < 0.5:
                return MarketState.STABLE_MEAN_REVERSION
            else:
                return MarketState.NORMAL_BALANCED
