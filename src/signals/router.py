from typing import Dict, Any
from src.models.state_detector import MarketState
from src.utils.logger import logger

class ExecutionAction:
    """
    라우터가 내리는 실행(Execution) 액션의 형태
    """
    def __init__(self, action: str, price_offset: float = 0.0, size_multiplier: float = 1.0):
        self.action = action # "PASSIVE_MAKE", "AGGRESSIVE_TAKE", "HALT", "REDUCE"
        self.price_offset = price_offset
        self.size_multiplier = size_multiplier

class PipelineRouter:
    """
    시장 미시구조 기반의 MarketState(8개 클래스)에 따라
    가장 적합한 모델 예측치 해석 및 실행 정책(Execution Policy)을 동적으로 선택하는 라우터 계층.
    """
    def __init__(self):
        pass

    def route_execution(self, state: MarketState, prediction: float) -> ExecutionAction:
        """
        상태에 따른 모델 평가 가중치 조절 및 리스크 실행 통제
        prediction: ML 모델의 단기 상승/하락 예측 강도 (단위: 수익률 bps)
        """
        # 1. 고위험 상태: 즉시 거래 중단 또는 노출 축소 (Kill Switch 연계)
        if state in [MarketState.EVENT_SHOCK, MarketState.WIDE_SPREAD_ILLIQUID, MarketState.TIME_ANOMALY, MarketState.HIGH_VOL_TOXIC]:
            logger.warning(f"[Pipeline Router] Market State is {state.name}. Halting trading to avoid adverse selection and toxic flow.")
            return ExecutionAction(action="HALT", size_multiplier=0.0)

        # 2. 안정적 추세 / 강한 모멘텀 상태: 공격적 유동성 취득 (Aggressive Taking)
        if state in [MarketState.STABLE_TREND, MarketState.HIGH_VOL_TREND]:
            if abs(prediction) > 0.5: # 0.5 bps 이상 강한 신호 시
                logger.info(f"[Pipeline Router] State: {state.name}. Trending. Using Aggressive Taking Strategy.")
                # 상대의 호가를 직접 치는(Crossing the spread) 공격적 주문으로 라우팅
                return ExecutionAction(action="AGGRESSIVE_TAKE", size_multiplier=1.0)
            else:
                return ExecutionAction(action="PASSIVE_MAKE", size_multiplier=0.5) # 신호가 약하면 관망 또는 소극적 패시브

        # 3. 안정적 평균회귀장 / 정상 변동성 균형장: 스프레드 수취 마켓메이킹 (Passive Making)
        if state in [MarketState.STABLE_MEAN_REVERSION, MarketState.NORMAL_BALANCED]:
            logger.info(f"[Pipeline Router] State: {state.name}. Mean Reverting / Balanced. Using Passive Making Strategy.")
            # 예측값에 따라 스프레드를 벌리거나 좁히는 유동성 공급 (Market Making)
            return ExecutionAction(action="PASSIVE_MAKE", price_offset=0.01, size_multiplier=1.0)

        # Fallback
        return ExecutionAction(action="HALT", size_multiplier=0.0)
