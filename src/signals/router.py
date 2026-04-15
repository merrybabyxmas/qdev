from typing import Dict, Any
from src.models.state_detector import MarketState
from src.utils.logger import logger

class ExecutionAction:
    """
    라우터가 내리는 실행(Execution) 액션의 형태
    """
    def __init__(self, action: str, price_offset: float = 0.0, size_multiplier: float = 1.0):
        self.action = action  # "PASSIVE_MAKE", "AGGRESSIVE_TAKE", "HALT", "REDUCE"
        self.price_offset = price_offset
        self.size_multiplier = size_multiplier

class PipelineRouter:
    """
    시장 미시구조 기반의 MarketState(8개 클래스)와
    상위 제어기(Control Plane)의 Policy를 결합하여
    동적으로 가장 적합한 실행 정책을 라우팅하는 계층.
    """
    def __init__(self):
        pass

    def route_execution(self, state: MarketState, prediction: float, policy: dict = None, symbol: str = None) -> ExecutionAction:
        """
        상태 및 스케줄러 정책에 따른 모델 라우팅
        prediction: ML 모델의 단기 상승/하락 예측 강도 (단위: 수익률 bps)
        """
        # 1. Macro Policy Filter (상위 스케줄러 제어)
        if policy is not None:
            allow_hft = policy.get("allow_hft", False)
            symbol_config = policy.get("symbols", {}).get(symbol, {}) if symbol else {}
            sym_enabled = symbol_config.get("enabled", True) if allow_hft else False

            if not sym_enabled:
                reason = symbol_config.get("reason", "HFT Globally Disabled or Symbol Disabled")
                logger.debug(f"[Pipeline Router] Policy Override -> HALT for {symbol}. Reason: {reason}")
                return ExecutionAction(action="HALT", size_multiplier=0.0)

            thresholds = policy.get("thresholds", {})
            min_pred = thresholds.get("prediction_bps_min", 0.5)
        else:
            min_pred = 0.5

        # 2. Microstructure State Filter — 고위험 상태: 즉시 거래 중단
        if state in [MarketState.EVENT_SHOCK, MarketState.WIDE_SPREAD_ILLIQUID, MarketState.TIME_ANOMALY, MarketState.HIGH_VOL_TOXIC]:
            logger.warning(f"[Pipeline Router] Market State is {state.name}. Halting trading to avoid adverse selection and toxic flow.")
            return ExecutionAction(action="HALT", size_multiplier=0.0)

        # 3. Trending Regimes → Aggressive Taker
        if state in [MarketState.STABLE_TREND, MarketState.HIGH_VOL_TREND]:
            if abs(prediction) > min_pred:
                logger.info(f"[Pipeline Router] State: {state.name}. Trending. Using Aggressive Taking Strategy.")
                return ExecutionAction(action="AGGRESSIVE_TAKE", size_multiplier=1.0)
            else:
                return ExecutionAction(action="PASSIVE_MAKE", size_multiplier=0.5)

        # 4. Mean Reverting / Balanced Regimes → Passive Maker
        if state in [MarketState.STABLE_MEAN_REVERSION, MarketState.NORMAL_BALANCED]:
            logger.info(f"[Pipeline Router] State: {state.name}. Mean Reverting / Balanced. Using Passive Making Strategy.")
            return ExecutionAction(action="PASSIVE_MAKE", price_offset=0.01, size_multiplier=1.0)

        # Fallback
        return ExecutionAction(action="HALT", size_multiplier=0.0)
