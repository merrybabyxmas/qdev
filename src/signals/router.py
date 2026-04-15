from typing import Dict, Any
from src.models.state_detector import MarketState
from src.utils.logger import logger

class ExecutionAction:
    def __init__(self, action: str, price_offset: float = 0.0, size_multiplier: float = 1.0):
        self.action = action # "PASSIVE_MAKE", "AGGRESSIVE_TAKE", "HALT", "MICROPRICE_RULE"
        self.price_offset = price_offset
        self.size_multiplier = size_multiplier

class PipelineRouter:
    """
    시장 미시구조(Micro-state)와 거시정책(Macro-policy)을 결합.
    HFT_HYB_002: Regime-Aware Threshold Routing 지원.
    HFT_RISK_001: Toxicity / Spread Pre-check Gate.
    HFT_SDE_002: Burst / Jump Risk Overlay.
    HFT_BASE_003: Microprice Override.
    """
    def __init__(self):
        pass

    def route_execution(self, state: MarketState, prediction: float, policy: dict, symbol: str, features: dict) -> ExecutionAction:

        # 1. HFT_HYB_001 Macro Policy Filter (스케줄러 통제)
        allow_hft = policy.get("allow_hft", False)
        symbol_config = policy.get("symbols", {}).get(symbol, {})
        sym_enabled = symbol_config.get("enabled", True) if allow_hft else False

        if not sym_enabled:
            reason = symbol_config.get("reason", "HFT Globally Disabled or Symbol Disabled")
            logger.debug(f"[Router] Policy Override -> HALT for {symbol}. Reason: {reason}")
            return ExecutionAction(action="HALT", size_multiplier=0.0)

        thresholds = policy.get("thresholds", {})

        # 2. HFT_RISK_001: Spread & Toxicity Pre-check Gate
        max_spread = thresholds.get("spread_max_bps", 5.0) / 10000.0 * features.get("mid_price", 1.0)
        max_toxicity = thresholds.get("toxicity_max", 0.7)

        if features.get("spread", 0.0) > max_spread:
            logger.warning(f"[HFT_RISK_001] HALT: Spread ({features.get('spread')} > {max_spread}) too wide.")
            return ExecutionAction(action="HALT", size_multiplier=0.0)

        if features.get("toxicity_vpin", 0.0) > max_toxicity:
            logger.warning(f"[HFT_RISK_001] HALT: Toxicity ({features.get('toxicity_vpin')} > {max_toxicity}) too high.")
            return ExecutionAction(action="HALT", size_multiplier=0.0)

        # 3. HFT_SDE_002: Burst / Jump Risk Overlay
        jump_proxy = features.get("jump_proxy", 0.0)
        vol_burst = features.get("volatility_burst", 0.0)

        # Jumps reduce sizing aggressiveness
        size_mult = 1.0
        if jump_proxy > 0.0:
            logger.info(f"[HFT_SDE_002] Jump risk detected ({jump_proxy}). Reducing size multiplier.")
            size_mult = max(0.1, 1.0 - (jump_proxy * 0.2)) # Arbitrary heuristic dampener

        # 4. HFT_BASE_003: Microprice Override Rule
        mprice_drift = features.get("microprice_drift", 0.0)
        microprice_thresh = thresholds.get("microprice_drift_trigger", 0.005) # 50 bps drift
        if abs(mprice_drift) > microprice_thresh:
            logger.info(f"[HFT_BASE_003] Microprice drift extreme ({mprice_drift:.4f}). Triggering immediate trade override.")
            return ExecutionAction(action="AGGRESSIVE_TAKE", size_multiplier=size_mult)

        # 5. HFT_HYB_002: Regime-Aware Threshold Routing
        # Use tighter/looser prediction threshold based on regime
        base_pred_min = thresholds.get("prediction_bps_min", 0.5)

        if state in [MarketState.EVENT_SHOCK, MarketState.WIDE_SPREAD_ILLIQUID, MarketState.TIME_ANOMALY, MarketState.HIGH_VOL_TOXIC]:
            # Severe regimes always halt despite earlier gates
            logger.warning(f"[Router] Micro-State is {state.name}. Safe Halting trading.")
            return ExecutionAction(action="HALT", size_multiplier=0.0)

        elif state in [MarketState.STABLE_TREND, MarketState.HIGH_VOL_TREND]:
            # In trend regimes, lower threshold slightly to ride the wave
            regime_pred_min = base_pred_min * 0.8
            if abs(prediction) > regime_pred_min:
                return ExecutionAction(action="AGGRESSIVE_TAKE", size_multiplier=size_mult)
            else:
                return ExecutionAction(action="PASSIVE_MAKE", size_multiplier=size_mult * 0.5)

        elif state in [MarketState.STABLE_MEAN_REVERSION, MarketState.NORMAL_BALANCED]:
            # In mean reversion, need a stronger signal to override passive making
            regime_pred_min = base_pred_min * 1.5
            if abs(prediction) > regime_pred_min:
                return ExecutionAction(action="AGGRESSIVE_TAKE", size_multiplier=size_mult * 0.5)
            else:
                return ExecutionAction(action="PASSIVE_MAKE", size_multiplier=size_mult)

        return ExecutionAction(action="HALT", size_multiplier=0.0)
