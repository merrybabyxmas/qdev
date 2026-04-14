from typing import Dict
from src.utils.logger import logger

class RiskManager:
    """
    Implements core risk policies:
    - Position Cap (single name max weight)
    - Drawdown Kill Switch
    """
    def __init__(self, max_position_cap: float = 0.20, max_drawdown: float = 0.15):
        self.max_position_cap = max_position_cap
        self.max_drawdown = max_drawdown
        self.kill_switch_active = False

    def check_drawdown(self, current_drawdown: float):
        if current_drawdown >= self.max_drawdown:
            logger.error(f"DRAWDOWN KILL SWITCH ACTIVATED: {current_drawdown:.2%} >= {self.max_drawdown:.2%}")
            self.kill_switch_active = True
        return self.kill_switch_active

    def apply_position_caps(self, target_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Caps individual positions at max_position_cap.
        """
        if self.kill_switch_active:
            logger.warning("Kill switch is active. Forcing all weights to 0.")
            return {sym: 0.0 for sym in target_weights}

        capped_weights = {}
        for sym, weight in target_weights.items():
            if weight > self.max_position_cap:
                logger.debug(f"Capping {sym} weight from {weight:.2%} to {self.max_position_cap:.2%}")
                capped_weights[sym] = self.max_position_cap
            else:
                capped_weights[sym] = weight

        return capped_weights

    def pretrade_check(
        self,
        target_weights: Dict[str, float],
        current_drawdown: float = 0.0,
        stale_data: bool = False,
        max_gross_exposure: float | None = None,
    ) -> tuple[bool, list[str]]:
        """Basic pre-trade gate for paper/live-safe execution."""
        reasons: list[str] = []

        if stale_data:
            reasons.append("stale_data")

        if self.check_drawdown(current_drawdown):
            reasons.append("drawdown_kill_switch")

        gross_exposure = sum(abs(weight) for weight in target_weights.values())
        if max_gross_exposure is not None and gross_exposure > max_gross_exposure:
            reasons.append(f"gross_exposure>{max_gross_exposure}")

        oversized = [sym for sym, weight in target_weights.items() if abs(weight) > self.max_position_cap]
        if oversized:
            reasons.append(f"position_cap_exceeded:{','.join(sorted(oversized))}")

        if reasons:
            logger.warning(f"Pre-trade gate blocked execution: {reasons}")
            return False, reasons

        return True, reasons
