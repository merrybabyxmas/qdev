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
                logger.info(f"Capping {sym} weight from {weight:.2%} to {self.max_position_cap:.2%}")
                capped_weights[sym] = self.max_position_cap
            else:
                capped_weights[sym] = weight

        return capped_weights
