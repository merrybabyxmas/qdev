import pandas as pd
from typing import Dict
from src.utils.logger import logger

class MLStrategy:
    """
    Allocates portfolio weights based on cross-sectional rank predictions.
    """
    def __init__(self, symbols: list):
        self.symbols = symbols

    def generate_weights(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """
        Generates target weights for the top performing predicted signals.
        Very basic Equal Weight among positive signals for safety.
        """
        positive_preds = {sym: pred for sym, pred in predictions.items() if pred > 0}

        if not positive_preds:
            logger.warning("No positive predictions, assigning 0 weight to all.")
            return {sym: 0.0 for sym in self.symbols}

        weight = 1.0 / len(positive_preds)
        target_weights = {sym: (weight if sym in positive_preds else 0.0) for sym in self.symbols}

        logger.info(f"MLStrategy generated targets for {len(positive_preds)} symbols: {target_weights}")
        return target_weights
