import pandas as pd
import vectorbt as vbt
from typing import Dict
from src.utils.logger import logger

class BacktestEngine:
    """
    Wraps vectorbt to simulate a multi-asset portfolio based on target weights.
    Implements transaction costs and slippage as per spec.
    """
    def __init__(self, fees: float = 0.001, slippage: float = 0.001):
        self.fees = fees
        self.slippage = slippage

    def run(self, price_data: pd.DataFrame, weights: pd.DataFrame) -> vbt.Portfolio:
        """
        Runs the backtest using price data and aligned weight targets.
        Grouping allows multi-asset portfolio computation rather than independent portfolios.
        """
        if price_data.empty:
            raise ValueError("price_data cannot be empty")
        if weights.empty:
            raise ValueError("weights cannot be empty")

        logger.info(f"Running backtest with fees={self.fees}, slippage={self.slippage}")

        common_idx = price_data.index.intersection(weights.index)
        if common_idx.empty:
            raise ValueError("No overlapping index between price_data and weights")

        prices = price_data.reindex(common_idx).sort_index()
        w = weights.reindex(common_idx).fillna(0.0).sort_index()

        portfolio = vbt.Portfolio.from_orders(
            close=prices,
            size=w,
            size_type='targetpercent',
            fees=self.fees,
            slippage=self.slippage,
            freq='D',
            group_by=True  # Groups multiple columns into a single portfolio
        )

        # Calculate single portfolio return
        ret_val = portfolio.total_return()

        logger.info(f"Backtest completed. Total Return: {ret_val:.2%}")
        return portfolio
