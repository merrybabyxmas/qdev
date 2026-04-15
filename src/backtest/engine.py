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
        logger.info(f"Running backtest with fees={self.fees}, slippage={self.slippage}")

        common_idx = price_data.index.intersection(weights.index)
        prices = price_data.loc[common_idx]
        w = weights.loc[common_idx]

        portfolio = vbt.Portfolio.from_orders(
            close=prices,
            size=w,
            size_type='targetpercent',
            fees=self.fees,
            slippage=self.slippage,
            freq='D',
            group_by=True # Groups multiple columns into a single portfolio
        )

        # Calculate single portfolio return
        ret_val = portfolio.total_return()

        logger.info(f"Backtest completed. Total Return: {ret_val:.2%}")
        return portfolio
