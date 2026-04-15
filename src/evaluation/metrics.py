from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceSummary:
    total_return_pct: float
    annualized_return_pct: float
    annualized_vol_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    avg_turnover: float
    total_turnover: float
    avg_gross_exposure: float
    cost_drag_pct: float
    active_days: int
    final_equity: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _safe_std(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return 0.0
    std = float(np.std(array, ddof=1))
    return std if std > 0 else 0.0


def compute_equity_curve(
    portfolio_returns: pd.Series,
    *,
    starting_equity: float = 1.0,
) -> pd.Series:
    if portfolio_returns.empty:
        return pd.Series(dtype=float)
    equity = (1.0 + portfolio_returns.fillna(0.0)).cumprod() * starting_equity
    equity.name = "equity"
    return equity


def compute_drawdown(equity_curve: pd.Series) -> pd.Series:
    if equity_curve.empty:
        return pd.Series(dtype=float)
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    drawdown.name = "drawdown"
    return drawdown


def summarize_performance(
    portfolio_returns: pd.Series,
    turnover: pd.Series,
    gross_exposure: pd.Series,
    *,
    cost_drag: pd.Series | None = None,
    periods_per_year: int = 252,
) -> PerformanceSummary:
    portfolio_returns = portfolio_returns.fillna(0.0)
    turnover = turnover.fillna(0.0)
    gross_exposure = gross_exposure.fillna(0.0)
    cost_drag = cost_drag.fillna(0.0) if cost_drag is not None else pd.Series(0.0, index=portfolio_returns.index)

    equity = compute_equity_curve(portfolio_returns)
    drawdown = compute_drawdown(equity)

    total_return = float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0
    mean_return = float(portfolio_returns.mean()) if not portfolio_returns.empty else 0.0
    std_return = _safe_std(portfolio_returns)
    downside = portfolio_returns[portfolio_returns < 0.0]
    downside_std = _safe_std(downside) if len(downside) else 0.0

    periods = max(len(portfolio_returns), 1)
    annualized_return = equity.iloc[-1] ** (periods_per_year / periods) - 1.0 if not equity.empty and equity.iloc[-1] > 0 else -1.0
    annualized_vol = std_return * sqrt(periods_per_year)
    sharpe = (mean_return / std_return * sqrt(periods_per_year)) if std_return > 0 else 0.0
    sortino = (mean_return / downside_std * sqrt(periods_per_year)) if downside_std > 0 else 0.0
    win_rate = float((portfolio_returns > 0.0).mean()) if len(portfolio_returns) else 0.0
    cost_drag_pct = float(cost_drag.sum()) * 100.0 if len(cost_drag) else 0.0

    return PerformanceSummary(
        total_return_pct=total_return * 100.0,
        annualized_return_pct=annualized_return * 100.0,
        annualized_vol_pct=annualized_vol * 100.0,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown_pct=float(drawdown.min() * 100.0) if not drawdown.empty else 0.0,
        win_rate_pct=win_rate * 100.0,
        avg_turnover=float(turnover.mean()) if len(turnover) else 0.0,
        total_turnover=float(turnover.sum()) if len(turnover) else 0.0,
        avg_gross_exposure=float(gross_exposure.mean()) if len(gross_exposure) else 0.0,
        cost_drag_pct=cost_drag_pct,
        active_days=int((gross_exposure > 0.0).sum()) if len(gross_exposure) else 0,
        final_equity=float(equity.iloc[-1]) if not equity.empty else 1.0,
    )


def classify_candidate(
    summary: PerformanceSummary,
    baseline: PerformanceSummary | None,
    *,
    min_return_delta_pct: float = 0.5,
    max_drawdown_slack_pct: float = 5.0,
    min_sharpe_delta: float = 0.05,
) -> str:
    if baseline is None:
        return "reference"

    return_delta = summary.total_return_pct - baseline.total_return_pct
    sharpe_delta = summary.sharpe_ratio - baseline.sharpe_ratio
    drawdown_ok = summary.max_drawdown_pct <= baseline.max_drawdown_pct + max_drawdown_slack_pct

    if return_delta >= min_return_delta_pct and sharpe_delta >= min_sharpe_delta and drawdown_ok:
        return "promote"
    if return_delta >= 0.0 and drawdown_ok:
        return "keep"
    return "archive"
