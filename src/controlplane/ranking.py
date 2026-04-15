from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd


DECISION_BONUS = {
    "promote": 6.0,
    "reference": 5.0,
    "keep": 3.0,
    "pending": 1.0,
    "archive": 0.0,
}

BASE_MODEL_LATENCY = {
    "equal_weight": 1.0,
    "random_walk": 0.9,
    "linear": 0.85,
    "bayesian_linear": 0.8,
    "hmm_router": 0.72,
    "lightgbm": 0.68,
}


def _safe_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _scale_score(series: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    if not higher_is_better:
        values = -values
    if values.empty:
        return values
    minimum = float(values.min())
    maximum = float(values.max())
    if np.isclose(minimum, maximum):
        return pd.Series(50.0, index=values.index, dtype=float)
    return ((values - minimum) / (maximum - minimum) * 100.0).astype(float)


def _overlay_count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, tuple):
        return len(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            parsed = [item for item in value.split(",") if item.strip()]
        if isinstance(parsed, (list, tuple)):
            return len(parsed)
    return 0


def build_leaderboard(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    frame = results.copy()
    frame["decision_bonus_raw"] = frame.get("decision", pd.Series("", index=frame.index)).map(DECISION_BONUS).fillna(0.0)
    frame["overlay_count"] = frame.get("overlays", pd.Series(index=frame.index, dtype=object)).map(_overlay_count).fillna(0).astype(int)

    test_return = _safe_series(frame, "test_summary.total_return_pct")
    validation_return = _safe_series(frame, "validation_summary.total_return_pct")
    train_return = _safe_series(frame, "train_summary.total_return_pct")
    test_sharpe = _safe_series(frame, "test_summary.sharpe_ratio")
    validation_sharpe = _safe_series(frame, "validation_summary.sharpe_ratio")
    train_sharpe = _safe_series(frame, "train_summary.sharpe_ratio")
    test_drawdown = _safe_series(frame, "test_summary.max_drawdown_pct").abs()
    test_cost_drag = _safe_series(frame, "test_summary.cost_drag_pct").abs()
    test_turnover = _safe_series(frame, "test_summary.avg_turnover").abs()
    test_gross_exposure = _safe_series(frame, "test_summary.avg_gross_exposure").abs()
    feature_count = _safe_series(frame, "feature_count")

    frame["risk_adjusted_return_raw"] = 0.55 * test_sharpe + 0.25 * validation_sharpe + 0.20 * (test_return / 10.0)
    frame["drawdown_raw"] = -test_drawdown
    frame["stability_raw"] = -(
        pd.concat([train_return, validation_return, test_return], axis=1).std(axis=1).fillna(0.0)
        + 2.0 * pd.concat([train_sharpe, validation_sharpe, test_sharpe], axis=1).std(axis=1).fillna(0.0)
    )
    positive_split_count = (
        (train_return > 0).astype(float) + (validation_return > 0).astype(float) + (test_return > 0).astype(float)
    )
    frame["regime_robustness_raw"] = positive_split_count + 0.5 * np.minimum(validation_sharpe, test_sharpe)
    frame["cost_sensitivity_raw"] = -(1.5 * test_cost_drag + test_turnover)
    frame["fill_quality_raw"] = -(0.75 * test_turnover + (test_gross_exposure - 0.35).abs())
    frame["latency_suitability_raw"] = (
        frame.get("base_model", pd.Series("", index=frame.index)).map(BASE_MODEL_LATENCY).fillna(0.6)
        - 0.02 * feature_count
        - 0.03 * frame["overlay_count"]
        + 0.03 * frame.get("implementation_mode", pd.Series("", index=frame.index)).eq("direct").astype(float)
    )

    frame["risk_adjusted_return_score"] = _scale_score(frame["risk_adjusted_return_raw"])
    frame["drawdown_score"] = _scale_score(frame["drawdown_raw"])
    frame["stability_score"] = _scale_score(frame["stability_raw"])
    frame["regime_robustness_score"] = _scale_score(frame["regime_robustness_raw"])
    frame["cost_sensitivity_score"] = _scale_score(frame["cost_sensitivity_raw"])
    frame["fill_quality_score"] = _scale_score(frame["fill_quality_raw"])
    frame["latency_suitability_score"] = _scale_score(frame["latency_suitability_raw"])

    frame["final_score"] = (
        0.30 * frame["risk_adjusted_return_score"]
        + 0.20 * frame["drawdown_score"]
        + 0.15 * frame["stability_score"]
        + 0.10 * frame["regime_robustness_score"]
        + 0.10 * frame["cost_sensitivity_score"]
        + 0.10 * frame["fill_quality_score"]
        + 0.05 * frame["latency_suitability_score"]
        + frame["decision_bonus_raw"]
    )

    frame["score_rank"] = frame["final_score"].rank(method="dense", ascending=False).astype(int)
    frame["promotion_candidate"] = frame.get("decision", pd.Series("", index=frame.index)).isin({"promote", "reference"})
    frame = frame.sort_values(["promotion_candidate", "final_score", "test_summary.sharpe_ratio"], ascending=[False, False, False])
    return frame.reset_index(drop=True)
