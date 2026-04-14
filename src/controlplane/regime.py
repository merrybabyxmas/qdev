from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _safe_abs_mean(values: pd.Series) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(np.abs(finite)))


def _latest_feature_event(soak_records: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not soak_records:
        return None
    for record in reversed(soak_records):
        status = record.get("status")
        if not isinstance(status, dict):
            continue
        stream = status.get("stream")
        if not isinstance(stream, dict):
            continue
        details = stream.get("details")
        if not isinstance(details, dict):
            continue
        feature = details.get("last_feature_event")
        if isinstance(feature, dict):
            return feature
    return None


def classify_current_regime(panel: pd.DataFrame, soak_records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if panel.empty:
        return {
            "regime": "unknown",
            "reason": "panel_unavailable",
            "metrics": {},
        }

    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    daily = (
        frame.groupby("date", sort=True)
        .agg(
            market_return_1d=("market_return_1d", "mean"),
            market_return_5d=("market_return_5d", "mean"),
            market_volatility_20d=("market_volatility_20d", "mean"),
            market_dispersion_1d=("market_dispersion_1d", "mean"),
            shock_score=("shock_score", "mean"),
            jump_rate=("jump_flag", "mean"),
            corr_mean=("corr_to_market_20d", _safe_abs_mean),
        )
        .fillna(0.0)
    )
    latest = daily.iloc[-1]
    vol_low = float(daily["market_volatility_20d"].quantile(0.33))
    vol_high = float(daily["market_volatility_20d"].quantile(0.67))
    disp_low = float(daily["market_dispersion_1d"].quantile(0.33))
    shock_high = float(daily["shock_score"].quantile(0.80))

    feature_event = _latest_feature_event(soak_records)
    spread = float(feature_event.get("spread")) if isinstance(feature_event, dict) and feature_event.get("spread") is not None else None
    obi = float(feature_event.get("obi")) if isinstance(feature_event, dict) and feature_event.get("obi") is not None else None

    metrics = {
        "market_return_1d": float(latest["market_return_1d"]),
        "market_return_5d": float(latest["market_return_5d"]),
        "market_volatility_20d": float(latest["market_volatility_20d"]),
        "market_dispersion_1d": float(latest["market_dispersion_1d"]),
        "shock_score": float(latest["shock_score"]),
        "jump_rate": float(latest["jump_rate"]),
        "corr_mean": float(latest["corr_mean"]),
        "spread": spread,
        "obi": obi,
        "as_of": daily.index[-1].isoformat(),
    }

    if latest["jump_rate"] >= 0.20 or latest["shock_score"] >= max(1.75, shock_high):
        return {"regime": "event_shock", "reason": "elevated_jump_and_shock_activity", "metrics": metrics}

    if spread is not None and spread >= 75.0 and latest["market_volatility_20d"] >= vol_high:
        return {"regime": "high_vol_toxic_flow", "reason": "wide_spread_and_high_volatility", "metrics": metrics}

    if latest["corr_mean"] >= 0.60 and latest["market_return_1d"] < -0.01:
        return {"regime": "correlated_selloff", "reason": "high_correlation_with_negative_market_return", "metrics": metrics}

    if latest["market_return_1d"] * latest["market_return_5d"] < 0:
        return {"regime": "mean_reversion", "reason": "short_and_medium_horizon_returns_disagree", "metrics": metrics}

    if latest["market_return_5d"] >= 0.015 and latest["market_dispersion_1d"] <= max(disp_low, 0.01):
        return {"regime": "trend", "reason": "positive_medium_horizon_market_drift", "metrics": metrics}

    if (
        abs(latest["market_return_5d"]) <= 0.01
        and latest["market_volatility_20d"] <= vol_low
        and (spread is None or spread <= 60.0)
    ):
        return {"regime": "low_vol_stable_spread", "reason": "quiet_market_and_tight_spread", "metrics": metrics}

    if spread is not None and spread >= 75.0:
        return {"regime": "thin_liquidity", "reason": "wide_spread_without_full_volatility_breakout", "metrics": metrics}

    return {"regime": "mixed_transition", "reason": "no_single_regime_dominates", "metrics": metrics}
