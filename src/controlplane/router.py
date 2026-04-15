from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd


REGIMES = (
    "trend",
    "mean_reversion",
    "low_vol_stable_spread",
    "high_vol_toxic_flow",
    "event_shock",
    "correlated_selloff",
    "thin_liquidity",
    "mixed_transition",
)


def _overlay_tokens(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [token.strip(" []'\"") for token in value.split(",") if token.strip(" []'\"")]
    return []


def _regime_bonus(row: pd.Series, regime: str) -> float:
    pipeline_id = str(row.get("pipeline_id", ""))
    feature_profile = str(row.get("feature_profile", ""))
    base_model = str(row.get("base_model", ""))
    decision = str(row.get("decision", ""))
    overlays = set(_overlay_tokens(row.get("overlays", [])))
    bonus = 0.0

    if decision in {"promote", "reference"}:
        bonus += 4.0
    if row.get("implementation_mode") == "direct":
        bonus += 0.5

    if regime == "trend":
        if pipeline_id == "F015":
            bonus += 6.0
        if feature_profile in {"residual_momentum", "technical"}:
            bonus += 3.0
        if base_model == "lightgbm":
            bonus += 1.0

    elif regime == "mean_reversion":
        if pipeline_id == "S011":
            bonus += 6.0
        if feature_profile == "volatility_proxy":
            bonus += 3.0
        if "inverse_vol_allocation" in overlays:
            bonus += 2.0

    elif regime == "low_vol_stable_spread":
        if pipeline_id == "BASE_EQ":
            bonus += 6.0
        if base_model == "equal_weight":
            bonus += 3.0
        if feature_profile in {"technical", "factor_proxy"}:
            bonus += 1.5

    elif regime == "high_vol_toxic_flow":
        if pipeline_id in {"S011", "F019"}:
            bonus += 5.0
        if feature_profile in {"volatility_proxy", "correlation_overlay"}:
            bonus += 3.0
        if "inverse_vol_allocation" in overlays or "correlation_penalty" in overlays:
            bonus += 2.0

    elif regime == "event_shock":
        if pipeline_id in {"F009", "F027", "S004"}:
            bonus += 6.0
        if feature_profile == "news_shock_proxy":
            bonus += 3.0
        if "jump_filter" in overlays:
            bonus += 2.0

    elif regime == "correlated_selloff":
        if pipeline_id == "F019":
            bonus += 6.0
        if feature_profile == "correlation_overlay":
            bonus += 3.0
        if "correlation_penalty" in overlays or "cvar_overlay" in overlays:
            bonus += 2.0

    elif regime == "thin_liquidity":
        if pipeline_id in {"BASE_EQ", "S011"}:
            bonus += 4.0
        if base_model in {"equal_weight", "linear", "bayesian_linear"}:
            bonus += 2.0
        if "jump_filter" in overlays:
            bonus += 1.0

    elif regime == "mixed_transition":
        if pipeline_id in {"F015", "F019", "BASE_EQ"}:
            bonus += 2.0

    return bonus


def _pick_top(frame: pd.DataFrame, regime: str) -> dict[str, Any]:
    if frame.empty:
        return {"pipeline_id": None, "reason": "no_candidates"}

    ranked = frame.copy()
    ranked["regime_bonus"] = ranked.apply(lambda row: _regime_bonus(row, regime), axis=1)
    ranked["regime_score"] = ranked["final_score"] + 8.0 * ranked["regime_bonus"]
    ranked = ranked.sort_values(["regime_score", "final_score"], ascending=[False, False]).reset_index(drop=True)
    top = ranked.iloc[0]
    return {
        "pipeline_id": str(top["pipeline_id"]),
        "name": str(top.get("name", "")),
        "family": str(top.get("family", "")),
        "final_score": float(top.get("final_score", 0.0)),
        "regime_score": float(top.get("regime_score", 0.0)),
        "reason": f"best_match_for_{regime}",
    }


def build_router_registry(
    leaderboard: pd.DataFrame,
    *,
    current_regime: str,
    existing_registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing_registry = existing_registry or {}
    candidates = leaderboard[leaderboard.get("promotion_candidate", False)].copy()
    if candidates.empty:
        candidates = leaderboard.copy()

    if candidates.empty:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "champion_pipeline_id": None,
            "challenger_pipeline_ids": [],
            "current_regime": current_regime,
            "active_pipeline_id": None,
            "override_enabled": False,
            "manual_champion_pipeline_id": None,
            "regime_assignments": {},
        }

    ranked = candidates.sort_values(["final_score", "test_summary.sharpe_ratio"], ascending=[False, False]).reset_index(drop=True)

    override_enabled = bool(existing_registry.get("override_enabled", False))
    manual_champion_pipeline_id = existing_registry.get("manual_champion_pipeline_id")

    champion_pipeline_id = str(ranked.iloc[0]["pipeline_id"])
    if override_enabled and manual_champion_pipeline_id in set(ranked["pipeline_id"].tolist()):
        champion_pipeline_id = str(manual_champion_pipeline_id)

    challenger_ids = [str(item) for item in ranked["pipeline_id"].tolist() if str(item) != champion_pipeline_id][:3]

    regime_assignments = {regime: _pick_top(ranked, regime) for regime in REGIMES}
    active_pipeline_id = champion_pipeline_id if override_enabled else regime_assignments.get(current_regime, {}).get("pipeline_id") or champion_pipeline_id

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "override_enabled": override_enabled,
        "manual_champion_pipeline_id": manual_champion_pipeline_id,
        "champion_pipeline_id": champion_pipeline_id,
        "challenger_pipeline_ids": challenger_ids,
        "current_regime": current_regime,
        "active_pipeline_id": active_pipeline_id,
        "regime_assignments": regime_assignments,
    }
