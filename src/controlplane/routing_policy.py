"""
RoutingPolicyEngine — 3-layer unified routing policy generator.

Reads leaderboard + regime snapshot and emits routing_policy.json:
  Layer 1  macro_daily   — best promoted daily/DL model per regime
  Layer 2  intraday_swing — best promoted intraday model (or disabled)
  Layer 3  hft           — hft champion + allow flag + symbol thresholds

The HFT engine (execution plane) reads the 'hft' section.
The model scheduler reads the full policy for reporting / dashboard.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logger import logger

# ---------------------------------------------------------------------------
# Regime → recommended macro pipeline (fallback to BASE_EQ)
# ---------------------------------------------------------------------------
_MACRO_REGIME_MAP: dict[str, str] = {
    "trend":               "F015",
    "correlated_selloff":  "F019",
    "high_vol_toxic_flow": "F019",
    "mean_reversion":      "S011",
    "low_vol_stable_spread": "BASE_EQ",
    "thin_liquidity":      "BASE_EQ",
    "event_shock":         "F009",
    "mixed_transition":    "F015",
}

_INTRADAY_REGIME_MAP: dict[str, str | None] = {
    "trend":               "I001",
    "mean_reversion":      "I002",
    "high_vol_toxic_flow": None,   # halt intraday on toxic flow
    "correlated_selloff":  "I005",
    "event_shock":         "I006",
    "thin_liquidity":      None,
    "low_vol_stable_spread": "I004",
    "mixed_transition":    "I001",
}

_HFT_HALT_REGIMES = {"high_vol_toxic_flow", "event_shock", "thin_liquidity"}

_MACRO_FAMILIES = {
    "Baseline", "Factor Model", "Financial + DL",
    "Deep Learning", "Bayesian", "SDE",
}
_INTRADAY_FAMILIES = {"Intraday Swing"}


def _best_promoted(leaderboard: pd.DataFrame, families: set[str], fallback_id: str) -> str:
    """Return pipeline_id of the top promoted/reference model in given families."""
    if leaderboard.empty or "family" not in leaderboard.columns:
        return fallback_id
    subset = leaderboard[leaderboard["family"].isin(families)]
    if subset.empty:
        return fallback_id
    promoted = subset[subset.get("decision", pd.Series("", index=subset.index)).isin({"promote", "reference"})]
    pool = promoted if not promoted.empty else subset
    if "final_score" in pool.columns:
        pool = pool.sort_values("final_score", ascending=False)
    return str(pool.iloc[0]["pipeline_id"]) if not pool.empty else fallback_id


class RoutingPolicyEngine:
    """
    Generates a unified 3-layer routing_policy.json.

    Usage::

        engine = RoutingPolicyEngine(policy_path)
        engine.generate(regime, leaderboard_df, symbol_configs, thresholds)
    """

    def __init__(self, policy_path: str | Path = "artifacts/control_plane/routing_policy.json"):
        self.policy_path = str(policy_path)
        os.makedirs(os.path.dirname(self.policy_path), exist_ok=True)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def generate(
        self,
        regime: str,
        leaderboard: pd.DataFrame,
        *,
        symbol_configs: dict[str, dict[str, Any]] | None = None,
        hft_thresholds: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        symbol_configs = symbol_configs or {
            "BTC/USD": {"enabled": True, "max_position_usd": 500.0},
            "ETH/USD": {"enabled": True, "max_position_usd": 300.0},
        }
        hft_thresholds = hft_thresholds or {
            "min_prediction_bps": 0.2,
            "max_spread_bps": 50.0,
        }

        # --- Layer 1: Macro/Daily ---
        macro_fallback = _MACRO_REGIME_MAP.get(regime, "BASE_EQ")
        macro_champion = _best_promoted(leaderboard, _MACRO_FAMILIES, macro_fallback)
        macro_challengers = self._challengers(leaderboard, _MACRO_FAMILIES, macro_champion, n=3)

        # --- Layer 2: Intraday/Swing ---
        intraday_fallback = _INTRADAY_REGIME_MAP.get(regime)
        intraday_subset = leaderboard[leaderboard["family"].isin(_INTRADAY_FAMILIES)] \
            if not leaderboard.empty and "family" in leaderboard.columns else pd.DataFrame()
        intraday_has_promoted = not intraday_subset.empty and any(
            intraday_subset.get("decision", pd.Series()).isin({"promote", "reference"})
        )
        intraday_active = intraday_has_promoted and intraday_fallback is not None
        intraday_champion = (
            _best_promoted(intraday_subset, _INTRADAY_FAMILIES, intraday_fallback or "I001")
            if intraday_active else None
        )

        # --- Layer 3: HFT ---
        allow_hft = regime not in _HFT_HALT_REGIMES
        hft_champion = "HFT_BASE_001"  # OnlineSGD is always the live champion
        hft_challengers = ["HFT_BASE_002", "HFT_DL_001", "HFT_SDE_001"]

        # --- Mandatory overlays ---
        mandatory_overlays = {
            "RISK_GLOBAL_001": {
                "active": True,
                "description": "stale-data / heartbeat / kill-switch",
            },
            "RISK_GLOBAL_002": {
                "active": True,
                "description": "drawdown clamp / exposure cap",
                "max_drawdown": 0.15,
                "max_gross_exposure": 1.0,
            },
            "EXEC_GLOBAL_001": {
                "active": True,
                "description": "duplicate-order blocking / cancel reconcile",
            },
            "EXEC_GLOBAL_002": {
                "active": True,
                "description": "fill-probability gate / order-age control",
                "min_fill_prob": 0.30,
                "max_order_age_seconds": 60,
            },
        }

        policy: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "regime": regime,
            "layers": {
                "macro_daily": {
                    "active": True,
                    "champion_pipeline_id": macro_champion,
                    "challenger_ids": macro_challengers,
                    "regime_hint": macro_fallback,
                    "allocation_scale": 0.60,
                },
                "intraday_swing": {
                    "active": intraday_active,
                    "champion_pipeline_id": intraday_champion,
                    "regime_hint": intraday_fallback,
                    "reason": "no_intraday_models_promoted" if not intraday_active else "active",
                    "allocation_scale": 0.25,
                },
                "hft": {
                    "active": allow_hft,
                    "allow_hft": allow_hft,
                    "champion_pipeline_id": hft_champion if allow_hft else None,
                    "challenger_ids": hft_challengers if allow_hft else [],
                    "symbols": symbol_configs,
                    "thresholds": hft_thresholds,
                    "reason": "halted_by_regime" if not allow_hft else "active",
                },
            },
            "mandatory_overlays": mandatory_overlays,
        }

        self._write(policy)
        logger.info(
            f"Routing policy published: regime={regime} | "
            f"macro={macro_champion} | intraday={'active' if intraday_active else 'off'} | "
            f"hft={'active' if allow_hft else 'halted'}"
        )
        return policy

    def read(self) -> dict[str, Any]:
        """Read the current routing policy (used by execution engines)."""
        if not os.path.exists(self.policy_path):
            return self._safe_default()
        try:
            with open(self.policy_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read routing policy: {e}")
            return self._safe_default()

    # Backwards-compatible shim so HFT engine still works with old hft_policy.json
    def read_hft_layer(self) -> dict[str, Any]:
        policy = self.read()
        hft = policy.get("layers", {}).get("hft", {})
        return {
            "timestamp": policy.get("timestamp", ""),
            "regime": policy.get("regime", "unknown"),
            "allow_hft": hft.get("allow_hft", False),
            "symbols": hft.get("symbols", {}),
            "thresholds": hft.get("thresholds", {}),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _challengers(
        self, leaderboard: pd.DataFrame, families: set[str], champion_id: str, n: int = 3
    ) -> list[str]:
        if leaderboard.empty or "family" not in leaderboard.columns:
            return []
        subset = leaderboard[leaderboard["family"].isin(families)]
        if "final_score" in subset.columns:
            subset = subset.sort_values("final_score", ascending=False)
        ids = [str(r["pipeline_id"]) for _, r in subset.iterrows() if str(r["pipeline_id"]) != champion_id]
        return ids[:n]

    def _write(self, policy: dict[str, Any]) -> None:
        tmp = self.policy_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(policy, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self.policy_path)
        except Exception as e:
            logger.error(f"Failed to write routing policy: {e}")

    def _safe_default(self) -> dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "regime": "UNKNOWN",
            "layers": {
                "macro_daily": {"active": False, "champion_pipeline_id": "BASE_EQ"},
                "intraday_swing": {"active": False, "champion_pipeline_id": None},
                "hft": {"active": False, "allow_hft": False, "symbols": {}, "thresholds": {}},
            },
            "mandatory_overlays": {},
        }
