from __future__ import annotations

import unittest

import pandas as pd

from src.controlplane.ranking import build_leaderboard
from src.controlplane.regime import classify_current_regime
from src.controlplane.router import build_router_registry


class TestControlPlane(unittest.TestCase):
    def test_build_leaderboard_scores_candidates(self):
        frame = pd.DataFrame(
            [
                {
                    "pipeline_id": "F015",
                    "family": "Financial + DL",
                    "decision": "promote",
                    "base_model": "lightgbm",
                    "feature_profile": "residual_momentum",
                    "implementation_mode": "proxy",
                    "overlays": "[]",
                    "feature_count": 10,
                    "train_summary.total_return_pct": 12.0,
                    "validation_summary.total_return_pct": 4.0,
                    "test_summary.total_return_pct": 6.0,
                    "train_summary.sharpe_ratio": 1.8,
                    "validation_summary.sharpe_ratio": 1.2,
                    "test_summary.sharpe_ratio": 1.5,
                    "test_summary.max_drawdown_pct": -10.0,
                    "test_summary.cost_drag_pct": 2.5,
                    "test_summary.avg_turnover": 0.4,
                    "test_summary.avg_gross_exposure": 0.5,
                },
                {
                    "pipeline_id": "BASE_EQ",
                    "family": "Baseline",
                    "decision": "reference",
                    "base_model": "equal_weight",
                    "feature_profile": "technical",
                    "implementation_mode": "direct",
                    "overlays": "[]",
                    "feature_count": 5,
                    "train_summary.total_return_pct": 4.0,
                    "validation_summary.total_return_pct": 1.0,
                    "test_summary.total_return_pct": 2.0,
                    "train_summary.sharpe_ratio": 0.8,
                    "validation_summary.sharpe_ratio": 0.4,
                    "test_summary.sharpe_ratio": 0.6,
                    "test_summary.max_drawdown_pct": -14.0,
                    "test_summary.cost_drag_pct": 0.1,
                    "test_summary.avg_turnover": 0.0,
                    "test_summary.avg_gross_exposure": 1.0,
                },
                {
                    "pipeline_id": "BASE_RW",
                    "family": "Baseline",
                    "decision": "archive",
                    "base_model": "random_walk",
                    "feature_profile": "technical",
                    "implementation_mode": "direct",
                    "overlays": "[]",
                    "feature_count": 5,
                    "train_summary.total_return_pct": -4.0,
                    "validation_summary.total_return_pct": -2.0,
                    "test_summary.total_return_pct": -3.0,
                    "train_summary.sharpe_ratio": -0.8,
                    "validation_summary.sharpe_ratio": -0.3,
                    "test_summary.sharpe_ratio": -0.4,
                    "test_summary.max_drawdown_pct": -18.0,
                    "test_summary.cost_drag_pct": 4.0,
                    "test_summary.avg_turnover": 0.7,
                    "test_summary.avg_gross_exposure": 0.5,
                },
            ]
        )

        leaderboard = build_leaderboard(frame)

        self.assertFalse(leaderboard.empty)
        self.assertIn("final_score", leaderboard.columns)
        self.assertEqual(leaderboard.iloc[0]["pipeline_id"], "F015")
        self.assertGreater(float(leaderboard.iloc[0]["final_score"]), float(leaderboard.iloc[-1]["final_score"]))

    def test_regime_classifier_detects_event_shock(self):
        dates = pd.date_range("2026-04-01", periods=6, freq="D")
        rows = []
        for date in dates:
            for symbol in ("BTC/USD", "ETH/USD"):
                rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "market_return_1d": 0.02 if date == dates[-1] else 0.001,
                        "market_return_5d": 0.03 if date == dates[-1] else 0.005,
                        "market_volatility_20d": 0.04 if date == dates[-1] else 0.01,
                        "market_dispersion_1d": 0.03 if date == dates[-1] else 0.01,
                        "shock_score": 2.5 if date == dates[-1] else 0.6,
                        "jump_flag": 1.0 if date == dates[-1] else 0.0,
                        "corr_to_market_20d": 0.4,
                    }
                )
        panel = pd.DataFrame(rows)
        soak_records = [
            {
                "kind": "iteration",
                "status": {
                    "stream": {
                        "details": {
                            "last_feature_event": {"spread": 90.0, "obi": -0.6},
                        }
                    }
                },
            }
        ]

        regime = classify_current_regime(panel, soak_records)

        self.assertEqual(regime["regime"], "event_shock")
        self.assertIn("shock_score", regime["metrics"])

    def test_router_registry_uses_manual_override(self):
        leaderboard = pd.DataFrame(
            [
                {
                    "pipeline_id": "F015",
                    "name": "Residual Momentum",
                    "family": "Financial + DL",
                    "decision": "promote",
                    "promotion_candidate": True,
                    "feature_profile": "residual_momentum",
                    "base_model": "lightgbm",
                    "implementation_mode": "proxy",
                    "overlays": "[]",
                    "final_score": 88.0,
                    "test_summary.sharpe_ratio": 1.5,
                },
                {
                    "pipeline_id": "F009",
                    "name": "Event Model",
                    "family": "Financial + DL",
                    "decision": "promote",
                    "promotion_candidate": True,
                    "feature_profile": "news_shock_proxy",
                    "base_model": "lightgbm",
                    "implementation_mode": "proxy",
                    "overlays": "['jump_filter']",
                    "final_score": 72.0,
                    "test_summary.sharpe_ratio": 0.8,
                },
            ]
        )

        registry = build_router_registry(
            leaderboard,
            current_regime="event_shock",
            existing_registry={"override_enabled": True, "manual_champion_pipeline_id": "F009"},
        )

        self.assertEqual(registry["champion_pipeline_id"], "F009")
        self.assertEqual(registry["active_pipeline_id"], "F009")
        self.assertEqual(registry["regime_assignments"]["event_shock"]["pipeline_id"], "F009")


if __name__ == "__main__":
    unittest.main()
