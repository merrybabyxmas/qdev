from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.controlplane.artifacts import (
    CONTROL_PLANE_ROOT,
    latest_experiment_run,
    load_dataset_panel,
    load_experiment_manifest,
    load_results_frame,
    load_soak_records,
    load_status_file,
)
from src.controlplane.ranking import build_leaderboard
from src.controlplane.regime import classify_current_regime
from src.controlplane.router import build_router_registry
from src.controlplane.service import collect_service_statuses
from src.evaluation.hft_evaluator import build_hft_leaderboard_rows


LEADERBOARD_JSON = CONTROL_PLANE_ROOT / "leaderboard.json"
LEADERBOARD_CSV = CONTROL_PLANE_ROOT / "leaderboard.csv"
REGISTRY_JSON = CONTROL_PLANE_ROOT / "champion_registry.json"
REGIME_JSON = CONTROL_PLANE_ROOT / "regime_snapshot.json"
DASHBOARD_JSON = CONTROL_PLANE_ROOT / "dashboard_snapshot.json"


def _save_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _serialize_frame(frame: pd.DataFrame, *, limit: int = 50) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    trimmed = frame.head(limit).copy()
    for column in trimmed.columns:
        if pd.api.types.is_datetime64_any_dtype(trimmed[column]):
            trimmed[column] = trimmed[column].astype(str)
    return trimmed.to_dict(orient="records")


def _summarize_soak(records: list[dict[str, Any]]) -> dict[str, Any]:
    iterations = [record for record in records if record.get("kind") == "iteration"]
    run_end_records = [record for record in records if record.get("kind") == "run_end"]
    latest_iteration = iterations[-1] if iterations else {}
    latest_status = latest_iteration.get("status", {}) if isinstance(latest_iteration, dict) else {}

    timeline: list[dict[str, Any]] = []
    for record in iterations[-100:]:
        status = record.get("status", {})
        stream = ((status.get("stream") or {}).get("details") or {}) if isinstance(status, dict) else {}
        timeline.append(
            {
                "recorded_at": record.get("recorded_at"),
                "run_id": record.get("run_id"),
                "healthy": bool(status.get("healthy")),
                "failure_count": int(status.get("failure_count", 0)),
                "kill_switch_active": bool(status.get("kill_switch_active", False)),
                "stream_status": stream.get("status"),
                "stream_age_seconds": stream.get("age_seconds"),
            }
        )

    return {
        "record_count": len(records),
        "iteration_count": len(iterations),
        "run_count": len(run_end_records),
        "latest_status": latest_status,
        "timeline": timeline,
    }


def build_dashboard_snapshot() -> dict[str, Any]:
    CONTROL_PLANE_ROOT.mkdir(parents=True, exist_ok=True)

    run_dir = latest_experiment_run()
    manifest = load_experiment_manifest(run_dir)
    results = load_results_frame(run_dir)
    leaderboard = build_leaderboard(results)

    # Append live HFT rows (OnlineSGD per symbol) — computed independently so
    # their final_score is not distorted by cross-scaling with macro models.
    try:
        hft_rows = build_hft_leaderboard_rows()
        if not hft_rows.empty:
            leaderboard = pd.concat([leaderboard, hft_rows], ignore_index=True)
    except Exception:
        pass  # HFT rows are best-effort; never block snapshot generation

    if not leaderboard.empty:
        LEADERBOARD_JSON.write_text(json.dumps(_serialize_frame(leaderboard, limit=500), indent=2), encoding="utf-8")
        leaderboard.to_csv(LEADERBOARD_CSV, index=False)

    soak_records = load_soak_records()
    panel = load_dataset_panel(run_dir, manifest=manifest)
    regime = classify_current_regime(panel, soak_records)
    registry = build_router_registry(leaderboard, current_regime=str(regime["regime"]), existing_registry=_load_json(REGISTRY_JSON))

    _save_json(REGISTRY_JSON, registry)
    _save_json(REGIME_JSON, regime)

    collector_status = load_status_file("data_collector_status")
    model_scheduler_status = load_status_file("model_scheduler_status")
    soak_summary = _summarize_soak(soak_records)
    services = collect_service_statuses()

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "experiment_manifest": manifest,
        "dataset": manifest.get("dataset", {}),
        "services": services,
        "collector_status": collector_status,
        "model_scheduler_status": model_scheduler_status,
        "soak_summary": soak_summary,
        "regime": regime,
        "registry": registry,
        "leaderboard_top": _serialize_frame(leaderboard, limit=20),
        "leaderboard_full_path": str(LEADERBOARD_CSV) if LEADERBOARD_CSV.exists() else None,
    }
    _save_json(DASHBOARD_JSON, snapshot)
    return snapshot
