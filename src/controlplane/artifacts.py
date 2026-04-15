from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "artifacts"
CONTROL_PLANE_ROOT = ARTIFACT_ROOT / "control_plane"
EXPERIMENT_RUN_ROOT = ARTIFACT_ROOT / "experiments" / "runs"
EXPERIMENT_DATASET_ROOT = ARTIFACT_ROOT / "experiments" / "datasets"
SOAK_RECORD_PATH = ARTIFACT_ROOT / "paper_soak" / "soak_records.jsonl"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def latest_subdir(root: Path) -> Path | None:
    root = Path(root)
    if not root.exists():
        return None
    directories = [path for path in root.iterdir() if path.is_dir()]
    if not directories:
        return None
    return max(directories, key=lambda path: path.stat().st_mtime)


def latest_experiment_run(run_root: Path | None = None) -> Path | None:
    return latest_subdir(run_root or EXPERIMENT_RUN_ROOT)


def load_results_frame(run_dir: Path | None) -> pd.DataFrame:
    if run_dir is None:
        return pd.DataFrame()
    results_path = Path(run_dir) / "results.csv"
    if not results_path.exists():
        return pd.DataFrame()
    return pd.read_csv(results_path)


def load_experiment_manifest(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    return _load_json(Path(run_dir) / "manifest.json")


def resolve_dataset_root(manifest: dict[str, Any]) -> Path | None:
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        return None
    version = dataset.get("version")
    if not version:
        return None
    root = EXPERIMENT_DATASET_ROOT / str(version)
    return root if root.exists() else None


def load_dataset_panel(run_dir: Path | None = None, *, manifest: dict[str, Any] | None = None) -> pd.DataFrame:
    manifest = manifest or load_experiment_manifest(run_dir)
    dataset_root = resolve_dataset_root(manifest)
    if dataset_root is None:
        return pd.DataFrame()
    panel_path = dataset_root / "panel.csv"
    if not panel_path.exists():
        return pd.DataFrame()
    return pd.read_csv(panel_path, parse_dates=["date"])


def load_soak_records(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or SOAK_RECORD_PATH
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def load_status_file(name: str) -> dict[str, Any]:
    return _load_json(CONTROL_PLANE_ROOT / f"{name}.json")

