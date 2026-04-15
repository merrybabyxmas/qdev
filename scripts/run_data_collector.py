#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from _bootstrap import ensure_project_root

ROOT = ensure_project_root()

from src.controlplane.artifacts import CONTROL_PLANE_ROOT  # noqa: E402
from src.controlplane.snapshot import build_dashboard_snapshot  # noqa: E402
from src.evaluation.dataset import DatasetSpec, build_dataset_bundle  # noqa: E402
from src.features.builder import build_technical_features  # noqa: E402
from src.ingestion.loader import fetch_data_alpaca  # noqa: E402
from src.utils.logger import logger  # noqa: E402


STATUS_PATH = CONTROL_PLANE_ROOT / "data_collector_status.json"
LOG_PATH = CONTROL_PLANE_ROOT / "logs" / "data_collector.jsonl"
DEFAULT_RAW_ROOT = ROOT / "artifacts" / "data" / "raw"
DEFAULT_FEATURE_ROOT = ROOT / "artifacts" / "data" / "feature_ready"


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
        handle.write("\n")


def _resolve_window(lookback_days: int) -> tuple[str, str]:
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=lookback_days)
    return start.isoformat(), end.isoformat()


def _normalize_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(" ", "_")


def _collect_symbol(symbol: str, start_date: str, end_date: str, raw_root: Path, feature_root: Path) -> dict[str, object]:
    raw_root.mkdir(parents=True, exist_ok=True)
    feature_root.mkdir(parents=True, exist_ok=True)

    raw = fetch_data_alpaca(symbol, start_date, end_date)
    raw_frame = raw.copy()
    raw_frame.index.name = "date"
    raw_path = raw_root / f"{_normalize_symbol(symbol)}.csv"
    raw_frame.to_csv(raw_path)

    features = build_technical_features(raw_frame)
    feature_path = feature_root / f"{_normalize_symbol(symbol)}.csv"
    if not features.empty:
        features.index.name = "date"
        features.to_csv(feature_path)

    return {
        "symbol": symbol,
        "raw_path": str(raw_path),
        "feature_path": str(feature_path),
        "row_count": int(len(raw_frame)),
        "feature_row_count": int(len(features)),
        "latest_timestamp": str(raw_frame.index.max()) if not raw_frame.empty else None,
        "data_source": raw.attrs.get("data_source", "unknown"),
        "healthy": bool(not raw_frame.empty and not features.empty),
    }


def _run_cycle(args: argparse.Namespace) -> dict[str, object]:
    start_date, end_date = _resolve_window(args.lookback_days)
    symbols = tuple(symbol.strip() for symbol in args.symbols.split(",") if symbol.strip())
    raw_root = Path(args.raw_root)
    feature_root = Path(args.feature_root)

    per_symbol = [_collect_symbol(symbol, start_date, end_date, raw_root, feature_root) for symbol in symbols]
    spec = DatasetSpec(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        source="alpaca_or_synthetic",
        target_horizon=1,
    )
    bundle = build_dataset_bundle(spec, refresh=True)

    status = {
        "service": "collector",
        "healthy": all(bool(item["healthy"]) for item in per_symbol),
        "last_cycle_at": datetime.now(timezone.utc).isoformat(),
        "window": {"start_date": start_date, "end_date": end_date},
        "symbols": per_symbol,
        "dataset_version": bundle.version,
        "dataset_root": str(bundle.root),
        "dataset_row_count": int(bundle.manifest.get("row_count", len(bundle.frame))),
        "source_counts": bundle.manifest.get("source_counts", {}),
    }

    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    _append_jsonl(LOG_PATH, status)
    if args.refresh_snapshot:
        build_dashboard_snapshot()
    logger.info("Data collector cycle complete", status=status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuously collect data snapshots and refresh training-ready artifacts.")
    parser.add_argument("--symbols", default="BTC/USD,ETH/USD,SOL/USD,AVAX/USD,DOGE/USD")
    parser.add_argument("--lookback-days", type=int, default=180)
    parser.add_argument("--iterations", type=int, default=1, help="0 means run forever.")
    parser.add_argument("--interval", type=float, default=300.0, help="Seconds between cycles.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--refresh-snapshot", action="store_true")
    args = parser.parse_args()

    iteration = 0
    while True:
        _run_cycle(args)
        iteration += 1
        if args.iterations > 0 and iteration >= args.iterations:
            break
        if args.interval > 0:
            time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
