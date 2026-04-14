#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import ensure_project_root

ROOT = ensure_project_root()

from src.controlplane.artifacts import CONTROL_PLANE_ROOT, latest_experiment_run  # noqa: E402
from src.controlplane.snapshot import build_dashboard_snapshot  # noqa: E402
from src.utils.logger import logger  # noqa: E402


STATUS_PATH = CONTROL_PLANE_ROOT / "model_scheduler_status.json"
LOG_PATH = CONTROL_PLANE_ROOT / "logs" / "model_scheduler.jsonl"
PYTHON = ROOT / ".venv" / "bin" / "python"


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
        handle.write("\n")


def _python_executable() -> str:
    return str(PYTHON if PYTHON.exists() else Path(sys.executable))


def _run_cycle(args: argparse.Namespace) -> tuple[dict[str, object], int]:
    report_path = ROOT / "reports" / f"model_cycle_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md"
    command = [
        _python_executable(),
        str(ROOT / "scripts" / "run_candidate_batch.py"),
        "--suite",
        args.suite,
        "--report-path",
        str(report_path),
        "--days",
        str(args.days),
        "--symbols",
        args.symbols,
    ]
    if args.refresh_dataset:
        command.append("--refresh-dataset")

    started_at = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    snapshot = build_dashboard_snapshot()
    latest_run = latest_experiment_run()

    payload = {
        "service": "model_scheduler",
        "healthy": completed.returncode == 0,
        "last_cycle_started_at": started_at,
        "last_cycle_finished_at": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite,
        "days": args.days,
        "symbols": args.symbols,
        "refresh_dataset": bool(args.refresh_dataset),
        "report_path": str(report_path),
        "latest_run": str(latest_run) if latest_run is not None else None,
        "stdout_tail": completed.stdout.splitlines()[-20:],
        "stderr_tail": completed.stderr.splitlines()[-20:],
        "snapshot_generated_at": snapshot.get("generated_at"),
    }
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _append_jsonl(LOG_PATH, payload)
    logger.info("Model scheduler cycle complete", status=payload)
    return payload, completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run recurring candidate training and leaderboard refresh cycles.")
    parser.add_argument("--suite", choices=["baselines", "shortlist", "full"], default="shortlist")
    parser.add_argument("--symbols", default="BTC/USD,ETH/USD,SOL/USD,AVAX/USD,DOGE/USD")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--iterations", type=int, default=1, help="0 means run forever.")
    parser.add_argument("--interval", type=float, default=3600.0, help="Seconds between cycles.")
    parser.add_argument("--refresh-dataset", action="store_true")
    args = parser.parse_args()

    iteration = 0
    final_exit_code = 0
    while True:
        _, exit_code = _run_cycle(args)
        final_exit_code = exit_code
        iteration += 1
        if args.iterations > 0 and iteration >= args.iterations:
            break
        if args.interval > 0:
            time.sleep(args.interval)
    return final_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
