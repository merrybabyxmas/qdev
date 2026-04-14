#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import ensure_project_root

ROOT = ensure_project_root()

from src.controlplane.artifacts import CONTROL_PLANE_ROOT  # noqa: E402
from src.controlplane.service import load_service_state, save_service_state, service_status  # noqa: E402


PYTHON = ROOT / ".venv" / "bin" / "python"
LOG_ROOT = CONTROL_PLANE_ROOT / "logs"


def _python_executable() -> str:
    return str(PYTHON if PYTHON.exists() else Path(sys.executable))


def _default_command(service: str) -> list[str]:
    if service == "collector":
        return [
            _python_executable(),
            str(ROOT / "scripts" / "run_data_collector.py"),
            "--iterations",
            "0",
            "--interval",
            "300",
            "--refresh-snapshot",
        ]
    if service == "model_scheduler":
        return [
            _python_executable(),
            str(ROOT / "scripts" / "run_model_scheduler.py"),
            "--iterations",
            "0",
            "--interval",
            "3600",
            "--suite",
            "shortlist",
            "--refresh-dataset",
        ]
    if service == "paper_soak":
        return [
            _python_executable(),
            str(ROOT / "scripts" / "run_paper_soak.py"),
            "--broker",
            "paper",
            "--stream-mode",
            "live",
            "--symbol",
            "BTC/USD",
            "--iterations",
            "240",
            "--interval",
            "60",
            "--stream-warmup-seconds",
            "10",
            "--stale-after",
            "120",
            "--record-root",
            str(ROOT / "artifacts" / "paper_soak"),
            "--session-name",
            "paper-soak-service",
        ]
    raise ValueError(f"Unsupported service: {service}")


def _is_running(payload: dict[str, object]) -> bool:
    pid = payload.get("pid")
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _start_service(service: str) -> dict[str, object]:
    current = load_service_state(service)
    if current and _is_running(current):
        return service_status(service)

    command = _default_command(service)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{service}.log"
    handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    payload = {
        "service": service,
        "pid": process.pid,
        "pgid": process.pid,
        "command": command,
        "log_path": str(log_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    save_service_state(service, payload)
    return service_status(service)


def _stop_service(service: str) -> dict[str, object]:
    payload = load_service_state(service)
    if not payload:
        return {"service": service, "running": False}

    pid = int(payload.get("pid", 0))
    pgid = int(payload.get("pgid", pid))
    if pid > 0:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            pass
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if not _is_running(payload):
                break
            time.sleep(0.2)
        if _is_running(payload):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                pass

    payload["stopped_at"] = datetime.now(timezone.utc).isoformat()
    save_service_state(service, payload)
    return service_status(service)


def main() -> int:
    parser = argparse.ArgumentParser(description="Start, stop, or inspect long-running control-plane services.")
    parser.add_argument("service", choices=["collector", "model_scheduler", "paper_soak"])
    parser.add_argument("action", choices=["start", "stop", "status"])
    args = parser.parse_args()

    if args.action == "start":
        payload = _start_service(args.service)
    elif args.action == "stop":
        payload = _stop_service(args.service)
    else:
        payload = service_status(args.service)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
