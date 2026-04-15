from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.controlplane.artifacts import CONTROL_PLANE_ROOT


SERVICE_ROOT = CONTROL_PLANE_ROOT / "services"
DEFAULT_SERVICES = ("collector", "model_scheduler", "paper_soak")


def _service_path(name: str) -> Path:
    SERVICE_ROOT.mkdir(parents=True, exist_ok=True)
    return SERVICE_ROOT / f"{name}.json"


def load_service_state(name: str) -> dict[str, Any]:
    path = _service_path(name)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_service_state(name: str, payload: dict[str, Any]) -> Path:
    path = _service_path(name)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def clear_service_state(name: str) -> None:
    path = _service_path(name)
    if path.exists():
        path.unlink()


def is_process_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def service_status(name: str) -> dict[str, Any]:
    payload = load_service_state(name)
    pid = payload.get("pid")
    running = is_process_alive(int(pid)) if pid is not None else False
    payload.setdefault("service", name)
    payload["running"] = running
    return payload


def collect_service_statuses(names: tuple[str, ...] = DEFAULT_SERVICES) -> dict[str, dict[str, Any]]:
    return {name: service_status(name) for name in names}
