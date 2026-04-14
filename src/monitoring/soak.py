from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOAK_ROOT = ROOT / "artifacts" / "paper_soak"
DEFAULT_RECORD_FILE = "soak_records.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "__dict__"):
        return {key: val for key, val in vars(value).items() if not key.startswith("_")}
    return str(value)


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "dict"):
        return dict(value.dict())
    return {key: val for key, val in vars(value).items() if not key.startswith("_")}


@dataclass(frozen=True)
class SoakRecord:
    kind: str
    run_id: str
    recorded_at: str
    session_name: str
    iteration: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SoakRunResult:
    run_id: str
    record_path: Path
    statuses: list[dict[str, Any]]
    healthy_count: int
    unhealthy_count: int
    stopped_early: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class SoakRecordStore:
    """
    Append-only JSONL store for soak monitoring records.

    The store is deliberately independent from the validation transcript
    machinery so a long-running soak can keep accumulating records across
    repeated invocations without coupling to other runtime paths.
    """

    def __init__(self, root: Path | None = None, *, filename: str = DEFAULT_RECORD_FILE):
        self.root = Path(root) if root is not None else DEFAULT_SOAK_ROOT
        self.path = self.root / filename
        self.root.mkdir(parents=True, exist_ok=True)

    def append(self, record: Mapping[str, Any]) -> Path:
        payload = dict(record)
        payload.setdefault("recorded_at", _now_iso())
        payload.setdefault("source", "paper_soak")
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return self.path

    def append_run_start(
        self,
        *,
        run_id: str,
        session_name: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        return self.append(
            {
                "kind": "run_start",
                "run_id": run_id,
                "session_name": session_name,
                "metadata": _coerce_mapping(metadata),
            }
        )

    def append_iteration(
        self,
        *,
        run_id: str,
        iteration: int,
        session_name: str,
        status: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        return self.append(
            {
                "kind": "iteration",
                "run_id": run_id,
                "iteration": iteration,
                "session_name": session_name,
                "metadata": _coerce_mapping(metadata),
                "status": _coerce_mapping(status),
            }
        )

    def append_run_end(
        self,
        *,
        run_id: str,
        session_name: str,
        summary: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        return self.append(
            {
                "kind": "run_end",
                "run_id": run_id,
                "session_name": session_name,
                "metadata": _coerce_mapping(metadata),
                "summary": _coerce_mapping(summary),
            }
        )

    def load_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records


class SoakRunner:
    """
    Execute a health monitor loop while persistently recording every iteration.
    """

    def __init__(
        self,
        monitor: Any,
        store: SoakRecordStore | None = None,
        *,
        session_name: str = "paper-soak",
        metadata: Mapping[str, Any] | None = None,
    ):
        self.monitor = monitor
        self.store = store or SoakRecordStore()
        self.session_name = session_name
        self.metadata = _coerce_mapping(metadata)

    def run(
        self,
        *,
        iterations: int,
        interval_seconds: float = 0.0,
        stop_on_unhealthy: bool = False,
    ) -> SoakRunResult:
        run_id = uuid4().hex[:12]
        statuses: list[dict[str, Any]] = []
        healthy_count = 0
        unhealthy_count = 0
        stopped_early = False

        self.store.append_run_start(
            run_id=run_id,
            session_name=self.session_name,
            metadata={**self.metadata, "iterations": iterations, "interval_seconds": interval_seconds},
        )

        try:
            for iteration in range(iterations):
                status = self.monitor.run_once()
                statuses.append(status)
                if bool(status.get("healthy")):
                    healthy_count += 1
                else:
                    unhealthy_count += 1

                self.store.append_iteration(
                    run_id=run_id,
                    iteration=iteration,
                    session_name=self.session_name,
                    status=status,
                    metadata=self.metadata,
                )

                if stop_on_unhealthy and not bool(status.get("healthy")):
                    stopped_early = True
                    break

                if interval_seconds > 0 and iteration < iterations - 1:
                    time.sleep(interval_seconds)
        finally:
            summary = {
                "iterations_requested": iterations,
                "iterations_recorded": len(statuses),
                "healthy_count": healthy_count,
                "unhealthy_count": unhealthy_count,
                "stopped_early": stopped_early,
                "last_status": statuses[-1] if statuses else None,
            }
            self.store.append_run_end(
                run_id=run_id,
                session_name=self.session_name,
                summary=summary,
                metadata=self.metadata,
            )

        return SoakRunResult(
            run_id=run_id,
            record_path=self.store.path,
            statuses=statuses,
            healthy_count=healthy_count,
            unhealthy_count=unhealthy_count,
            stopped_early=stopped_early,
            metadata=dict(self.metadata),
        )
