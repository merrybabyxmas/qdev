from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.brokers.base import BrokerInterface
from src.risk.manager import RiskManager
from src.utils.logger import logger


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class HealthCheckResult:
    name: str
    healthy: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "healthy": self.healthy, "details": self.details}


class HealthMonitor:
    """
    Lightweight runtime health loop for broker connectivity, stream freshness,
    and risk kill-switch wiring.
    """

    def __init__(
        self,
        broker: BrokerInterface,
        stream_manager: Any | None = None,
        risk_manager: RiskManager | None = None,
        *,
        stale_after_seconds: float = 5.0,
        failure_threshold: int = 3,
        require_stream: bool = True,
        auto_sync: bool = True,
    ):
        self.broker = broker
        self.stream_manager = stream_manager
        self.risk_manager = risk_manager
        self.stale_after_seconds = stale_after_seconds
        self.failure_threshold = failure_threshold
        self.require_stream = require_stream
        self.auto_sync = auto_sync
        self.failure_count = 0
        self.last_status: dict[str, Any] | None = None
        self._equity_baseline: float | None = None

    def _heartbeat(self) -> dict[str, Any]:
        heartbeat = self.broker.heartbeat()
        if not isinstance(heartbeat, dict):
            heartbeat = {"ok": bool(heartbeat)}
        heartbeat.setdefault("ok", True)
        return heartbeat

    def _sync_state(self) -> dict[str, Any] | None:
        if self.auto_sync and hasattr(self.broker, "sync_state"):
            snapshot = self.broker.sync_state()
            if isinstance(snapshot, dict):
                return snapshot
        return None

    def _check_broker(self) -> HealthCheckResult:
        try:
            snapshot = self._sync_state()
            heartbeat = self._heartbeat()
            account = self.broker.get_account()
            healthy = bool(heartbeat.get("ok", False)) and account is not None
            details = {
                "heartbeat": heartbeat,
                "account": account,
                "snapshot": snapshot,
            }
            return HealthCheckResult("broker", healthy, details)
        except Exception as exc:  # pragma: no cover - defensive runtime path
            logger.error("Broker healthcheck failed", error=str(exc))
            return HealthCheckResult("broker", False, {"error": str(exc)})

    def _check_stream(self) -> HealthCheckResult:
        if self.stream_manager is None:
            healthy = not self.require_stream
            return HealthCheckResult(
                "stream",
                healthy,
                {"status": "not_configured", "require_stream": self.require_stream},
            )

        last_seen = getattr(self.stream_manager, "last_event_received_at", None)
        last_feature = getattr(self.stream_manager, "last_feature_event", None)
        if last_seen is None:
            healthy = not self.require_stream
            return HealthCheckResult(
                "stream",
                healthy,
                {"status": "no_events", "require_stream": self.require_stream, "last_feature_event": last_feature},
            )

        age = time.monotonic() - float(last_seen)
        healthy = age <= self.stale_after_seconds
        return HealthCheckResult(
            "stream",
            healthy,
            {
                "status": "fresh" if healthy else "stale",
                "age_seconds": age,
                "stale_after_seconds": self.stale_after_seconds,
                "last_feature_event": last_feature,
            },
        )

    def _check_risk(self, account: dict[str, Any] | None) -> HealthCheckResult:
        if self.risk_manager is None:
            return HealthCheckResult("risk", True, {"status": "not_configured"})

        equity = _safe_float((account or {}).get("equity"))
        if equity > 0 and self._equity_baseline is None:
            self._equity_baseline = equity

        drawdown = 0.0
        if self._equity_baseline and equity > 0:
            drawdown = max(0.0, 1.0 - (equity / self._equity_baseline))

        kill_switch_before = self.risk_manager.kill_switch_active
        kill_switch_after = self.risk_manager.check_drawdown(drawdown)
        healthy = not kill_switch_after
        details = {
            "equity": equity,
            "equity_baseline": self._equity_baseline,
            "drawdown": drawdown,
            "kill_switch_before": kill_switch_before,
            "kill_switch_after": kill_switch_after,
        }
        return HealthCheckResult("risk", healthy, details)

    def run_once(self) -> dict[str, Any]:
        broker_check = self._check_broker()
        account = broker_check.details.get("account") if broker_check.details else None
        stream_check = self._check_stream()
        risk_check = self._check_risk(account if isinstance(account, dict) else None)

        healthy = broker_check.healthy and stream_check.healthy and risk_check.healthy
        if healthy:
            self.failure_count = 0
        else:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold and self.risk_manager is not None:
                self.risk_manager.kill_switch_active = True

        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "healthy": healthy,
            "failure_count": self.failure_count,
            "kill_switch_active": bool(self.risk_manager.kill_switch_active) if self.risk_manager else False,
            "broker": broker_check.to_dict(),
            "stream": stream_check.to_dict(),
            "risk": risk_check.to_dict(),
        }
        self.last_status = status
        logger.info("Health monitor status", status=status)
        return status

    def run_loop(self, iterations: int = 1, interval_seconds: float = 1.0, stop_on_unhealthy: bool = False) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for index in range(iterations):
            status = self.run_once()
            results.append(status)
            if stop_on_unhealthy and not status["healthy"]:
                break
            if interval_seconds > 0 and index < iterations - 1:
                time.sleep(interval_seconds)
        return results

    def summary(self) -> dict[str, Any]:
        return self.last_status or {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "healthy": False,
            "failure_count": self.failure_count,
            "kill_switch_active": bool(self.risk_manager.kill_switch_active) if self.risk_manager else False,
        }
