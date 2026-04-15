from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from _bootstrap import ensure_project_root

ensure_project_root()

from src.brokers.mock import MockBroker
from src.brokers.paper import PaperBroker
from src.ingestion.websocket_client import HFTStreamManager
from src.monitoring.health import HealthMonitor
from src.risk.manager import RiskManager


def _load_events(path: Path | None) -> list[dict[str, object]]:
    if path is None:
        return [
            {"type": "quote", "timestamp_ms": 1_000.0, "bid": 100.0, "bid_size": 5.0, "ask": 100.1, "ask_size": 4.0},
            {"type": "trade", "timestamp_ms": 1_010.0, "price": 100.05, "size": 1.0, "taker_side": "B"},
        ]
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Replay fixture must contain a JSON list: {path}")
    return payload


def _build_broker(name: str):
    if name == "mock":
        broker = MockBroker()
        broker.connect()
        return broker

    if name != "paper":
        raise ValueError(f"Unsupported broker mode: {name}")

    api_key = os.getenv("BROKER_API_KEY")
    secret_key = os.getenv("BROKER_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError("BROKER_API_KEY and BROKER_SECRET_KEY are required for paper broker health checks.")

    broker = PaperBroker(api_key=api_key, secret_key=secret_key, paper=True)
    broker.connect()
    return broker


def main() -> int:
    parser = argparse.ArgumentParser(description="Run broker/stream health checks with optional captured replay.")
    parser.add_argument("--broker", choices=["mock", "paper"], default="mock", help="Broker adapter to validate.")
    parser.add_argument("--replay-fixture", type=Path, default=None, help="JSON replay fixture with captured trade/quote events.")
    parser.add_argument("--iterations", type=int, default=2, help="Number of health loop iterations to run.")
    parser.add_argument("--interval", type=float, default=0.0, help="Seconds to sleep between health iterations.")
    parser.add_argument("--stale-after", type=float, default=30.0, help="Stream freshness threshold in seconds.")
    args = parser.parse_args()

    broker = _build_broker(args.broker)
    stream = HFTStreamManager(enable_live_stream=False)
    monitor = HealthMonitor(
        broker=broker,
        stream_manager=stream,
        risk_manager=RiskManager(max_position_cap=0.40, max_drawdown=0.20),
        stale_after_seconds=args.stale_after,
        failure_threshold=1,
    )

    events = _load_events(args.replay_fixture)
    stream.replay_events(events)

    results = monitor.run_loop(iterations=args.iterations, interval_seconds=args.interval)
    last = results[-1]
    print(json.dumps(last, ensure_ascii=False, indent=2))

    if not last["healthy"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
