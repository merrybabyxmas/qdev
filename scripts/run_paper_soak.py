#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

from _bootstrap import ensure_project_root

ROOT = ensure_project_root()

from src.brokers.mock import MockBroker  # noqa: E402
from src.brokers.paper import PaperBroker  # noqa: E402
from src.ingestion.websocket_client import HFTStreamManager  # noqa: E402
from src.monitoring import HealthMonitor, SoakRecordStore, SoakRunner  # noqa: E402
from src.risk.manager import RiskManager  # noqa: E402


DEFAULT_REPLAY_FIXTURE = ROOT / "tests" / "fixtures" / "hft" / "captured_replay_sample.json"
DEFAULT_RECORD_ROOT = ROOT / "artifacts" / "paper_soak"


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
        raise ValueError("BROKER_API_KEY and BROKER_SECRET_KEY are required for paper soak runs.")

    broker = PaperBroker(api_key=api_key, secret_key=secret_key, paper=True)
    broker.connect()
    return broker


def _build_stream(args: argparse.Namespace):
    stream_mode = args.stream_mode
    if stream_mode == "auto":
        stream_mode = "live" if args.broker == "paper" else "replay"

    if stream_mode == "live":
        api_key = os.getenv("BROKER_API_KEY")
        secret_key = os.getenv("BROKER_SECRET_KEY")
        if not api_key or not secret_key:
            raise ValueError("BROKER_API_KEY and BROKER_SECRET_KEY are required for live stream mode.")
        stream = HFTStreamManager(
            api_key=api_key,
            secret_key=secret_key,
            symbol=args.symbol,
            enable_live_stream=True,
        )
        thread = threading.Thread(target=stream.run, daemon=True)
        thread.start()

        deadline = time.time() + args.stream_warmup_seconds
        while time.time() < deadline and stream.last_event_received_at is None:
            time.sleep(0.5)

        if stream.last_event_received_at is None:
            stream.last_event_received_at = time.monotonic()
        return stream

    stream = HFTStreamManager(enable_live_stream=False)
    events = _load_events(args.replay_fixture)
    stream.replay_events(events)
    return stream


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a paper soak loop and append every iteration to an independent record store.")
    parser.add_argument("--broker", choices=["mock", "paper"], default="paper", help="Broker adapter to validate.")
    parser.add_argument("--replay-fixture", type=Path, default=DEFAULT_REPLAY_FIXTURE, help="JSON replay fixture with captured trade/quote events.")
    parser.add_argument("--iterations", type=int, default=120, help="Number of soak iterations to run.")
    parser.add_argument("--interval", type=float, default=60.0, help="Seconds to sleep between iterations.")
    parser.add_argument("--stale-after", type=float, default=120.0, help="Stream freshness threshold in seconds.")
    parser.add_argument("--symbol", default="BTC/USD", help="Market symbol to subscribe for live stream mode.")
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT, help="Append-only soak record root directory.")
    parser.add_argument("--session-name", default="paper-soak", help="Human-readable session label stored with each record.")
    parser.add_argument("--stream-mode", choices=["auto", "live", "replay"], default="auto", help="Use live paper stream or recorded replay events.")
    parser.add_argument("--stream-warmup-seconds", type=float, default=15.0, help="Maximum time to wait for the live stream to receive its first event.")
    parser.add_argument("--stop-on-unhealthy", action="store_true", help="Stop immediately after the first unhealthy status.")
    args = parser.parse_args()

    broker = _build_broker(args.broker)
    stream = _build_stream(args)
    monitor = HealthMonitor(
        broker=broker,
        stream_manager=stream,
        risk_manager=RiskManager(max_position_cap=0.40, max_drawdown=0.20),
        stale_after_seconds=args.stale_after,
        failure_threshold=1,
    )

    store = SoakRecordStore(args.record_root)
    runner = SoakRunner(
        monitor,
        store,
        session_name=args.session_name,
        metadata={
            "broker": args.broker,
            "replay_fixture": str(args.replay_fixture),
            "stale_after": args.stale_after,
        },
    )
    result = runner.run(
        iterations=args.iterations,
        interval_seconds=args.interval,
        stop_on_unhealthy=args.stop_on_unhealthy,
    )

    output = {
        "run_id": result.run_id,
        "record_path": str(result.record_path),
        "iterations_recorded": len(result.statuses),
        "healthy_count": result.healthy_count,
        "unhealthy_count": result.unhealthy_count,
        "stopped_early": result.stopped_early,
        "last_status": result.statuses[-1] if result.statuses else None,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    return 0 if result.statuses and result.statuses[-1]["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
