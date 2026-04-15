#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from uuid import uuid4

from _bootstrap import ensure_project_root

ensure_project_root()

from src.brokers.paper import PaperBroker
from src.brokers.paper_session import PaperSessionRecorder, RecordedPaperSessionClient, run_paper_broker_checklist


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "paper" / "recorded_paper_session_sample.json"


def _build_broker(mode: str, fixture: Path) -> PaperBroker:
    if mode == "replay":
        client = RecordedPaperSessionClient.from_file(fixture)
        return PaperBroker(trading_client=client, paper=True, price_provider=lambda symbol: 101.25)

    api_key = os.getenv("BROKER_API_KEY")
    secret_key = os.getenv("BROKER_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError("BROKER_API_KEY and BROKER_SECRET_KEY are required for live paper validation.")
    return PaperBroker(api_key=api_key, secret_key=secret_key, paper=True)


def _build_orders(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    run_tag = uuid4().hex[:10]
    open_client_order_id = args.open_client_order_id or f"paper-open-{run_tag}"
    fill_client_order_id = args.fill_client_order_id or f"paper-fill-{run_tag}"
    open_order = {
        "symbol": args.symbol,
        "qty": args.open_qty,
        "side": "buy",
        "type": "limit",
        "limit_price": args.open_limit_price,
        "time_in_force": args.time_in_force,
        "client_order_id": open_client_order_id,
    }
    fill_order = {
        "symbol": args.symbol,
        "side": "buy",
        "type": "market",
        "time_in_force": args.time_in_force,
        "client_order_id": fill_client_order_id,
    }
    if args.fill_notional is not None:
        fill_order["notional"] = args.fill_notional
    else:
        fill_order["qty"] = args.fill_qty
    return open_order, fill_order


def _find_order(orders: list[dict[str, object]], *, order_id: str, client_order_id: str) -> bool:
    for order in orders:
        if str(order.get("order_id") or order.get("id")) == str(order_id):
            return True
        if str(order.get("client_order_id")) == str(client_order_id):
            return True
    return False


def _validate_summary(summary: dict[str, object], open_order: dict[str, object], fill_order: dict[str, object], require_reconnect: bool) -> list[str]:
    errors: list[str] = []

    if not summary.get("connect", {}).get("connected", False):
        errors.append("connect_failed")
    if not summary.get("heartbeat", {}).get("ok", False):
        errors.append("heartbeat_failed")

    initial_sync = summary.get("initial_sync", {})
    if not isinstance(initial_sync, dict) or not initial_sync.get("account"):
        errors.append("account_sync_failed")

    open_section = summary.get("open_order", {})
    duplicate_blocking = summary.get("duplicate_blocking", {})
    cancel_section = summary.get("cancel_reconciliation", {})
    if open_order:
        if not isinstance(open_section, dict) or not open_section.get("order_id"):
            errors.append("open_order_missing")
        if not isinstance(duplicate_blocking, dict) or not duplicate_blocking.get("blocked", False):
            errors.append("duplicate_blocking_failed")
        cancel_sync = cancel_section.get("sync", {}) if isinstance(cancel_section, dict) else {}
        if isinstance(cancel_sync, dict):
            open_orders = cancel_sync.get("open_orders", []) or []
            if _find_order(
                [dict(order) for order in open_orders],
                order_id=str(open_section.get("order_id", "")),
                client_order_id=str(open_order.get("client_order_id", "")),
            ):
                errors.append("cancel_reconciliation_failed")

    fill_section = summary.get("fill_reconciliation", {})
    if fill_order:
        if not isinstance(fill_section, dict) or not fill_section.get("order_id"):
            errors.append("fill_order_missing")
        else:
            fill_sync = fill_section.get("sync", {})
            if isinstance(fill_sync, dict):
                fills = fill_sync.get("fills", []) or []
                if not _find_order(
                    [dict(order) for order in fills],
                    order_id=str(fill_section.get("order_id", "")),
                    client_order_id=str(fill_order.get("client_order_id", "")),
                ):
                    errors.append("fill_reconciliation_failed")

    if require_reconnect:
        reconnect_section = summary.get("reconnect", {})
        if not isinstance(reconnect_section, dict):
            errors.append("reconnect_missing")
        else:
            if not reconnect_section.get("heartbeat", {}).get("ok", False):
                errors.append("reconnect_heartbeat_failed")
            if not reconnect_section.get("sync", {}).get("account"):
                errors.append("reconnect_sync_failed")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the paper broker against live paper or recorded replay state.")
    parser.add_argument("--mode", choices=["replay", "paper"], default="replay", help="Use a recorded replay fixture or a live paper account.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE, help="Recorded paper-session replay fixture.")
    parser.add_argument("--record-output", type=Path, default=None, help="Write the executed transcript to this JSON file.")
    parser.add_argument("--symbol", default="BTC/USD", help="Symbol to exercise in the checklist.")
    parser.add_argument("--open-client-order-id", default=None, help="Client order id for the open-order check. Defaults to a unique generated id.")
    parser.add_argument("--open-qty", type=float, default=10.0, help="Quantity for the open-order check. Must satisfy broker minimum notional.")
    parser.add_argument("--open-limit-price", type=float, default=1.0, help="Limit price designed to keep the test order open.")
    parser.add_argument("--fill-client-order-id", default=None, help="Client order id for the fill reconciliation check. Defaults to a unique generated id.")
    parser.add_argument("--fill-qty", type=float, default=0.0001, help="Fallback quantity for the fill reconciliation check when fill_notional is not set.")
    parser.add_argument("--fill-notional", type=float, default=10.0, help="Notional for the fill reconciliation check. Crypto paper trading requires at least $10.")
    parser.add_argument("--settle-seconds", type=float, default=0.0, help="Optional pause after the fill-order check.")
    parser.add_argument(
        "--time-in-force",
        default="gtc",
        choices=["gtc", "ioc", "day"],
        help="Time in force for the paper validation orders. Crypto paper trading requires gtc or ioc.",
    )
    parser.add_argument("--no-reconnect", action="store_true", help="Skip the disconnect/reconnect verification.")
    args = parser.parse_args()

    if args.mode == "replay" and not args.fixture.exists():
        raise FileNotFoundError(f"Missing replay fixture: {args.fixture}")

    broker = _build_broker(args.mode, args.fixture)
    recorder = PaperSessionRecorder(
        broker,
        metadata={
            "mode": args.mode,
            "fixture": str(args.fixture),
            "symbol": args.symbol,
        },
    )
    open_order, fill_order = _build_orders(args)
    summary = run_paper_broker_checklist(
        recorder,
        open_order=open_order,
        fill_order=fill_order,
        reconnect=not args.no_reconnect,
        settle_seconds=args.settle_seconds,
    )

    if args.record_output is not None:
        recorder.export(args.record_output)

    errors = _validate_summary(summary, open_order, fill_order, require_reconnect=not args.no_reconnect)
    output = {
        "mode": args.mode,
        "fixture": str(args.fixture) if args.mode == "replay" else None,
        "recorded_to": str(args.record_output) if args.record_output is not None else None,
        "errors": errors,
        "summary": summary,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
