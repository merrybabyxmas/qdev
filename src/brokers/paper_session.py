from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.brokers.base import BrokerInterface
from src.utils.logger import logger


def _now_iso() -> str:
    from datetime import datetime, timezone

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


def _coerce_mapping(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        return dict(obj.model_dump())
    if hasattr(obj, "dict"):
        return dict(obj.dict())
    return {key: value for key, value in vars(obj).items() if not key.startswith("_")}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_scalar(value: Any) -> Any:
    return getattr(value, "value", value)


def _normalize_positions(positions: Any) -> dict[str, float]:
    if isinstance(positions, dict):
        if "positions" in positions and isinstance(positions["positions"], list):
            positions = positions["positions"]
        else:
            return {str(symbol): _safe_float(qty) for symbol, qty in positions.items()}

    normalized: dict[str, float] = {}
    for position in positions or []:
        data = _coerce_mapping(position)
        symbol = str(data.get("symbol") or data.get("asset_id") or "").strip()
        if not symbol:
            continue
        qty = data.get("qty", data.get("net_qty", data.get("position_qty", 0.0)))
        normalized[symbol] = _safe_float(qty)
    return normalized


def _request_fingerprint(request: Mapping[str, Any]) -> str:
    return "|".join(
        [
            str(request.get("symbol", "")),
            str(_normalize_scalar(request.get("side", ""))).lower(),
            str(request.get("qty", request.get("notional", ""))),
            str(request.get("limit_price", "")),
            str(_normalize_scalar(request.get("type", "market"))).lower(),
            str(_normalize_scalar(request.get("time_in_force", ""))),
        ]
    )


def _orders_from_snapshot(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    orders: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in ("orders", "open_orders", "fills"):
        for order in snapshot.get(key, []) or []:
            data = _coerce_mapping(order)
            order_id = str(data.get("id") or data.get("order_id") or data.get("client_order_id") or uuid4())
            if order_id in seen:
                continue
            data.setdefault("id", order_id)
            data.setdefault("order_id", order_id)
            orders.append(data)
            seen.add(order_id)
    return orders


def _snapshot_to_state(snapshot: Mapping[str, Any], *, clock: Mapping[str, Any] | None = None) -> dict[str, Any]:
    state = {
        "account": _coerce_mapping(snapshot.get("account")),
        "positions": _normalize_positions(snapshot.get("positions")),
        "orders": _orders_from_snapshot(snapshot),
        "open_orders": [_coerce_mapping(order) for order in snapshot.get("open_orders", []) or []],
        "fills": [_coerce_mapping(order) for order in snapshot.get("fills", []) or []],
        "connected": snapshot.get("connected"),
        "paper": snapshot.get("paper", True),
        "synced_at": snapshot.get("synced_at"),
    }
    if clock is not None:
        state["clock"] = _coerce_mapping(clock)
    elif snapshot.get("clock") is not None:
        state["clock"] = _coerce_mapping(snapshot.get("clock"))
    return state


def _order_match(match: Mapping[str, Any], request: Mapping[str, Any]) -> bool:
    for key, expected in match.items():
        actual = _normalize_scalar(request.get(key))
        expected = _normalize_scalar(expected)
        if actual is None and expected is None:
            continue
        if actual is None or expected is None:
            return False
        actual_number = _safe_float(actual, default=float("nan"))
        expected_number = _safe_float(expected, default=float("nan"))
        if actual_number == actual_number and expected_number == expected_number:
            if abs(actual_number - expected_number) <= 1e-12:
                continue
        if str(actual) != str(expected):
            return False
    return True


class PaperSessionRecorder(BrokerInterface):
    """
    Wrap a broker and record every paper-session action into a replayable transcript.
    """

    def __init__(self, broker: BrokerInterface, *, metadata: Mapping[str, Any] | None = None):
        self.broker = broker
        self.metadata = dict(metadata or {})
        self.events: list[dict[str, Any]] = []
        self.initial_state: dict[str, Any] | None = None
        self.final_state: dict[str, Any] | None = None
        self.last_clock: dict[str, Any] | None = None
        self.connected: bool | None = getattr(broker, "connected", None)
        self._active_client_order_ids: set[str] = set()
        self._active_fingerprints: set[str] = set()
        self._active_orders: dict[str, dict[str, str]] = {}

    def _mark_active(self, order_id: str, request: Mapping[str, Any]) -> None:
        client_order_id = str(request.get("client_order_id") or order_id)
        fingerprint = _request_fingerprint(request)
        self._active_client_order_ids.add(client_order_id)
        self._active_fingerprints.add(fingerprint)
        self._active_orders[order_id] = {
            "client_order_id": client_order_id,
            "fingerprint": fingerprint,
        }

    def _clear_active(self, order_id: str | None = None, *, client_order_id: str | None = None, fingerprint: str | None = None) -> None:
        if order_id is not None and order_id in self._active_orders:
            active = self._active_orders.pop(order_id)
            self._active_client_order_ids.discard(active.get("client_order_id", ""))
            self._active_fingerprints.discard(active.get("fingerprint", ""))
        if client_order_id is not None:
            self._active_client_order_ids.discard(str(client_order_id))
        if fingerprint is not None:
            self._active_fingerprints.discard(str(fingerprint))

    def _record(self, action: str, *, request: Any | None = None, response: Any | None = None, error: Exception | None = None) -> None:
        event: dict[str, Any] = {
            "action": action,
            "timestamp": _now_iso(),
        }
        if request is not None:
            event["request"] = request
        if response is not None:
            event["response"] = response
        if error is not None:
            event["error"] = str(error)
        self.events.append(event)

    def _record_order_state(self, action: str, request: dict[str, Any], response: Any) -> dict[str, Any]:
        order_details: dict[str, Any] = {}
        try:
            if hasattr(self.broker, "get_open_orders"):
                for order in self.broker.get_open_orders() or []:
                    data = _coerce_mapping(order)
                    if str(data.get("client_order_id")) == str(request.get("client_order_id")):
                        order_details = data
                        break
                    if str(data.get("order_id") or data.get("id")) == str(response):
                        order_details = data
                        break
            if not order_details and hasattr(self.broker, "get_fills"):
                for order in self.broker.get_fills() or []:
                    data = _coerce_mapping(order)
                    if str(data.get("client_order_id")) == str(request.get("client_order_id")):
                        order_details = data
                        break
                    if str(data.get("order_id") or data.get("id")) == str(response):
                        order_details = data
                        break
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning("Paper session recorder failed to inspect order state", action=action, error=str(exc))

        if not order_details:
            order_details = dict(request)
            order_details["order_id"] = str(response)
            order_details.setdefault("id", str(response))
        return order_details

    def connect(self):
        try:
            result = self.broker.connect()
            self.connected = getattr(self.broker, "connected", None)
            response = {"connected": getattr(self.broker, "connected", None)}
            if result is not None:
                response["result"] = _coerce_mapping(result) if isinstance(result, Mapping) else result
            self._record("connect", response=response)
            return result
        except Exception as exc:
            self._record("connect", error=exc)
            raise

    def disconnect(self):
        try:
            result = self.broker.disconnect()
            self.connected = getattr(self.broker, "connected", None)
            response = {"connected": getattr(self.broker, "connected", None)}
            if result is not None:
                response["result"] = _coerce_mapping(result) if isinstance(result, Mapping) else result
            self._record("disconnect", response=response)
            return result
        except Exception as exc:
            self._record("disconnect", error=exc)
            raise

    def heartbeat(self):
        try:
            result = self.broker.heartbeat()
            payload = _coerce_mapping(result)
            clock = payload.get("clock")
            if isinstance(clock, dict):
                self.last_clock = dict(clock)
            self._record("heartbeat", response=payload)
            return result
        except Exception as exc:
            self._record("heartbeat", error=exc)
            raise

    def sync_state(self):
        try:
            result = self.broker.sync_state()
            payload = _coerce_mapping(result)
            open_orders = payload.get("open_orders", []) or []
            fills = payload.get("fills", []) or []
            for order in open_orders:
                data = _coerce_mapping(order)
                order_id = str(data.get("order_id") or data.get("id") or data.get("client_order_id") or uuid4())
                client_order_id = str(data.get("client_order_id") or order_id)
                fingerprint = _request_fingerprint(data)
                self._active_client_order_ids.add(client_order_id)
                self._active_fingerprints.add(fingerprint)
                self._active_orders.setdefault(
                    order_id,
                    {"client_order_id": client_order_id, "fingerprint": fingerprint},
                )
            for order in fills:
                data = _coerce_mapping(order)
                order_id = str(data.get("order_id") or data.get("id") or data.get("client_order_id") or uuid4())
                self._clear_active(
                    order_id,
                    client_order_id=data.get("client_order_id"),
                    fingerprint=_request_fingerprint(data),
                )
            if self.initial_state is None:
                self.initial_state = _snapshot_to_state(payload, clock=self.last_clock)
            self.final_state = _snapshot_to_state(payload, clock=self.last_clock)
            self._record("sync_state", response=payload)
            return result
        except Exception as exc:
            self._record("sync_state", error=exc)
            raise

    def get_account(self):
        try:
            result = self.broker.get_account()
            self._record("get_account", response=_coerce_mapping(result))
            return result
        except Exception as exc:
            self._record("get_account", error=exc)
            raise

    def get_positions(self):
        try:
            result = self.broker.get_positions()
            self._record("get_positions", response=_coerce_mapping(result))
            return result
        except Exception as exc:
            self._record("get_positions", error=exc)
            raise

    def get_latest_price(self, symbol):
        try:
            result = self.broker.get_latest_price(symbol)
            self._record("get_latest_price", request={"symbol": symbol}, response=result)
            return result
        except Exception as exc:
            self._record("get_latest_price", request={"symbol": symbol}, error=exc)
            raise

    def place_order(self, order):
        request = _coerce_mapping(order)
        try:
            client_order_id = str(request.get("client_order_id") or "")
            fingerprint = _request_fingerprint(request)
            if client_order_id and client_order_id in self._active_client_order_ids:
                raise ValueError(f"Duplicate order blocked by session recorder: {client_order_id}")
            if fingerprint in self._active_fingerprints:
                raise ValueError("Duplicate order blocked by session recorder: fingerprint match")
            result = self.broker.place_order(order)
            order_details = self._record_order_state("place_order", request, result)
            order_id = str(result)
            if str(request.get("type", "market")).lower() == "limit":
                self._mark_active(order_id, request)
            self._record(
                "place_order",
                request=request,
                response={
                    "order_id": str(result),
                    "order": order_details,
                },
            )
            return result
        except Exception as exc:
            self._record("place_order", request=request, error=exc)
            raise

    def cancel_order(self, order_id):
        request = {"order_id": order_id}
        try:
            result = self.broker.cancel_order(order_id)
            self._clear_active(str(order_id))
            open_orders = []
            fills = []
            if hasattr(self.broker, "get_open_orders"):
                open_orders = [_coerce_mapping(order) for order in self.broker.get_open_orders() or []]
            if hasattr(self.broker, "get_fills"):
                fills = [_coerce_mapping(order) for order in self.broker.get_fills() or []]
            self._record(
                "cancel_order",
                request=request,
                response={
                    "result": _coerce_mapping(result) if result is not None else None,
                    "open_orders": open_orders,
                    "fills": fills,
                },
            )
            return result
        except Exception as exc:
            self._record("cancel_order", request=request, error=exc)
            raise

    def get_open_orders(self):
        try:
            result = self.broker.get_open_orders()
            self._record("get_open_orders", response=[_coerce_mapping(order) for order in result or []])
            return result
        except Exception as exc:
            self._record("get_open_orders", error=exc)
            raise

    def get_fills(self):
        try:
            result = self.broker.get_fills()
            self._record("get_fills", response=[_coerce_mapping(order) for order in result or []])
            return result
        except Exception as exc:
            self._record("get_fills", error=exc)
            raise

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "kind": "paper_session_transcript",
            "captured_at": _now_iso(),
            "metadata": dict(self.metadata),
            "clock": self.last_clock,
            "initial_state": self.initial_state,
            "final_state": self.final_state,
            "scripted_responses": {
                "submit_order": [
                    {
                        "match": {
                            key: value
                            for key, value in {
                                "client_order_id": event.get("request", {}).get("client_order_id"),
                                "symbol": event.get("request", {}).get("symbol"),
                                "side": event.get("request", {}).get("side"),
                                "qty": event.get("request", {}).get("qty"),
                                "limit_price": event.get("request", {}).get("limit_price"),
                                "type": event.get("request", {}).get("type"),
                                "time_in_force": event.get("request", {}).get("time_in_force"),
                            }.items()
                            if value is not None
                        },
                        "response": event.get("response", {}).get("order"),
                    }
                    for event in self.events
                    if event.get("action") == "place_order" and "error" not in event and event.get("response", {}).get("order") is not None
                ]
            },
            "events": self.events,
        }

    def export(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        return path


class RecordedPaperSessionClient:
    """
    Lightweight replay client that mimics the Alpaca TradingClient methods used by PaperBroker.
    """

    def __init__(self, session: Mapping[str, Any] | Sequence[dict[str, Any]] | str | Path):
        if isinstance(session, (str, Path)):
            session = json.loads(Path(session).read_text(encoding="utf-8"))
        if isinstance(session, Sequence) and not isinstance(session, Mapping):
            session = {"events": list(session)}
        if not isinstance(session, Mapping):
            raise TypeError("RecordedPaperSessionClient requires a mapping, path, or event sequence.")

        self._session = dict(session)
        self.paper = bool(self._session.get("paper", True))
        self.connected = False

        initial_state = self._session.get("initial_state") or self._derive_initial_state()
        self._clock = _coerce_mapping(initial_state.get("clock") or self._session.get("clock") or {"timestamp": _now_iso(), "is_open": True})
        self._account = _coerce_mapping(initial_state.get("account"))
        self._positions = _normalize_positions(initial_state.get("positions"))
        self._orders = [_coerce_mapping(order) for order in initial_state.get("orders", []) or []]
        self._scripted_submissions = self._load_scripted_submissions()
        self._submitted_orders: list[dict[str, Any]] = []
        self._reindex_orders()

    @classmethod
    def from_file(cls, path: str | Path) -> "RecordedPaperSessionClient":
        return cls(Path(path))

    def _derive_initial_state(self) -> dict[str, Any]:
        clock = dict(self._session.get("clock") or {"timestamp": _now_iso(), "is_open": True})
        checkpoints = self._session.get("checkpoints")
        if isinstance(checkpoints, list) and checkpoints:
            first = checkpoints[0]
            if isinstance(first, Mapping):
                return _snapshot_to_state(first, clock=first.get("clock") or clock)

        events = self._session.get("events", [])
        for event in events:
            if not isinstance(event, Mapping):
                continue
            if event.get("action") == "sync_state":
                response = event.get("response", {})
                if isinstance(response, Mapping):
                    return _snapshot_to_state(response, clock=clock)
            if event.get("action") == "heartbeat":
                clock = event.get("response", {}).get("clock") if isinstance(event.get("response"), Mapping) else None
                if clock is not None:
                    clock = _coerce_mapping(clock)
        return {"account": {}, "positions": {}, "orders": [], "clock": clock}

    def _load_scripted_submissions(self) -> list[dict[str, Any]]:
        scripted = []
        scripted_payload = self._session.get("scripted_responses", {})
        if isinstance(scripted_payload, Mapping):
            for entry in scripted_payload.get("submit_order", []) or []:
                if isinstance(entry, Mapping):
                    response_payload = entry.get("response", {})
                    if isinstance(response_payload, Mapping) and "order" in response_payload:
                        response = _coerce_mapping(response_payload.get("order"))
                    else:
                        response = _coerce_mapping(response_payload)
                    scripted.append({
                        "match": _coerce_mapping(entry.get("match", {})),
                        "response": response,
                    })
        if scripted:
            return scripted

        for event in self._session.get("events", []) or []:
            if not isinstance(event, Mapping):
                continue
            if event.get("action") == "place_order" and "error" not in event:
                request = _coerce_mapping(event.get("request"))
                response_payload = event.get("response", {})
                if isinstance(response_payload, Mapping) and "order" in response_payload:
                    response = _coerce_mapping(response_payload.get("order"))
                else:
                    response = _coerce_mapping(response_payload)
                if request and response:
                    scripted.append({"match": request, "response": _coerce_mapping(response)})
        return scripted

    def _reindex_orders(self) -> None:
        self._orders_by_id: dict[str, dict[str, Any]] = {}
        self._orders_by_client_id: dict[str, dict[str, Any]] = {}
        for order in self._orders:
            order_id = str(order.get("id") or order.get("order_id") or order.get("client_order_id") or uuid4())
            order.setdefault("id", order_id)
            order.setdefault("order_id", order_id)
            self._orders_by_id[order_id] = order
            client_order_id = order.get("client_order_id")
            if client_order_id:
                self._orders_by_client_id[str(client_order_id)] = order

    def _apply_fill_effects(self, order: dict[str, Any]) -> None:
        status = str(order.get("status", "")).lower()
        filled_qty = _safe_float(order.get("filled_qty"))
        if status not in {"filled", "partially_filled", "closed"} and filled_qty <= 0:
            return

        symbol = str(order.get("symbol", "")).strip()
        if not symbol:
            return

        qty = filled_qty if filled_qty > 0 else _safe_float(order.get("qty"))
        side = str(order.get("side", "")).lower()
        signed_qty = qty if side == "buy" else -qty
        self._positions[symbol] = self._positions.get(symbol, 0.0) + signed_qty

        fill_price = _safe_float(order.get("filled_avg_price", order.get("limit_price", order.get("price", 0.0))))
        if fill_price > 0 and qty > 0:
            cash = _safe_float(self._account.get("cash", self._account.get("equity", 0.0)))
            cash += -(fill_price * qty) if side == "buy" else fill_price * qty
            self._account["cash"] = cash
            self._account.setdefault("equity", cash)

    def _fallback_submission(self, request: Mapping[str, Any]) -> dict[str, Any]:
        order_id = str(request.get("client_order_id") or f"replay-{uuid4().hex[:10]}")
        side = str(_normalize_scalar(request.get("side", ""))).lower()
        order_type = str(_normalize_scalar(request.get("type", "market"))).lower()
        status = "open" if order_type == "limit" or request.get("limit_price") is not None else "filled"
        qty = _safe_float(request.get("qty") or request.get("notional"))
        price = _safe_float(request.get("limit_price") or request.get("price") or request.get("notional"))
        order = {
            "id": order_id,
            "order_id": order_id,
            "client_order_id": request.get("client_order_id", order_id),
            "symbol": request.get("symbol"),
            "side": side,
            "qty": request.get("qty"),
            "notional": request.get("notional"),
            "type": order_type,
            "time_in_force": request.get("time_in_force", "day"),
            "status": status,
            "filled_qty": qty if status == "filled" else 0.0,
            "filled_avg_price": price if status == "filled" else None,
            "limit_price": request.get("limit_price"),
        }
        return order

    def _consume_scripted_response(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        for index, entry in enumerate(list(self._scripted_submissions)):
            match = entry.get("match", {})
            if isinstance(match, Mapping) and _order_match(match, request):
                self._scripted_submissions.pop(index)
                return _coerce_mapping(entry.get("response"))
        return None

    def get_clock(self):
        return dict(self._clock)

    def get_account(self):
        account = dict(self._account)
        account.setdefault("paper", self.paper)
        account.setdefault("connected", self.connected)
        return account

    def get_all_positions(self):
        return [{"symbol": symbol, "qty": qty} for symbol, qty in sorted(self._positions.items())]

    def get_orders(self, filter=None):
        return [dict(order) for order in self._orders]

    def get_order_by_client_id(self, client_id):
        order = self._orders_by_client_id.get(str(client_id))
        if order is None:
            raise KeyError(client_id)
        return dict(order)

    def submit_order(self, order_data):
        request = _coerce_mapping(order_data)
        response = self._consume_scripted_response(request) or self._fallback_submission(request)
        order_id = str(response.get("id") or response.get("order_id") or response.get("client_order_id") or uuid4())
        response = dict(response)
        response.setdefault("id", order_id)
        response.setdefault("order_id", order_id)
        response.setdefault("client_order_id", request.get("client_order_id", order_id))
        response.setdefault("symbol", request.get("symbol"))
        response.setdefault("side", _normalize_scalar(request.get("side")))
        response.setdefault("qty", request.get("qty"))
        response.setdefault("type", _normalize_scalar(request.get("type", "market")))
        response.setdefault("time_in_force", _normalize_scalar(request.get("time_in_force", "day")))
        order_type = str(request.get("type", "market")).lower()
        if order_type == "limit" or request.get("limit_price") is not None:
            response.setdefault("status", "open")
        else:
            response.setdefault("status", "filled")
        self._orders = [order for order in self._orders if str(order.get("id") or order.get("order_id")) != order_id]
        self._orders.append(response)
        self._reindex_orders()
        self._apply_fill_effects(response)
        self._submitted_orders.append({"request": request, "response": dict(response)})
        return dict(response)

    def cancel_order_by_id(self, order_id):
        candidate = str(order_id)
        order = self._orders_by_id.get(candidate) or self._orders_by_client_id.get(candidate)
        if order is None:
            raise KeyError(candidate)
        order["status"] = "canceled"
        self._reindex_orders()
        return {"order_id": candidate, "status": "canceled"}

    def state_snapshot(self) -> dict[str, Any]:
        return {
            "clock": dict(self._clock),
            "account": dict(self._account),
            "positions": dict(self._positions),
            "orders": [dict(order) for order in self._orders],
            "submitted_orders": list(self._submitted_orders),
        }


def run_paper_broker_checklist(
    broker: BrokerInterface,
    *,
    open_order: Mapping[str, Any] | None = None,
    fill_order: Mapping[str, Any] | None = None,
    reconnect: bool = True,
    settle_seconds: float = 0.0,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    broker.connect()
    summary["connect"] = {"connected": getattr(broker, "connected", None)}
    summary["heartbeat"] = _coerce_mapping(broker.heartbeat())
    summary["initial_sync"] = _coerce_mapping(broker.sync_state())

    if open_order is not None:
        open_order_request = dict(open_order)
        open_order_id = broker.place_order(open_order_request)
        summary["open_order"] = {
            "request": open_order_request,
            "order_id": open_order_id,
            "sync": _coerce_mapping(broker.sync_state()),
        }

        duplicate_blocked = False
        duplicate_error = None
        try:
            broker.place_order(open_order_request)
        except ValueError as exc:
            duplicate_blocked = True
            duplicate_error = str(exc)
        summary["duplicate_blocking"] = {
            "blocked": duplicate_blocked,
            "error": duplicate_error,
        }

        broker.cancel_order(open_order_id)
        summary["cancel_reconciliation"] = {
            "order_id": open_order_id,
            "sync": _coerce_mapping(broker.sync_state()),
        }

    if fill_order is not None:
        fill_order_request = dict(fill_order)
        fill_order_id = broker.place_order(fill_order_request)
        if settle_seconds > 0:
            time.sleep(settle_seconds)
        fill_sync = _coerce_mapping(broker.sync_state())
        summary["fill_reconciliation"] = {
            "request": fill_order_request,
            "order_id": fill_order_id,
            "sync": fill_sync,
            "fills": len(fill_sync.get("fills", []) or []),
        }

    if reconnect:
        broker.disconnect()
        disconnected_state = getattr(broker, "connected", None)
        broker.connect()
        summary["reconnect"] = {
            "disconnect": {"connected": disconnected_state},
            "connect": {"connected": getattr(broker, "connected", None)},
            "heartbeat": _coerce_mapping(broker.heartbeat()),
            "sync": _coerce_mapping(broker.sync_state()),
        }

    return summary
