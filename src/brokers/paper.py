from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable
from uuid import uuid4

try:  # pragma: no cover - optional dependency guard
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, OrderType, QueryOrderStatus, TimeInForce
    from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, MarketOrderRequest
except Exception:  # pragma: no cover - optional dependency guard
    TradingClient = None
    OrderSide = None
    OrderType = None
    QueryOrderStatus = None
    TimeInForce = None
    GetOrdersRequest = None
    LimitOrderRequest = None
    MarketOrderRequest = None

from src.brokers.base import BrokerInterface
from src.utils.logger import logger


def _coerce_mapping(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        return dict(obj.model_dump())
    if hasattr(obj, "dict"):
        return dict(obj.dict())
    return {key: value for key, value in vars(obj).items() if not key.startswith("_")}


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class PaperBroker(BrokerInterface):
    """
    Alpaca paper/live broker adapter with explicit heartbeat and order reconciliation.

    The adapter is deliberately testable: a fake trading client can be injected
    so unit tests do not need external credentials or connectivity.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
        *,
        allow_live: bool = False,
        trading_client: Any | None = None,
        price_provider: Callable[[str], float] | None = None,
        default_time_in_force: Any = None,
    ):
        if not paper and not allow_live:
            raise ValueError("PaperBroker requires allow_live=True before enabling live mode.")

        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self._client = trading_client
        self._price_provider = price_provider
        self._default_time_in_force = default_time_in_force or TimeInForce.DAY
        self.connected = False

        self._account_cache: Dict[str, Any] = {}
        self._positions_cache: Dict[str, float] = {}
        self._open_orders_cache: Dict[str, Dict[str, Any]] = {}
        self._fills_cache: Dict[str, Dict[str, Any]] = {}
        self._active_client_order_ids: set[str] = set()
        self._active_fingerprints: set[str] = set()
        self._latest_prices: Dict[str, float] = {}
        self.last_heartbeat_at: str | None = None
        self.last_sync_at: str | None = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if TradingClient is None:
            raise ImportError("alpaca-py is not installed; PaperBroker cannot create a TradingClient.")
        if not self.api_key or not self.secret_key:
            raise ValueError("PaperBroker requires api_key and secret_key unless a trading_client is injected.")
        self._client = TradingClient(api_key=self.api_key, secret_key=self.secret_key, paper=self.paper)
        return self._client

    @staticmethod
    def _fingerprint(order: Dict[str, Any]) -> str:
        return "|".join(
            [
                str(order.get("symbol", "")),
                str(order.get("side", "")).lower(),
                str(order.get("qty", order.get("notional", ""))),
                str(order.get("limit_price", "")),
                str(order.get("type", "market")).lower(),
                str(order.get("time_in_force", "")),
            ]
        )

    @staticmethod
    def _order_status(order: Dict[str, Any]) -> str:
        return str(order.get("status", "")).lower()

    @staticmethod
    def _is_active_status(status: str) -> bool:
        return status in {"new", "accepted", "open", "pending_new", "partially_filled"}

    def _record_order(self, order: Dict[str, Any]) -> None:
        order = dict(order)
        order_id = str(order.get("id") or order.get("order_id") or order.get("client_order_id") or uuid4())
        order["order_id"] = order_id
        client_order_id = order.get("client_order_id")
        if client_order_id:
            self._active_client_order_ids.add(str(client_order_id))

        fingerprint = self._fingerprint(order)
        status = self._order_status(order)
        is_active = self._is_active_status(status)
        if status in {"new", "accepted", "open", "pending_new", "partially_filled"}:
            self._open_orders_cache[order_id] = order
            self._active_fingerprints.add(fingerprint)
        else:
            self._open_orders_cache.pop(order_id, None)
            self._active_fingerprints.discard(fingerprint)

        filled_qty = _safe_float(order.get("filled_qty"))
        if status in {"filled", "partially_filled", "closed"} or filled_qty > 0:
            self._fills_cache[order_id] = order

        if client_order_id:
            if is_active:
                self._active_client_order_ids.add(str(client_order_id))
            else:
                self._active_client_order_ids.discard(str(client_order_id))

    def _fetch_orders(self) -> Iterable[Dict[str, Any]]:
        client = self._ensure_client()
        if GetOrdersRequest is None or QueryOrderStatus is None:
            raw_orders = client.get_orders()
        else:
            raw_orders = client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500, nested=True)
            )
        if isinstance(raw_orders, dict):
            raw_orders = raw_orders.get("orders", [])
        return [_coerce_mapping(order) for order in raw_orders]

    def _fetch_positions(self) -> Dict[str, float]:
        client = self._ensure_client()
        raw_positions = client.get_all_positions()
        if isinstance(raw_positions, dict):
            raw_positions = raw_positions.get("positions", [])

        positions: Dict[str, float] = {}
        for position in raw_positions or []:
            data = _coerce_mapping(position)
            symbol = str(data.get("symbol") or data.get("asset_id") or "").strip()
            if not symbol:
                continue
            qty = data.get("qty", data.get("net_qty", data.get("position_qty", 0.0)))
            positions[symbol] = _safe_float(qty)
        return positions

    def _fetch_account(self) -> Dict[str, Any]:
        client = self._ensure_client()
        account = _coerce_mapping(client.get_account())
        account.update(
            {
                "paper": self.paper,
                "connected": self.connected,
                "last_heartbeat_at": self.last_heartbeat_at,
                "last_sync_at": self.last_sync_at,
            }
        )
        return account

    def connect(self):
        self._ensure_client()
        self.connected = True
        self.heartbeat()
        self.sync_state()
        logger.info("PaperBroker connected", paper=self.paper)

    def disconnect(self):
        self.connected = False
        logger.info("PaperBroker disconnected", paper=self.paper)

    def heartbeat(self):
        client = self._ensure_client()
        clock_payload: Dict[str, Any] = {}
        try:
            clock_payload = _coerce_mapping(client.get_clock())
            ok = True
        except Exception as exc:  # pragma: no cover - network failure path
            ok = False
            clock_payload = {"error": str(exc)}

        now = datetime.now(timezone.utc).isoformat()
        self.last_heartbeat_at = now
        self.connected = self.connected or ok
        heartbeat = {
            "ok": ok,
            "source": "alpaca",
            "paper": self.paper,
            "connected": self.connected,
            "timestamp": now,
            "clock": clock_payload,
        }
        logger.info("PaperBroker heartbeat", heartbeat=heartbeat)
        return heartbeat

    def sync_state(self) -> Dict[str, Any]:
        client = self._ensure_client()
        now = datetime.now(timezone.utc).isoformat()

        account = self._fetch_account()
        positions = self._fetch_positions()
        open_orders: Dict[str, Dict[str, Any]] = dict(self._open_orders_cache)
        fills: Dict[str, Dict[str, Any]] = {}
        active_client_ids: set[str] = set(self._active_client_order_ids)
        active_fingerprints: set[str] = set(self._active_fingerprints)

        for order in self._fetch_orders():
            order_id = str(order.get("id") or order.get("order_id") or order.get("client_order_id") or uuid4())
            order["order_id"] = order_id
            fingerprint = self._fingerprint(order)
            status = self._order_status(order)
            client_order_id = order.get("client_order_id")
            is_active = self._is_active_status(status)
            if is_active:
                open_orders[order_id] = order
                if client_order_id:
                    active_client_ids.add(str(client_order_id))
                active_fingerprints.add(fingerprint)
            else:
                open_orders.pop(order_id, None)
                if client_order_id:
                    active_client_ids.discard(str(client_order_id))
                active_fingerprints.discard(fingerprint)
            if status in {"filled", "partially_filled", "closed"} or _safe_float(order.get("filled_qty")) > 0:
                fills[order_id] = order

        self._account_cache = account
        self._positions_cache = positions
        self._open_orders_cache = open_orders
        self._fills_cache = {**self._fills_cache, **fills}
        self._active_client_order_ids = active_client_ids
        self._active_fingerprints = active_fingerprints
        self.last_sync_at = now

        snapshot = {
            "account": account,
            "positions": dict(positions),
            "open_orders": list(open_orders.values()),
            "fills": list(self._fills_cache.values()),
            "connected": self.connected,
            "paper": self.paper,
            "synced_at": now,
        }
        logger.info(
            "PaperBroker synced state",
            connected=self.connected,
            open_orders=len(open_orders),
            fills=len(self._fills_cache),
        )
        return snapshot

    def get_account(self) -> Dict[str, Any]:
        if self.connected:
            self.sync_state()
        return dict(self._account_cache)

    def get_positions(self) -> Dict[str, Any]:
        if self.connected:
            self.sync_state()
        return dict(self._positions_cache)

    def get_latest_price(self, symbol: str) -> float:
        if symbol in self._latest_prices:
            return self._latest_prices[symbol]
        if self._price_provider is not None:
            price = _safe_float(self._price_provider(symbol))
            if price > 0:
                self._latest_prices[symbol] = price
                return price
        raise ValueError(
            f"No latest price available for {symbol}. Inject a price_provider or update the local cache first."
        )

    def update_latest_price(self, symbol: str, price: float) -> None:
        self._latest_prices[symbol] = _safe_float(price)

    def place_order(self, order: Dict[str, Any]) -> str:
        client = self._ensure_client()
        required = {"symbol", "side"}
        missing = required - set(order)
        if missing:
            raise ValueError(f"Order is missing required fields: {sorted(missing)}")

        side = str(order.get("side", "")).lower()
        if side not in {"buy", "sell"}:
            raise ValueError("Order side must be 'buy' or 'sell'")

        qty = order.get("qty")
        notional = order.get("notional")
        if qty is None and notional is None:
            raise ValueError("Order must include either qty or notional")

        client_order_id = str(order.get("client_order_id") or f"paper-{uuid4().hex[:12]}")
        fingerprint = self._fingerprint({**order, "client_order_id": client_order_id})
        if client_order_id in self._active_client_order_ids or fingerprint in self._active_fingerprints:
            raise ValueError(f"Duplicate order blocked by PaperBroker: {client_order_id}")

        order_type = str(order.get("type", "market")).lower()
        tif = _enum_value(order.get("time_in_force", self._default_time_in_force))
        side_enum = OrderSide.BUY if side == "buy" else OrderSide.SELL
        limit_price = order.get("limit_price")

        request_kwargs = {
            "symbol": str(order["symbol"]),
            "qty": _safe_float(qty) if qty is not None else None,
            "notional": _safe_float(notional) if notional is not None else None,
            "side": side_enum,
            "time_in_force": tif,
            "client_order_id": client_order_id,
        }

        if order_type == "limit" or limit_price is not None:
            request_kwargs["type"] = OrderType.LIMIT
            request_kwargs["limit_price"] = _safe_float(limit_price)
            request = LimitOrderRequest(**request_kwargs)
        else:
            request_kwargs["type"] = OrderType.MARKET
            request = MarketOrderRequest(**request_kwargs)

        response = client.submit_order(request)
        order_dict = _coerce_mapping(response)
        order_dict.setdefault("client_order_id", client_order_id)
        order_id = str(order_dict.get("id") or order_dict.get("order_id") or client_order_id)
        self._record_order(order_dict)
        self.sync_state()

        if order_type == "limit":
            self._open_orders_cache[order_id] = order_dict
            self._active_client_order_ids.add(client_order_id)
            self._active_fingerprints.add(self._fingerprint(order_dict))

        logger.info("PaperBroker placed order", order_id=order_id, order=order_dict)
        return order_id

    def cancel_order(self, order_id: str):
        client = self._ensure_client()
        if self.connected:
            self.sync_state()

        candidate = order_id
        if candidate not in self._open_orders_cache:
            if hasattr(client, "get_order_by_client_id"):
                try:
                    resolved = _coerce_mapping(client.get_order_by_client_id(order_id))
                    candidate = str(resolved.get("id") or resolved.get("order_id") or order_id)
                except Exception:
                    pass
            if candidate in self._open_orders_cache:
                pass
            else:
                for cached in list(self._open_orders_cache.values()):
                    if str(cached.get("client_order_id")) == order_id:
                        candidate = str(cached.get("id") or cached.get("order_id") or order_id)
                        break

        if candidate in self._open_orders_cache:
            order = self._open_orders_cache[candidate]
            self._active_fingerprints.discard(self._fingerprint(order))
            self._active_client_order_ids.discard(str(order.get("client_order_id", "")))

        client.cancel_order_by_id(candidate)
        self._open_orders_cache.pop(candidate, None)
        self._fills_cache.pop(candidate, None)
        logger.info("PaperBroker canceled order", order_id=candidate)

    def get_open_orders(self) -> list[Dict[str, Any]]:
        if self.connected:
            self.sync_state()
        return list(self._open_orders_cache.values())

    def get_fills(self) -> list[Dict[str, Any]]:
        if self.connected:
            self.sync_state()
        return list(self._fills_cache.values())
