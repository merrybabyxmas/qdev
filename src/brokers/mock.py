from datetime import datetime, timezone
from typing import Dict, Any, List

from src.brokers.base import BrokerInterface
from src.utils.logger import logger

class MockBroker(BrokerInterface):
    def __init__(self):
        self.connected = False
        self.positions = {}
        self.orders = {}
        self.fills = []
        self.account_balance = 100000.0
        self.last_heartbeat_at: str | None = None

    def connect(self):
        self.connected = True
        logger.info("MockBroker connected")

    def disconnect(self):
        self.connected = False
        logger.info("MockBroker disconnected")

    def heartbeat(self):
        now = datetime.now(timezone.utc).isoformat()
        self.last_heartbeat_at = now
        return {
            "ok": self.connected,
            "source": "mock",
            "connected": self.connected,
            "timestamp": now,
        }

    def get_account(self) -> Dict[str, Any]:
        return {
            "balance": self.account_balance,
            "equity": self.account_balance,
            "buying_power": self.account_balance * 2.0,
            "connected": self.connected,
            "paper": True,
        }

    def get_positions(self) -> Dict[str, Any]:
        return dict(self.positions)

    def get_latest_price(self, symbol: str) -> float:
        # Mock price
        return 150.0

    def place_order(self, order: Dict[str, Any]) -> str:
        required = {"symbol", "qty", "side"}
        missing = required - set(order)
        if missing:
            raise ValueError(f"Order is missing required fields: {sorted(missing)}")

        side = order.get("side")
        if side not in {"buy", "sell"}:
            raise ValueError("Order side must be 'buy' or 'sell'")

        qty = float(order.get("qty", 0))
        if qty <= 0:
            raise ValueError("Order quantity must be positive")

        order_id = f"mock_{len(self.orders) + 1}"
        if not self.connected:
            logger.warning("MockBroker.place_order called before connect(); proceeding in mock mode")

        price = self.get_latest_price(order["symbol"])
        fill = {
            "order_id": order_id,
            "symbol": order["symbol"],
            "qty": qty,
            "side": side,
            "price": price,
            "status": "filled",
        }
        self.orders[order_id] = {**order, "status": "filled", "avg_fill_price": price}
        self.fills.append(fill)
        logger.info("MockBroker placed order", order_id=order_id, order=order, fill=fill)

        if side == "buy":
            self.positions[order["symbol"]] = self.positions.get(order["symbol"], 0) + qty
            self.account_balance -= price * qty
        else:
            self.positions[order["symbol"]] = self.positions.get(order["symbol"], 0) - qty
            self.account_balance += price * qty

        return order_id

    def cancel_order(self, order_id: str):
        if order_id in self.orders:
            del self.orders[order_id]
            logger.info("MockBroker canceled order", order_id=order_id)

    def sync_state(self) -> Dict[str, Any]:
        return {
            "account": self.get_account(),
            "positions": self.get_positions(),
            "open_orders": self.get_open_orders(),
            "fills": self.get_fills(),
            "connected": self.connected,
        }

    def get_open_orders(self) -> List[Dict[str, Any]]:
        return [dict(order, order_id=order_id) for order_id, order in self.orders.items() if order.get("status") != "filled"]

    def get_fills(self) -> List[Dict[str, Any]]:
        return list(self.fills)
