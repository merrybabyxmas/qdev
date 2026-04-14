from typing import Dict, Any, List
from src.brokers.base import BrokerInterface
from src.utils.logger import logger

class MockBroker(BrokerInterface):
    def __init__(self):
        self.connected = False
        self.positions = {}
        self.orders = {}
        self.account_balance = 100000.0

    def connect(self):
        self.connected = True
        logger.info("MockBroker connected")

    def disconnect(self):
        self.connected = False
        logger.info("MockBroker disconnected")

    def get_account(self) -> Dict[str, Any]:
        return {"balance": self.account_balance, "equity": self.account_balance}

    def get_positions(self) -> Dict[str, Any]:
        return self.positions

    def get_latest_price(self, symbol: str) -> float:
        # Mock price
        return 150.0

    def place_order(self, order: Dict[str, Any]) -> str:
        order_id = f"mock_{len(self.orders) + 1}"
        self.orders[order_id] = order
        logger.info("MockBroker placed order", order_id=order_id, order=order)

        # Auto-fill mock logic
        symbol = order.get("symbol")
        qty = order.get("qty", 0)
        side = order.get("side")
        price = self.get_latest_price(symbol)

        if side == "buy":
            self.positions[symbol] = self.positions.get(symbol, 0) + qty
            self.account_balance -= price * qty
        elif side == "sell":
            self.positions[symbol] = self.positions.get(symbol, 0) - qty
            self.account_balance += price * qty

        return order_id

    def cancel_order(self, order_id: str):
        if order_id in self.orders:
            del self.orders[order_id]
            logger.info("MockBroker canceled order", order_id=order_id)

    def get_open_orders(self) -> List[Dict[str, Any]]:
        return []

    def get_fills(self) -> List[Dict[str, Any]]:
        return [{"order_id": k, "status": "filled"} for k in self.orders.keys()]
