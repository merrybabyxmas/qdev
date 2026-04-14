from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.brokers.paper import PaperBroker


class FakeTradingClient:
    def __init__(self):
        self.submitted = []
        self.orders = []
        self.cancelled = []
        self.account = {
            "equity": "100000.0",
            "cash": "99500.0",
            "buying_power": "200000.0",
            "portfolio_value": "100000.0",
        }
        self.positions = [SimpleNamespace(symbol="BTC/USD", qty="1.0")]
        self.clock = {"timestamp": "2024-01-01T00:00:00Z", "is_open": True}

    def get_clock(self):
        return self.clock

    def get_account(self):
        return self.account

    def get_all_positions(self):
        return self.positions

    def get_orders(self, filter=None):
        return list(self.orders)

    def submit_order(self, order_data):
        order = SimpleNamespace(
            id=f"order-{len(self.submitted) + 1}",
            client_order_id=order_data.client_order_id,
            symbol=order_data.symbol,
            side=getattr(order_data.side, "value", order_data.side),
            qty=str(getattr(order_data, "qty", "")),
            filled_qty="0",
            filled_avg_price=None,
            status="open",
            type=getattr(order_data.type, "value", order_data.type),
        )
        self.submitted.append(order_data)
        self.orders.append(order)
        return order

    def cancel_order_by_id(self, order_id):
        self.cancelled.append(str(order_id))
        self.orders = [order for order in self.orders if str(order.id) != str(order_id)]

    def get_order_by_client_id(self, client_id):
        for order in self.orders:
            if str(order.client_order_id) == str(client_id):
                return order
        raise KeyError(client_id)


class TestPaperBroker(unittest.TestCase):
    def test_heartbeat_sync_duplicate_blocking_and_cancel(self):
        client = FakeTradingClient()
        broker = PaperBroker(
            trading_client=client,
            paper=True,
            price_provider=lambda symbol: 101.25,
        )

        broker.connect()
        heartbeat = broker.heartbeat()
        self.assertTrue(heartbeat["ok"])

        snapshot = broker.sync_state()
        self.assertEqual(snapshot["positions"]["BTC/USD"], 1.0)
        self.assertEqual(broker.get_latest_price("BTC/USD"), 101.25)

        order_id = broker.place_order(
            {
                "symbol": "BTC/USD",
                "qty": 1.0,
                "side": "buy",
                "client_order_id": "paper-test-001",
            }
        )
        self.assertEqual(order_id, "order-1")
        self.assertEqual(len(broker.get_open_orders()), 1)

        with self.assertRaises(ValueError):
            broker.place_order(
                {
                    "symbol": "BTC/USD",
                    "qty": 1.0,
                    "side": "buy",
                    "client_order_id": "paper-test-001",
                }
            )

        broker.cancel_order("paper-test-001")
        self.assertEqual(broker.get_open_orders(), [])
        self.assertEqual(client.cancelled, ["order-1"])


if __name__ == "__main__":
    unittest.main()
