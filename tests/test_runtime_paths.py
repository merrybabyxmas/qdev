import os
import asyncio
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from src.ingestion import loader
from src.ingestion.websocket_client import MultiSymbolHFTStreamManager as HFTStreamManager
from src.brokers.mock import MockBroker
from src.utils.config import SystemConfig


class BrokenClient:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("boom")


class TestRuntimePaths(unittest.TestCase):
    def test_live_mode_requires_explicit_flag(self):
        with patch.dict(os.environ, {"SYS_MODE": "live", "ALLOW_LIVE_TRADING": "false"}, clear=False):
            with self.assertRaises(ValueError):
                SystemConfig.load()

    def test_loader_fallback_generates_synthetic_data(self):
        with patch.object(loader, "CryptoHistoricalDataClient", BrokenClient):
            df = loader.fetch_data_alpaca("BTC/USD", "2024-01-01", "2024-01-10")

        self.assertFalse(df.empty)
        self.assertTrue({"open", "high", "low", "close", "volume"}.issubset(df.columns))

    def test_hft_stream_manager_replay_emits_features(self):
        manager = HFTStreamManager(symbols=["BTC/USD"], enable_live_stream=False)
        seen = []
        manager.on_feature_update = lambda sym, evt: seen.append(evt)

        manager.replay_events(
            [
                {"type": "quote", "symbol": "BTC/USD", "timestamp_ms": 1000.0, "bid": 100.0, "bid_size": 5.0, "ask": 100.1, "ask_size": 4.0},
                {"type": "trade", "symbol": "BTC/USD", "timestamp_ms": 1010.0, "price": 100.05, "size": 1.0, "taker_side": "B"},
            ]
        )

        self.assertGreaterEqual(len(seen), 2)
        self.assertIsNotNone(manager.last_feature_event.get("BTC/USD"))
        self.assertAlmostEqual(manager.last_feature_event["BTC/USD"]["spread"], 0.1, places=1)

    def test_live_hft_handlers_mark_event_receipts(self):
        manager = HFTStreamManager(symbols=["BTC/USD"], enable_live_stream=False)

        trade = SimpleNamespace(
            symbol="BTC/USD",
            timestamp=SimpleNamespace(timestamp=lambda: 1.0),
            price=100.0,
            size=1.0,
            side="B",
        )
        quote = SimpleNamespace(
            symbol="BTC/USD",
            timestamp=SimpleNamespace(timestamp=lambda: 2.0),
            bid_price=99.9,
            bid_size=5.0,
            ask_price=100.1,
            ask_size=4.0,
        )

        asyncio.run(manager._trade_handler(trade))
        first_seen = manager.last_event_received_at
        asyncio.run(manager._quote_handler(quote))

        self.assertIsNotNone(first_seen)
        self.assertIsNotNone(manager.last_event_received_at)
        self.assertGreaterEqual(manager.last_event_received_at, first_seen)
        self.assertIsNotNone(manager.last_feature_event.get("BTC/USD"))

    def test_mock_broker_records_fills(self):
        broker = MockBroker()
        broker.connect()

        order_id = broker.place_order({"symbol": "BTC/USD", "qty": 2, "side": "buy"})

        self.assertTrue(order_id.startswith("mock_"))
        self.assertEqual(broker.get_open_orders(), [])
        fills = broker.get_fills()
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]["symbol"], "BTC/USD")
        self.assertEqual(broker.get_positions()["BTC/USD"], 2.0)

    def test_pretrade_gate_blocks_stale_data_and_excess_exposure(self):
        from src.risk.manager import RiskManager

        risk = RiskManager(max_position_cap=0.25, max_drawdown=0.10)
        allowed, reasons = risk.pretrade_check(
            {"BTC/USD": 0.30, "ETH/USD": 0.20},
            current_drawdown=0.05,
            stale_data=True,
            max_gross_exposure=0.40,
        )

        self.assertFalse(allowed)
        self.assertTrue(any("stale_data" in reason for reason in reasons))
        self.assertTrue(any("position_cap" in reason for reason in reasons))
        self.assertTrue(any("gross_exposure" in reason for reason in reasons))
