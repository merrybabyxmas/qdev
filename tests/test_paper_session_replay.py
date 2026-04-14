from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.brokers import MockBroker, PaperBroker, PaperSessionRecorder, RecordedPaperSessionClient, run_paper_broker_checklist


class TestPaperSessionReplay(unittest.TestCase):
    def test_recorded_sample_fixture_replays_full_paper_checklist(self):
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "paper" / "recorded_paper_session_sample.json"
        client = RecordedPaperSessionClient.from_file(fixture_path)
        broker = PaperBroker(trading_client=client, paper=True, price_provider=lambda symbol: 101.25)

        summary = run_paper_broker_checklist(
            broker,
            open_order={
                "symbol": "BTC/USD",
                "qty": 0.0001,
                "side": "buy",
                "type": "limit",
                "limit_price": 1.0,
                "time_in_force": "day",
                "client_order_id": "paper-open-002",
            },
            fill_order={
                "symbol": "BTC/USD",
                "qty": 0.0001,
                "side": "buy",
                "type": "market",
                "time_in_force": "day",
                "client_order_id": "paper-fill-002",
            },
            reconnect=True,
            settle_seconds=0.0,
        )

        self.assertTrue(summary["connect"]["connected"])
        self.assertTrue(summary["heartbeat"]["ok"])
        self.assertEqual(len(summary["initial_sync"]["open_orders"]), 1)
        self.assertEqual(len(summary["initial_sync"]["fills"]), 1)
        self.assertTrue(summary["duplicate_blocking"]["blocked"])
        self.assertFalse(
            any(
                str(order.get("order_id") or order.get("id")) == str(summary["open_order"]["order_id"]) or str(order.get("client_order_id")) == "paper-open-002"
                for order in summary["cancel_reconciliation"]["sync"]["open_orders"]
            )
        )
        self.assertTrue(
            any(
                str(order.get("order_id") or order.get("id")) == str(summary["fill_reconciliation"]["order_id"])
                or str(order.get("client_order_id")) == "paper-fill-002"
                for order in summary["fill_reconciliation"]["sync"]["fills"]
            )
        )
        self.assertTrue(summary["reconnect"]["heartbeat"]["ok"])

    def test_session_recorder_exports_replayable_transcript(self):
        broker = MockBroker()
        recorder = PaperSessionRecorder(broker, metadata={"source": "unit-test"})

        recorder.connect()
        recorder.sync_state()
        recorder.place_order(
            {
                "symbol": "BTC/USD",
                "qty": 2,
                "side": "buy",
                "client_order_id": "recorder-test-001",
            }
        )
        recorder.sync_state()
        recorder.disconnect()

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "paper_session_transcript.json"
            recorder.export(export_path)

            replay_client = RecordedPaperSessionClient.from_file(export_path)
            replay_broker = PaperBroker(trading_client=replay_client, paper=True, price_provider=lambda symbol: 123.0)
            replay_broker.connect()
            replay_snapshot = replay_broker.sync_state()

            order_id = replay_broker.place_order(
                {
                    "symbol": "BTC/USD",
                    "qty": 2,
                    "side": "buy",
                    "client_order_id": "recorder-test-001",
                }
            )
            replay_sync = replay_broker.sync_state()

            self.assertEqual(order_id, "mock_1")
            self.assertTrue(replay_snapshot["account"])
            self.assertTrue(any(str(order.get("client_order_id")) == "recorder-test-001" for order in replay_sync["fills"]))


if __name__ == "__main__":
    unittest.main()
