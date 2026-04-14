from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.brokers.mock import MockBroker
from src.ingestion.websocket_client import HFTStreamManager
from src.monitoring.health import HealthMonitor
from src.risk.manager import RiskManager


class TestMonitoring(unittest.TestCase):
    def setUp(self):
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "hft" / "captured_replay_sample.json"
        self.events = json.loads(fixture_path.read_text(encoding="utf-8"))

    def test_health_monitor_handles_fresh_and_stale_streams(self):
        broker = MockBroker()
        broker.connect()

        stream = HFTStreamManager(enable_live_stream=False)
        stream.replay_events(self.events)

        monitor = HealthMonitor(
            broker=broker,
            stream_manager=stream,
            risk_manager=RiskManager(max_position_cap=0.40, max_drawdown=0.20),
            stale_after_seconds=30.0,
            failure_threshold=1,
        )

        results = monitor.run_loop(iterations=2, interval_seconds=0.0)
        self.assertEqual(len(results), 2)
        self.assertTrue(results[-1]["healthy"])
        self.assertTrue(results[-1]["broker"]["healthy"])
        self.assertTrue(results[-1]["stream"]["healthy"])

        stream.last_event_received_at -= 1_000.0
        stale_status = monitor.run_once()
        self.assertFalse(stale_status["healthy"])
        self.assertTrue(stale_status["failure_count"] >= 1)
        self.assertTrue(monitor.risk_manager.kill_switch_active)


if __name__ == "__main__":
    unittest.main()
