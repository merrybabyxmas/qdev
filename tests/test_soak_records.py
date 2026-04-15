from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.brokers.mock import MockBroker
from src.ingestion.websocket_client import MultiSymbolHFTStreamManager as HFTStreamManager
from src.monitoring import HealthMonitor, SoakRecordStore, SoakRunner
from src.risk.manager import RiskManager


class TestSoakRecords(unittest.TestCase):
    def setUp(self):
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "hft" / "captured_replay_sample.json"
        self.events = json.loads(fixture_path.read_text(encoding="utf-8"))

    def _build_runner(self, record_root: Path) -> SoakRunner:
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
        return SoakRunner(monitor, SoakRecordStore(record_root), session_name="unit-test", metadata={"broker": "mock"})

    def test_soak_runner_appends_iteration_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = self._build_runner(Path(tmpdir))
            result = runner.run(iterations=2, interval_seconds=0.0)

            self.assertEqual(len(result.statuses), 2)
            self.assertTrue(result.statuses[-1]["healthy"])
            self.assertTrue(result.record_path.exists())

            records = SoakRecordStore(Path(tmpdir)).load_all()
            self.assertEqual([record["kind"] for record in records], ["run_start", "iteration", "iteration", "run_end"])
            self.assertEqual(records[0]["run_id"], result.run_id)
            self.assertEqual(records[-1]["summary"]["iterations_recorded"], 2)
            self.assertEqual(records[-1]["summary"]["healthy_count"], 2)

    def test_soak_store_is_append_only_across_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record_root = Path(tmpdir)
            runner = self._build_runner(record_root)

            first = runner.run(iterations=1, interval_seconds=0.0)
            second = runner.run(iterations=1, interval_seconds=0.0)

            records = SoakRecordStore(record_root).load_all()
            self.assertGreaterEqual(len(records), 6)
            run_ids = {record["run_id"] for record in records if "run_id" in record}
            self.assertIn(first.run_id, run_ids)
            self.assertIn(second.run_id, run_ids)
            self.assertNotEqual(first.run_id, second.run_id)


if __name__ == "__main__":
    unittest.main()
