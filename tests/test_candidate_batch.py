from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from hashlib import sha256

import numpy as np
import pandas as pd

from src.evaluation import EXPERIMENTS, CandidateBatchRunner, DatasetSpec, build_dataset_bundle, finalize_decisions, results_to_frame


def fake_loader(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    seed = int.from_bytes(sha256(symbol.encode("utf-8")).digest()[:8], "big") % (2**32 - 1)
    rng = np.random.default_rng(seed)
    base = 100.0 + (seed % 11)
    drift = 0.0008 + (seed % 5) * 0.0001
    shocks = rng.normal(drift, 0.02, size=len(dates))
    close = base * np.exp(np.cumsum(shocks))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.01, size=len(dates)))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.01, size=len(dates)))
    volume = rng.uniform(1_000, 10_000, size=len(dates))
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    frame.index.name = "date"
    frame.attrs["data_source"] = "synthetic_test"
    return frame


class TestCandidateBatch(unittest.TestCase):
    def test_dataset_bundle_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = DatasetSpec(
                symbols=("BTC/USD", "ETH/USD", "SOL/USD"),
                start_date="2024-01-01",
                end_date="2024-02-29",
            )
            bundle = build_dataset_bundle(spec, cache_root=Path(tmpdir), data_loader=fake_loader, refresh=True)
            self.assertFalse(bundle.frame.empty)
            self.assertIn("target_return", bundle.frame.columns)
            self.assertIn("factor_proxy", bundle.frame.columns)
            self.assertGreater(bundle.frame["symbol"].nunique(), 1)

            loaded = bundle.load(bundle.root)
            self.assertEqual(loaded.version, bundle.version)
            self.assertEqual(len(loaded.frame), len(bundle.frame))
            split = loaded.split()
            self.assertFalse(split.train.empty)
            self.assertFalse(split.validation.empty)
            self.assertFalse(split.test.empty)

    def test_candidate_batch_runner_returns_decisions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = DatasetSpec(
                symbols=("BTC/USD", "ETH/USD", "SOL/USD"),
                start_date="2024-01-01",
                end_date="2024-03-15",
            )
            bundle = build_dataset_bundle(spec, cache_root=Path(tmpdir), data_loader=fake_loader, refresh=True)
            runner = CandidateBatchRunner(bundle)
            selected = [experiment for experiment in EXPERIMENTS if experiment.pipeline_id in {"BASE_EQ", "F001", "F023", "B001"}]
            result = finalize_decisions(runner.run(selected, baseline_pipeline_id="F001"))
            frame = results_to_frame(result)

            self.assertEqual(set(frame["pipeline_id"]), {"BASE_EQ", "F001", "F023", "B001"})
            self.assertEqual(frame.loc[frame["pipeline_id"] == "F001", "decision"].iloc[0], "reference")
            self.assertTrue(set(frame["decision"]).issubset({"reference", "promote", "keep", "archive"}))
            self.assertIn("test_summary.total_return_pct", frame.columns)


if __name__ == "__main__":
    unittest.main()
