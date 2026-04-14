from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import ensure_project_root

ensure_project_root()

from src.backtest.engine import BacktestEngine
from src.brokers.mock import MockBroker
from src.brokers.paper import PaperBroker
from src.brokers.paper_session import PaperSessionRecorder, RecordedPaperSessionClient, run_paper_broker_checklist
from src.features.builder import build_technical_features
from src.ingestion.websocket_client import HFTStreamManager
from src.models.hmm import SimpleHMMRegimeDetector
from src.models.lgbm import LightGBMRanker
from src.monitoring.health import HealthMonitor
from src.risk.manager import RiskManager
from src.strategies.ml_strategy import MLStrategy
from src.backtest.matching_engine import HFTMatchingEngine
from src.brokers.mock import MockBroker


def _make_synthetic_ohlcv(rows: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range(end=datetime.now(timezone.utc).date(), periods=rows, freq="D")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.02, size=rows)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.01, size=rows))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.01, size=rows))
    volume = rng.uniform(1_000, 5_000, size=rows)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


def main() -> int:
    print("Running offline smoke test...")

    raw = _make_synthetic_ohlcv()
    features = build_technical_features(raw)
    features["target_return"] = features["return_1d"].shift(-1)
    features = features.dropna()

    hmm = SimpleHMMRegimeDetector()
    hmm.fit(features)
    hmm_pred = hmm.predict(features.iloc[[-1]])

    lgbm = LightGBMRanker()
    lgbm.fit(features)
    lgbm_pred = lgbm.predict(features.iloc[[-1]])

    strategy = MLStrategy(symbols=["BTC/USD", "ETH/USD"])
    predictions = {
        "BTC/USD": float(lgbm_pred[0]) if len(lgbm_pred) else 0.0,
        "ETH/USD": float(hmm_pred[0]) if len(hmm_pred) else 0.0,
    }
    weights = strategy.generate_weights(predictions)
    risk = RiskManager(max_position_cap=0.40)
    allowed, reasons = risk.pretrade_check(weights, current_drawdown=0.0, stale_data=False, max_gross_exposure=1.0)
    if not allowed:
        print(f"Pre-trade gate blocked raw weights as expected: {reasons}")
    capped = risk.apply_position_caps(weights)
    allowed_after_cap, reasons_after_cap = risk.pretrade_check(
        capped,
        current_drawdown=0.0,
        stale_data=False,
        max_gross_exposure=1.0,
    )
    if not allowed_after_cap:
        raise RuntimeError(f"Smoke pretrade gate still blocked after capping: {reasons_after_cap}")

    if not any(capped.values()):
        capped = {"BTC/USD": 1.0, "ETH/USD": 0.0}

    price_df = pd.DataFrame(
        {
            "BTC/USD": raw["close"],
            "ETH/USD": raw["close"] * 1.02,
        }
    )
    weight_df = pd.DataFrame([capped] * len(price_df), index=price_df.index)
    portfolio = BacktestEngine().run(price_df, weight_df)
    stats = portfolio.stats()

    broker = MockBroker()
    broker.connect()
    broker.place_order({"symbol": "BTC/USD", "qty": 1, "side": "buy"})
    fills = broker.get_fills()

    stream = HFTStreamManager(enable_live_stream=False)
    seen_features: list[dict] = []
    stream.on_feature_update = seen_features.append
    stream.replay_events(
        [
            {"type": "quote", "timestamp_ms": 1_000.0, "bid": 100.0, "bid_size": 5.0, "ask": 100.1, "ask_size": 4.0},
            {"type": "trade", "timestamp_ms": 1_010.0, "price": 100.05, "size": 1.2, "taker_side": "B"},
        ]
    )

    hft_engine = HFTMatchingEngine(latency_ms=5.0, fee_bps=0.0)
    hft_engine.place_limit_order("BTC", "buy", price=100.0, size=1.0, current_time_ms=100.0)
    hft_engine.process_quote_update(110.0, bid=99.9, ask=100.0)

    paper_fixture = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "paper" / "recorded_paper_session_sample.json"
    paper_summary = None
    if paper_fixture.exists():
        replay_client = RecordedPaperSessionClient.from_file(paper_fixture)
        paper_broker = PaperBroker(trading_client=replay_client, paper=True, price_provider=lambda symbol: 101.25)
        recorder = PaperSessionRecorder(
            paper_broker,
            metadata={
                "mode": "replay",
                "fixture": str(paper_fixture),
            },
        )
        paper_summary = run_paper_broker_checklist(
            recorder,
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

    fixture_path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hft" / "captured_replay_sample.json"
    if fixture_path.exists():
        replay_events = json.loads(fixture_path.read_text(encoding="utf-8"))
    else:
        replay_events = [
            {"type": "quote", "timestamp_ms": 1_000.0, "bid": 100.0, "bid_size": 5.0, "ask": 100.1, "ask_size": 4.0},
            {"type": "trade", "timestamp_ms": 1_010.0, "price": 100.05, "size": 1.2, "taker_side": "B"},
        ]

    monitor_broker = MockBroker()
    monitor_broker.connect()
    monitor_stream = HFTStreamManager(enable_live_stream=False)
    monitor_stream.replay_events(replay_events)
    monitor = HealthMonitor(
        broker=monitor_broker,
        stream_manager=monitor_stream,
        risk_manager=RiskManager(max_position_cap=0.40, max_drawdown=0.20),
        stale_after_seconds=30.0,
        failure_threshold=1,
    )
    health_loop = monitor.run_loop(iterations=2, interval_seconds=0.0)

    print("Smoke summary:")
    print(f"- HMM states: {len(np.unique(hmm.predict(features)))}")
    print(f"- LGBM prediction: {float(lgbm_pred[0]) if len(lgbm_pred) else 0.0:.6f}")
    print(f"- Backtest total return: {float(stats['Total Return [%]']):.4f}%")
    print(f"- MockBroker fills: {len(fills)}")
    print(f"- Stream features: {len(seen_features)}")
    print(f"- HFT inventory: {hft_engine.inventory}")
    if paper_summary is not None:
        print(f"- Paper replay duplicate blocked: {paper_summary['duplicate_blocking']['blocked']}")
        print(f"- Paper replay fills: {len(paper_summary['fill_reconciliation']['sync']['fills'])}")
    else:
        print("- Paper replay fixture: missing")
    print(f"- Health loop healthy: {health_loop[-1]['healthy']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
