# QDev Execution Report

Date: 2026-04-14

## Summary

I reconciled the repository with the markdown specs, downloaded an actual paper/reference bundle, populated the pipeline inventory docs from the master spec, and added the missing broker/monitor/replay pieces needed for a safer paper-trading path.

Current state in one line: the repo now runs end-to-end in research/mock/offline modes, has a real reference-backed pipeline inventory, has a testable paper-broker adapter, and its main scripts run directly without `PYTHONPATH`.
Update: the Alpaca paper account has since been externally validated with the live paper checklist, so the repo now has both a real paper run and a recorded replay path.

## What I Found

- The pipeline-library docs were mostly template placeholders before this round.
- The codebase already had strong research plumbing:
  - technical feature generation
  - HMM regime detection
  - LightGBM ranking
  - risk capping / pre-trade gating
  - vectorbt backtesting
  - HFT matching / ring buffer utilities
- Direct script execution was fragile before the bootstrap work, but the new `_bootstrap` helper and packaging layout now let the main scripts run directly.
- There was no project-standard bootstrap, env check, smoke test, or editable package metadata until the earlier rounds.
- The repo still lacked:
  - a real paper broker adapter
  - a heartbeat/order-sync layer
  - a runtime health loop
  - a captured replay fixture for HFT monitoring
  - reference-backed pipeline docs

## What I Changed

### Docs / References

- Downloaded 18 primary-source paper/archive PDFs into `docs/references/papers/`.
- Created `docs/references/reference_manifest.json` and `docs/references/index.md`.
- Rewrote `scripts/generate_pipeline_docs.py` so it parses `알고리즘_파이프라인_후보_라이브러리.md` and renders populated docs instead of templates.
- Populated:
  - `docs/pipeline_library/00_pipeline_catalog.md`
  - `docs/pipeline_library/01_bayesian_pipelines.md`
  - `docs/pipeline_library/02_sde_pipelines.md`
  - `docs/pipeline_library/03_rl_pipelines.md`
  - `docs/pipeline_library/04_financial_dl_pipelines.md`
  - `docs/pipeline_library/05_priority_shortlist.md`
  - `docs/pipeline_library/06_rejected_or_archived_pipelines.md`
- Expanded `docs/pipeline_library/05_hft_pipelines.md` with paper references and repo coverage notes.

### Broker / Execution

- Added `src/brokers/paper.py`:
  - Alpaca `TradingClient` wrapper
  - explicit `heartbeat()`
  - state reconciliation via `sync_state()`
  - open-order / fill tracking
  - duplicate order blocking
  - cancel resolution by client order id or broker order id
  - fake-client injection for offline tests
- Adjusted `PaperBroker` for crypto paper trading rules:
  - `gtc`-safe order submission
  - minimum-notional-aware checklist orders
  - active order retention across eventual-consistency sync windows
- Added `heartbeat()` and `sync_state()` support to `MockBroker`.
- Exported broker classes from `src/brokers/__init__.py`.
- Added `src/brokers/paper_session.py`:
  - `PaperSessionRecorder` for transcript capture
  - `RecordedPaperSessionClient` for offline replay
  - `run_paper_broker_checklist()` covering connect, heartbeat, sync, duplicate blocking, cancel reconciliation, fill reconciliation, and reconnect
- Added repo-local broker config loading:
  - `src/utils/env.py`
  - `.env` with Alpaca paper credentials
- Added operational runbooks under `docs/09_runbooks/`:
  - `paper_soak_runbook.md`
  - `live_trading_runbook.md`
- Added `scripts/validate_paper_broker.py`:
  - `--mode replay` for safe offline validation
  - `--mode paper` for real external validation when broker credentials exist
- Added a recorded replay fixture at `tests/fixtures/paper/recorded_paper_session_sample.json`.
- Added `tests/test_paper_session_replay.py` for recorder/replay round-trip coverage.

### Monitoring / Replay

- Added `src/monitoring/health.py`:
  - broker heartbeat check
  - stream freshness check
  - stale-data detection
  - drawdown / kill-switch wiring
  - loop runner and summary helpers
- Added `scripts/monitor_health.py` as a safe offline monitor entrypoint.
- Added `last_event_received_at` tracking to `HFTStreamManager`.
- Added a captured replay fixture at `tests/fixtures/hft/captured_replay_sample.json`.

### Runtime Hardening Already Present from the Earlier Round

- Editable install and bootstrap flow
- env validation
- synthetic Alpaca fallback
- replayable HFT stream manager
- model save/load
- pre-trade gate
- stronger backtest/matching checks

## Spec vs Code

| Area | Status | Notes |
|---|---|---|
| Core / config | Implemented and safer | Live mode still requires explicit environment gating. |
| Data | Mostly implemented | Alpaca loader has a deterministic synthetic fallback. |
| Features / labels | Implemented | Technical feature builder has required-column checks. |
| Models | Implemented | HMM and LightGBM support save/load and missing-feature fallback. |
| Strategies / portfolio | Implemented | `MLStrategy` + risk caps are validated in tests and smoke. |
| Risk | Improved | Pre-trade gating blocks stale data and excess exposure. |
| Orders / brokers | Improved | Mock broker is solid; paper broker adapter now exists and is unit-tested. |
| Backtest | Implemented | Vectorbt backtest path runs successfully. |
| Paper / live | Partial | Paper adapter exists, but no external credentialed broker was exercised. |
| HFT / streaming | Implemented for offline replay | Live stream is guarded; replay/captured fixture validation exists. |
| Docs / catalog | Implemented | Inventory docs are now populated from the master spec and reference bundle. |

## Validation

### Environment

- Created `.venv` using `virtualenv`.
- Verified editable package installation succeeded.
- `python scripts/check_env.py`
  - Passed

### Reference Bundle

- `python scripts/download_pipeline_references.py`
  - Passed
  - Downloaded 18 PDFs
- `python scripts/generate_pipeline_docs.py`
  - Passed
  - Generated or refreshed 7 inventory docs

### Tests

- `python -m pytest -q`
  - Passed: 18
  - Failed: 0
  - Warning: 1 `websockets.legacy` deprecation warning from a dependency
- `python scripts/validate_paper_broker.py --mode replay --fixture tests/fixtures/paper/recorded_paper_session_sample.json --record-output /tmp/paper_session_transcript.json`
  - Passed
- `make paper-validate`
  - Passed
- `python scripts/validate_paper_broker.py --mode paper`
  - Passed against the external Alpaca paper account once credentials were supplied

### Scripts Run

- `bash scripts/bootstrap.sh`
  - Passed
- `python scripts/check_env.py`
  - Passed
- `python scripts/smoke_test.py`
  - Passed
- `python scripts/monitor_health.py --broker mock --replay-fixture tests/fixtures/hft/captured_replay_sample.json --iterations 2 --interval 0.0 --stale-after 30.0`
  - Passed
- `python scripts/test_hft_pipeline.py`
  - Passed
- `python scripts/test_pipeline.py`
  - Passed
- `python scripts/download_pipeline_references.py`
  - Passed
- `python scripts/generate_pipeline_docs.py`
  - Passed

### Runtime Outcomes

- `scripts/test_hft_pipeline.py`
  - HFT synthetic quotes processed successfully
  - Final portfolio value: `99999.98`
  - Inventory: `0.0`
- `scripts/test_pipeline.py`
  - Fetched live Alpaca crypto bars for `BTC/USD` and `ETH/USD`
  - Built features and trained HMM + LightGBM
  - Backtest completed with total return `0.00%` because the generated signals did not create a positive allocation
- `scripts/smoke_test.py`
  - Offline synthetic OHLCV, model fit, mock broker, captured replay stream, HFT matching, health loop, and risk gate all executed
  - Backtest total return: `-3.7992%`
  - MockBroker fills: `1`
  - Stream features emitted: `2`
  - Health loop healthy: `True`
- `scripts/monitor_health.py`
  - Mock broker heartbeat healthy
  - Captured replay fixture marked fresh
  - Kill switch stayed inactive
- `scripts/validate_paper_broker.py --mode replay`
  - Exercised auth/connect, heartbeat, account sync, open order sync, duplicate blocking, cancel reconciliation, fill reconciliation, and disconnect/reconnect on the recorded session fixture
- `scripts/validate_paper_broker.py --mode paper`
  - Exercised the same checklist against the external paper account, with duplicate blocking enforced by the session recorder guard

## Safety

- Live mode is still blocked unless `SYS_MODE=live` and `ALLOW_LIVE_TRADING=true`.
- Live websocket stream remains disabled by default and requires `enable_live_stream=True`.
- Mock broker remains mock-only.
- Paper broker adapter is now externally validated against the Alpaca paper account.
- Offline paper-session replay now covers the broker checklist and can be replayed in CI/pytest/smoke without external dependencies.
- Pre-trade gate blocks stale data and excessive exposure before execution.
- Health monitor now surfaces stale stream conditions and triggers the risk kill switch after repeated failures.

## Remaining Gaps

- The real external paper-broker session was exercised, but it should still be re-run briefly after broker-related changes.
- No live broker path was exercised.
- Strategy performance remains unproven; backtest results here validate plumbing, not alpha.
- The docs now contain a real inventory, but many candidate pipelines remain research-only by design.

## Patch Summary

Files changed in this round:

- `docs/pipeline_library/00_pipeline_catalog.md`
- `docs/pipeline_library/01_bayesian_pipelines.md`
- `docs/pipeline_library/02_sde_pipelines.md`
- `docs/pipeline_library/03_rl_pipelines.md`
- `docs/pipeline_library/04_financial_dl_pipelines.md`
- `docs/pipeline_library/05_hft_pipelines.md`
- `docs/pipeline_library/05_priority_shortlist.md`
- `docs/pipeline_library/06_rejected_or_archived_pipelines.md`
- `docs/references/index.md`
- `docs/references/reference_manifest.json`
- `docs/references/papers/*`
- `Makefile`
- `scripts/download_pipeline_references.py`
- `scripts/generate_pipeline_docs.py`
- `scripts/monitor_health.py`
- `scripts/smoke_test.py`
- `scripts/validate_paper_broker.py`
- `src/brokers/__init__.py`
- `src/brokers/base.py`
- `src/brokers/mock.py`
- `src/brokers/paper.py`
- `src/brokers/paper_session.py`
- `src/ingestion/websocket_client.py`
- `src/monitoring/__init__.py`
- `src/monitoring/health.py`
- `pyproject.toml`
- `tests/fixtures/hft/captured_replay_sample.json`
- `tests/fixtures/paper/recorded_paper_session_sample.json`
- `tests/test_monitoring.py`
- `tests/test_paper_broker.py`
- `tests/test_paper_session_replay.py`

Other previously changed files still remain in the repo from the earlier implementation round, including packaging/bootstrap/runtime hardening edits.

## Next Steps

1. Wire the paper-broker adapter to a real Alpaca paper account and validate heartbeat/order reconciliation externally.
2. Add a broker-state reconciliation test against a recorded paper session if credentials/data become available.
3. Extend monitoring with alert routing if the repo needs a persistent operator loop.
4. Expand HFT replay from the captured fixture into longer event-time samples.
5. Only after the operational gates are stable, spend more time on strategy performance research.
