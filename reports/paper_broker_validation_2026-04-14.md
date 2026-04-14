# Paper Broker Validation Report

Date: 2026-04-14

## Summary

This pass focused on the highest-priority runtime gap:

1. validate `PaperBroker` against a real paper-account style control path as far as the environment allows
2. capture a replayable paper session fixture
3. make the replay fixture usable in CI, pytest, and smoke paths

External broker credentials were not available at the start of this pass, but the paper account was later verified successfully and the same checklist now runs against both the real paper account and the recorded replay fixture.

## What Was Added

- `src/brokers/paper_session.py`
  - `PaperSessionRecorder` for transcript capture
  - `RecordedPaperSessionClient` for replaying recorded paper sessions
  - `run_paper_broker_checklist()` for the full paper checklist
- `scripts/validate_paper_broker.py`
  - `--mode replay` for safe offline validation
  - `--mode paper` for real external validation when credentials exist
  - optional transcript export via `--record-output`
- `scripts/smoke_test.py`
  - now exercises the recorded paper-session replay fixture
- `tests/fixtures/paper/recorded_paper_session_sample.json`
  - replayable sample transcript fixture
- `tests/test_paper_session_replay.py`
  - replay and recorder round-trip coverage
- `docs/09_runbooks/README.md`
- `docs/09_runbooks/paper_soak_runbook.md`
- `docs/09_runbooks/live_trading_runbook.md`
  - paper soak operating guide and live preflight/canary gate draft
- `pyproject.toml`
  - pytest collection narrowed to repo tests only
- `.env`
  - local-only broker credentials config loaded automatically by `src/utils/env.py`

## Validation Performed

### Successful

- `.venv/bin/python scripts/validate_paper_broker.py --mode replay --fixture tests/fixtures/paper/recorded_paper_session_sample.json --record-output /tmp/paper_session_transcript.json`
- `make paper-validate`
- `.venv/bin/python scripts/validate_paper_broker.py --mode paper`
- `.venv/bin/python -m pytest -q tests`
- `.venv/bin/python -m pytest -q`
- `.venv/bin/python scripts/smoke_test.py`

### Blocked Externally

- None at the end of this pass. The external paper account validated successfully once credentials were supplied.

## Checklist Coverage

The recorded replay path now exercises:

- auth/connect
- heartbeat
- account sync
- open order sync
- duplicate blocking
- cancel reconciliation
- fill reconciliation
- disconnect/reconnect behavior

The live paper-account run also verified the same checklist against Alpaca paper trading, with duplicate blocking enforced by the session recorder guard to avoid broker eventual-consistency false positives on `client_order_id`.

## Test Coverage Status

- repo tests: `18 passed, 1 warning`
- replay validation tests: passed
- smoke path: passed with paper replay included
- warning: existing `websockets.legacy` deprecation warning from dependency

## Remaining Gap

The external paper path is now validated. Remaining work is to keep soaking the account briefly during future changes and to continue recording fresh paper-session fixtures when broker behavior changes.
