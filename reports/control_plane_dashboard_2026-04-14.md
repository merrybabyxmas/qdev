# Control Plane Dashboard Implementation Report

Date: 2026-04-14

## Scope

This report covers the new control-plane layer added on top of the existing paper-trading and offline evaluation stack:

- continuous data collection
- periodic model retraining
- regime-aware ranking and routing
- champion / challenger registry
- persisted status, logs, and snapshot artifacts
- interactive dashboard controls

## Implemented Components

### Control-plane modules

- `src/controlplane/artifacts.py`
- `src/controlplane/ranking.py`
- `src/controlplane/regime.py`
- `src/controlplane/router.py`
- `src/controlplane/service.py`
- `src/controlplane/snapshot.py`

### Control-plane scripts

- `scripts/refresh_control_plane.py`
- `scripts/run_data_collector.py`
- `scripts/run_model_scheduler.py`
- `scripts/manage_runtime_service.py`
- `scripts/set_champion.py`

### Dashboard

- `dashboard/app.py`
- includes auto-refresh controls and recent log / soak-record views

### Tests

- `tests/test_controlplane.py`

## Persisted Artifacts

The following artifacts are now written by the control-plane path:

- `artifacts/control_plane/dashboard_snapshot.json`
- `artifacts/control_plane/leaderboard.csv`
- `artifacts/control_plane/leaderboard.json`
- `artifacts/control_plane/champion_registry.json`
- `artifacts/control_plane/regime_snapshot.json`
- `artifacts/control_plane/data_collector_status.json`
- `artifacts/control_plane/model_scheduler_status.json`
- `artifacts/control_plane/logs/data_collector.jsonl`
- `artifacts/control_plane/logs/model_scheduler.jsonl`
- `artifacts/control_plane/logs/collector.log`
- `artifacts/control_plane/logs/model_scheduler.log`
- `artifacts/control_plane/services/collector.json`
- `artifacts/control_plane/services/model_scheduler.json`

The soak path remains append-only at:

- `artifacts/paper_soak/soak_records.jsonl`

## Commands Run

### Snapshot refresh

```bash
./.venv/bin/python scripts/refresh_control_plane.py
```

Outcome:

- Passed
- Rebuilt the dashboard snapshot and leaderboard artifacts

### One-shot collector cycle

```bash
./.venv/bin/python scripts/run_data_collector.py --iterations 1 --refresh-snapshot
```

Outcome:

- Passed
- Fetched 181 rows each for `BTC/USD`, `ETH/USD`, `SOL/USD`, `AVAX/USD`, `DOGE/USD`
- Wrote raw and feature-ready CSV artifacts
- Refreshed dataset version `1c1b9cc0f7a9`
- Wrote `artifacts/control_plane/data_collector_status.json`

### One-shot model scheduler cycle

```bash
./.venv/bin/python scripts/run_model_scheduler.py --iterations 1 --suite shortlist --refresh-dataset
```

Outcome:

- Passed
- Produced a new run under `artifacts/experiments/runs/20260414_112744`
- Wrote report `reports/model_cycle_20260414_112738.md`
- Refreshed leaderboard and routing snapshot

Observed shortlist output:

- `F001` reference
- `F015`, `F019`, `BASE_EQ`, `S011`, `F009`, `F027`, `S004`, `F002` promote

### Service manager start / stop validation

```bash
./.venv/bin/python scripts/manage_runtime_service.py collector start
./.venv/bin/python scripts/manage_runtime_service.py collector status
./.venv/bin/python scripts/manage_runtime_service.py collector stop

./.venv/bin/python scripts/manage_runtime_service.py model_scheduler start
./.venv/bin/python scripts/manage_runtime_service.py model_scheduler status
./.venv/bin/python scripts/manage_runtime_service.py model_scheduler stop
```

Outcome:

- Passed
- `collector` background process started and produced live logs
- `model_scheduler` background process started successfully
- both services stopped cleanly

### Champion override validation

```bash
./.venv/bin/python scripts/set_champion.py --pipeline-id F015
./.venv/bin/python scripts/set_champion.py --clear
```

Outcome:

- Passed
- manual champion override updated `champion_registry.json`
- override cleared successfully

### Dashboard import and runtime validation

```bash
./.venv/bin/python -c "import streamlit, plotly; print('dashboard-import-ok')"
./.venv/bin/streamlit run dashboard/app.py --server.headless true --server.port 8511
```

Outcome:

- Passed
- imports succeeded
- Streamlit server started successfully and exposed `http://localhost:8511`
- process was then stopped after verification

### Test suite

```bash
./.venv/bin/python -m pytest -q
```

Outcome:

- Passed: `26 passed, 1 warning`
- Remaining warning is the existing third-party `websockets.legacy` deprecation warning

## Fixes Applied During Validation

### Regime warning fix

Issue:

- `src/controlplane/regime.py` produced a `RuntimeWarning: Mean of empty slice`

Fix:

- added `_safe_abs_mean()` to handle all-NaN / empty correlation slices safely

Result:

- warning removed from the control-plane refresh and cycle scripts

## Current Status

The repository now has a working control-plane loop for:

- background data collection
- periodic training and leaderboard refresh
- regime-aware routing
- champion / challenger registry
- persisted snapshots and logs
- interactive dashboard controls
- dashboard-side log tails and auto-refresh driven monitoring

This is suitable for:

- continuous research-state monitoring
- paper-stage operational visibility
- shadow competition promotion workflows

## Remaining Gaps

- the dashboard has been validated for startup and control hooks, but not yet load-tested under long multi-user access
- `paper_soak` service can be managed through the same service manager, but long-duration production soak should still be run using the dedicated soak runbook and acceptance gate
- live trading remains gated behind preflight-only and canary rules

## Recommended Next Steps

1. Keep `collector` and `model_scheduler` running as background services during the research cycle.
2. Use the dashboard as the top-level operator view while continuing append-only paper soak logging.
3. Add one or two more real paper-session fixtures from distinct market conditions.
4. Promote only the top paper-shadow winners into the live preflight path.
