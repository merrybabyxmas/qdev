# Paper Soak Recording Update

- date: `2026-04-14`
- scope: append-only soak record storage and dedicated soak runner

## What Changed

- Added an append-only JSONL store for soak records at `artifacts/paper_soak/soak_records.jsonl`.
- Added `SoakRunner` to persist `run_start`, `iteration`, and `run_end` records for every soak execution.
- Added a dedicated CLI: `scripts/run_paper_soak.py`.
- Added `make paper-soak` for repeatable soak execution.
- Updated the paper soak runbook to reference the persistent record store.

## Runtime Behavior

- Each soak run gets a unique `run_id`.
- Every iteration is appended as a separate JSONL record.
- The store is append-only and does not overwrite prior soak runs.
- The record store is independent from paper validation transcripts and test fixtures.

## Verification

Validated with:

```bash
./.venv/bin/python -m pytest -q tests/test_soak_records.py
./.venv/bin/python scripts/run_paper_soak.py --broker mock --iterations 2 --interval 0 --record-root /tmp/qdev-soak-records --session-name test-soak
./.venv/bin/python scripts/run_paper_soak.py --broker mock --iterations 1 --interval 0 --record-root /tmp/qdev-soak-records --session-name test-soak-2
```

Observed:

- `2 passed` in `tests/test_soak_records.py`
- soak records appended across repeated runs in the same `record-root`
- final JSONL file retained prior records instead of overwriting them

## Default Artifact Path

- `artifacts/paper_soak/soak_records.jsonl`

## Live Soak Verification

The append-only store was exercised against the real paper account with live stream validation, and the long soak was stopped after the health fix was verified.

Command in use:

```bash
./.venv/bin/python scripts/run_paper_soak.py \
  --broker paper \
  --stream-mode live \
  --symbol BTC/USD \
  --iterations 240 \
  --interval 60 \
  --stream-warmup-seconds 10 \
  --stale-after 120 \
  --record-root artifacts/paper_soak \
  --session-name paper-soak-live-20260414
```

Observed:

- live stream connects successfully
- broker heartbeat stays healthy
- iteration records append to `artifacts/paper_soak/soak_records.jsonl`
- the store remains append-only across repeated runs
- the stream freshness gate is holding at `stale_after=120`
- the latest short soak completed healthy with `healthy_count=3` and `unhealthy_count=0`

## Health Fix

The earlier stale-stream false positive came from live websocket handlers not updating `last_event_received_at`.
That was corrected in `src/ingestion/websocket_client.py`, and a fresh short live soak now completes healthy.

Verified command:

```bash
./.venv/bin/python scripts/run_paper_soak.py \
  --broker paper \
  --stream-mode live \
  --symbol BTC/USD \
  --iterations 3 \
  --interval 10 \
  --stream-warmup-seconds 10 \
  --stale-after 120 \
  --record-root artifacts/paper_soak \
  --session-name paper-soak-live-healthy-check
```

Observed result:

- `iterations_recorded: 3`
- `healthy_count: 3`
- `unhealthy_count: 0`
- `kill_switch_active: false`
- `stream.status: fresh`
