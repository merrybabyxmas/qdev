# HFT_EXEC_002 — Cancel/Replace Threshold Controller

- pipeline_id: `HFT_EXEC_002`
- family: `HFT Execution`
- status: `research_candidate`

## Objective
기존 주문을 언제 유지하고, 언제 cancel/replace할지 결정한다.

## Horizon
- per-tick active order management

## Inputs
- current queue proxy
- spread
- updated prediction
- toxicity
- order age
- burst score

## Target
- action ∈ {keep, cancel, replace}

## Model
- rule-based baseline
- later RL overlay 가능

## Execution Mode
- active order management
- stale signal removal

## Preferred Regimes
- `NORMAL_BALANCED`
- `LOW_VOL_TREND`
- `LOW_VOL_MEAN_REVERT`

## Blocked Regimes
- `CRISIS`
- `HIGH_TOXICITY`

## Control Plane Usage
- `max_order_age_seconds` (현재 60초가 기본값)
- `duplicate_order_blocking` 활성화 여부

## Training Protocol
- replay 기반 queue/fill simulation
- cancel impact measurement

## Evaluation Protocol
- adverse selection from stale orders
- cancel/replace overhead cost
- net pnl improvement

## Promotion Gate
- stale-order adverse selection 감소 확인
- cancel overhead < improvement delta

## Demotion Gate
- 과도한 cancel (thrashing)
- net cost 초과

## Notes
실제 체결 품질과 adverse selection 방어를 위해 중요하다.
`EXEC_GLOBAL_001` (duplicate-order blocking / cancel reconcile)과 연계.
RL overlay는 paper validation 이후에만 적용.
