# HFT_EXEC_001 — Fill Probability Gate

- pipeline_id: `HFT_EXEC_001`
- family: `HFT Execution`
- status: `research_candidate`

## Objective
주문이 실제 체결될 가능성을 추정해, 의미 없는 quote spam이나 low-fill order를 줄인다.

## Horizon
- per-order decision (pre-submission)

## Inputs
- spread
- queue size proxy
- top-of-book size
- recent fill ratio
- cancel intensity
- order age

## Target
- `fill_probability`

## Model
- logistic / lightgbm small model

## Execution Mode
- submit only if fill probability > threshold
- otherwise hold / cancel / replace

## Preferred Regimes
- `NORMAL_BALANCED`
- `LOW_VOL_MEAN_REVERT`

## Blocked Regimes
- `CRISIS`

## Control Plane Usage
- fill probability threshold를 policy에서 읽음
- `min_fill_prob` (현재 0.30이 기본값)

## Training Protocol
- replay 기반 fill/no-fill labeling
- queue proxy calibration
- rolling refit

## Evaluation Protocol
- fill rate improvement vs baseline
- false positive rate (blocked good fills)
- post-fill pnl quality

## Promotion Gate
- fill rate improvement ≥ 5pp over no-gate baseline
- false positive (blocked good fills) < 10%

## Demotion Gate
- fill prediction accuracy degradation
- queue proxy stale

## Notes
이 파이프라인은 alpha가 아니라 execution quality 개선용이다.
`EXEC_GLOBAL_002`의 소프트웨어 구현체와 연결된다.
