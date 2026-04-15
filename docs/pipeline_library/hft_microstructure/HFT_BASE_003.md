# HFT_BASE_003 — Microprice Drift Threshold Trader

- pipeline_id: `HFT_BASE_003`
- family: `HFT Microstructure`
- status: `research_candidate`

## Objective
모델을 복잡하게 쓰지 않고 microprice deviation 자체를 threshold rule로 거래한다.

## Horizon
- rule trigger 기반 (time-agnostic)

## Inputs
- microprice
- midprice
- spread
- top-of-book size imbalance

## Target
- 직접 예측보다 rule-trigger

## Model
- explicit threshold rule
- no ML

## Execution Mode
- trade only when microprice drift exceeds threshold
- spread and toxicity filters mandatory

## Preferred Regimes
- `LOW_VOL_TREND`
- `NORMAL_BALANCED`

## Blocked Regimes
- `WIDE_SPREAD_ILLIQUID`
- `HIGH_TOXICITY`

## Control Plane Usage
control-plane이 threshold 상한/하한을 조절한다.

## Training Protocol
- threshold search only
- no heavy fitting

## Evaluation Protocol
- simplicity baseline
- latency advantage
- threshold sensitivity

## Promotion Gate
- 다른 ML 모델 대비 비교 기준선 역할
- spread-adjusted 순수 rule edge 확인

## Demotion Gate
- edge 완전 소멸 (random과 동등)
- 더 단순한 zero-trade 대비 성능 미달

## Notes
가장 빠른 fallback HFT baseline으로 유지할 가치가 높다.
