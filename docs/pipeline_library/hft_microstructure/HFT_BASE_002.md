# HFT_BASE_002 — OBI + Logistic Direction Classifier

- pipeline_id: `HFT_BASE_002`
- family: `HFT Microstructure`
- status: `research_candidate`

## Objective
호가 불균형과 microprice 기반으로 next-tick direction을 분류한다.

## Horizon
- `next 1 tick`

## Inputs
- obi
- microprice_drift
- spread
- bid_size
- ask_size
- quote_update_intensity

## Target
- `next_tick_direction` ∈ {up, flat, down}

## Model
- logistic regression
- optional online calibration

## Execution Mode
- only trade when classifier confidence > threshold
- flat prediction은 no-trade

## Preferred Regimes
- `NORMAL_BALANCED`
- `TIGHT_SPREAD_DEEP_BOOK`

## Blocked Regimes
- `HIGH_TOXICITY`
- `EVENT_WINDOW`
- `WIDE_SPREAD_ILLIQUID`

## Control Plane Usage
상위 policy가 confidence threshold와 symbol enable 여부를 결정한다.

## Training Protocol
- rolling replay dataset
- class imbalance 보정
- confidence calibration 필수

## Evaluation Protocol
- directional accuracy
- precision on positive trades
- confusion matrix by regime
- cost-adjusted pnl

## Promotion Gate
- HFT_BASE_001 대비 trade precision 우위
- no-trade filtering 품질 우수

## Demotion Gate
- flat 구간 과거래
- spread 비용 반영 후 성능 소멸
