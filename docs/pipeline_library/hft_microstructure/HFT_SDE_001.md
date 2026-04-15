# HFT_SDE_001 — Inventory-Aware Quote Skew Controller

- pipeline_id: `HFT_SDE_001`
- family: `HFT SDE`
- status: `research_candidate`

## Objective
재고(inventory)와 단기 변동성 상태를 반영하여 quoting 방향과 aggressiveness를 조절한다.

## Inputs
- inventory
- realized short-term volatility
- spread
- microprice drift
- order imbalance

## Target
- quote skew
- aggressiveness adjustment

## Model
- Avellaneda-Stoikov inspired inventory control
- simplified SDE / control approximation

## Execution Mode
- market making or passive quoting bias
- inventory-dependent skew

## Preferred Regimes
- `NORMAL_BALANCED`
- `LOW_VOL_MEAN_REVERT`

## Blocked Regimes
- `HIGH_TOXICITY`
- `CRISIS`
- `WIDE_SPREAD_ILLIQUID`

## Control Plane Usage
- max inventory
- volatility widener
- quote skew strength

## Training Protocol
- parameter calibration
- replay-based control tuning

## Evaluation Protocol
- inventory stability
- adverse selection reduction
- quote fill quality
- pnl stability

## Notes
직접적인 alpha 모델이라기보다 execution control에 가깝다.
