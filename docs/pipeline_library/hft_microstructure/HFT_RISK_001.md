# HFT_RISK_001 — Toxicity / Wide-Spread Halt Filter

- pipeline_id: `HFT_RISK_001`
- family: `HFT Risk Overlay`
- status: `mandatory_overlay`

## Objective
독성 주문흐름, 넓은 스프레드, 얕은 호가 구간에서 HFT 엔진을 정지 또는 강하게 제한한다.

## Horizon
- real-time (every tick)

## Inputs
- spread_bps
- toxicity_vpin_proxy
- cancel_to_trade_ratio
- volatility_burst
- depth_proxy

## Target
- `allow_trade` boolean
- `halt_reason`

## Model
- rule-based
- optional Bayesian smoothing

## Execution Mode
- hard gate
- halt / reduce size / raise threshold

## Preferred Regimes
- all regimes에서 사용

## Blocked Regimes
- 없음
- 오히려 다른 파이프라인을 차단하는 역할

## Control Plane Usage
- `allow_hft`
- `max_spread_bps`
- `toxicity_max`
- `max_burst_score`

## Training Protocol
학습 모델이라기보다 calibration 대상.

## Evaluation Protocol
- false halt rate
- missed toxic-event rate
- protection value during paper replay

## Promotion Gate
이건 승격 대상이 아니라 기본 보호계층이다.

## Demotion Gate
해당 없음. 항상 활성.

## Artifacts
- `artifacts/control_plane/hft_status.json` → `halt_reason` 필드

## Notes
모든 HFT pipeline 앞단에 위치해야 한다.
