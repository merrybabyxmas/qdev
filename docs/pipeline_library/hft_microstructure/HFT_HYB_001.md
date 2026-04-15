# HFT_HYB_001 — Macro Regime → HFT Enable/Disable Gate

- pipeline_id: `HFT_HYB_001`
- family: `HFT Hybrid`
- status: `mandatory_bridge`

## Objective
macro/daily champion 또는 regime detector의 출력을 받아 HFT 엔진을 켤지 끌지 결정한다.

## Horizon
- policy cycle (per model_scheduler run)

## Inputs
- macro regime
- daily volatility posture
- event risk score
- symbol allowlist
- capital budget

## Target
- `allow_hft`
- `enabled_symbols`
- `max_position_usd`
- `global_thresholds`

## Model
- rule-based bridge
- control-plane artifact writer

## Execution Mode
- HFT 엔진이 직접 읽는 policy 파일 생성

## Preferred Regimes
- all regimes (항상 동작, regime에 따라 출력이 달라짐)

## Blocked Regimes
- 없음 (모든 regime에서 동작하되 출력으로 차단 전달)

## Control Plane Usage
현재 `qdev`의 `routing_policy.json` → `layers.hft.allow_hft` 발행 구조와 직접 연결된다.
`RoutingPolicyEngine`이 이 역할을 수행한다.

## Training Protocol
- 학습 없음
- threshold calibration만 존재

## Evaluation Protocol
- halt precision (올바른 상황에서 halt)
- false halt rate (불필요한 halt)
- upstream regime signal lag

## Notes
현재 `qdev`의 `hft_policy.json` / `routing_policy.json` 발행 구조와 직접 연결된다.
`RoutingPolicyEngine._HFT_HALT_REGIMES`가 이 로직의 현재 구현이다.
