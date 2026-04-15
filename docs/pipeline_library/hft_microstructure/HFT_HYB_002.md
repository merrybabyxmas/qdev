# HFT_HYB_002 — Regime-Aware HFT Threshold Router

- pipeline_id: `HFT_HYB_002`
- family: `HFT Hybrid`
- status: `research_candidate`

## Objective
상위 regime에 따라 HFT 모델은 그대로 두고 threshold, spread cap, prediction minimum을 조절한다.

## Horizon
- policy cycle (per model_scheduler run)
- intra-cycle fast update 가능

## Inputs
- regime
- recent paper shadow ranking
- recent soak status
- current spread regime
- current toxicity regime

## Target
- `min_prediction_bps`
- `max_spread_bps`
- `toxicity_max`
- `aggressiveness_multiplier`

## Model
- heuristic router
- later weighted policy optimizer 가능

## Execution Mode
- threshold 조절만 수행
- HFT 모델 자체는 교체하지 않음

## Preferred Regimes
- 모든 active regime에서 동작
- 특히 `HIGH_VOL_TREND`, `EVENT_WINDOW`에서 핵심 역할

## Blocked Regimes
- 없음

## Control Plane Usage
- `routing_policy.json` → `layers.hft.thresholds` 필드를 동적으로 조절
- HFT_HYB_001과 계층 관계: HYB_001이 on/off, HYB_002가 parameter tuning

## Training Protocol
- regime별 threshold calibration
- sensitivity analysis

## Evaluation Protocol
- regime별 threshold 효과 (pnl delta by regime)
- over-tuning 방지 (stability check)

## Promotion Gate
- regime별 threshold 효과 > uniform threshold baseline
- paper session 안정성 확인

## Demotion Gate
- 과도한 threshold 변동 (instability)
- regime 오분류로 인한 wrong threshold 적용

## Notes
모델 교체보다 더 안정적인 운영 수단으로 사용할 수 있다.
`RoutingPolicyEngine._MACRO_REGIME_MAP`과 연동해 regime별 recommended pipeline도 조절 가능.
