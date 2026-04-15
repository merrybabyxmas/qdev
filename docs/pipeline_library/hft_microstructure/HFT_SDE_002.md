# HFT_SDE_002 — Jump / Burst Risk Overlay

- pipeline_id: `HFT_SDE_002`
- family: `HFT SDE`
- status: `research_candidate`

## Objective
점프성 움직임과 burst volatility 상태를 탐지해 HFT 엔진의 aggressiveness를 축소한다.

## Horizon
- real-time detection, effect persists for short window

## Inputs
- volatility_burst
- jump proxy
- spread widening speed
- signed flow burst
- short realized vol

## Target
- risk_state
- aggressiveness multiplier

## Model
- jump-diffusion inspired overlay
- burst-risk scoring

## Execution Mode
- reduce size
- widen thresholds
- temporary halt

## Preferred Regimes
- `HIGH_VOL_TREND`
- `EVENT_WINDOW`

## Blocked Regimes
- 없음
- 오히려 다른 HFT 모델 보호 목적

## Control Plane Usage
- aggressiveness multiplier를 policy에 반영
- HFT_HYB_002와 연계해 threshold 동적 조절

## Training Protocol
- burst threshold calibration on replay
- jump detection sensitivity tuning

## Evaluation Protocol
- burst detection precision/recall
- protection value (pnl delta in burst windows)
- aggressiveness recovery speed

## Notes
HFT_RISK_001보다 더 연속적이고 부드러운 risk overlay.
HFT_RISK_001은 hard gate, HFT_SDE_002는 soft damper 역할.
