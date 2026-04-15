# HFT_BASE_001 — OBI + Microprice OnlineSGD

- pipeline_id: `HFT_BASE_001`
- family: `HFT Microstructure`
- status: `active_baseline_candidate`

## Objective
Top-of-book 기반의 매우 짧은 horizon 방향성과 강도를 추정하여 next-5-tick 수익률(bps)을 예측한다.

## Horizon
- inference horizon: `next 5 ticks`
- refresh frequency: `every tick`

## Inputs
- obi
- microprice_drift
- spread
- toxicity_vpin_proxy
- volatility_burst

## Target
- `future_return_bps_5tick`

## Model
- `OnlineSGDRegressor`

## Execution Mode
- liquidity-taking 또는 no-trade
- prediction threshold 기반
- spread / toxicity / volatility gate 포함

## Preferred Regimes
- `NORMAL_BALANCED`
- `LOW_VOL_TREND`
- `LOW_VOL_MEAN_REVERT`

## Blocked Regimes
- `WIDE_SPREAD_ILLIQUID`
- `HIGH_TOXICITY`
- `CRISIS`

## Control Plane Usage
- `allow_hft=true`일 때만 동작
- `min_prediction_bps`
- `max_spread_bps`
- `toxicity_max`
를 control-plane에서 읽음

## Training Protocol
- online update
- recent replay window 기반 periodic reset 가능
- warmup ticks 필요

## Evaluation Protocol
- replay hit rate
- post-cost pnl
- threshold별 precision
- spread-conditioned accuracy
- stale-signal rate

## Promotion Gate
- replay 성능이 random / zero baseline 초과
- paper session에서 false trigger 낮음
- toxicity 구간 halt 정상

## Demotion Gate
- prediction drift 심화
- spread widening 구간 오진 증가
- replay/paper 성능 붕괴

## Artifacts
- `artifacts/control_plane/hft_status.json`
- `artifacts/control_plane/logs/hft_ticks.jsonl`

## Notes
현재 `qdev`에서 가장 먼저 살아 있는 HFT 베이스라인으로 간주한다.
