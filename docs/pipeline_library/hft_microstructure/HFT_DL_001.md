# HFT_DL_001 — Compact DeepLOB Classifier

- pipeline_id: `HFT_DL_001`
- family: `HFT Deep Learning`
- status: `research_candidate`

## Objective
짧은 길이의 order book / top-of-book 시퀀스를 입력으로 받아 next price move를 분류한다.

## Inputs
- top-of-book snapshots
- bid/ask sizes
- spread
- microprice
- imbalance history
- short event window

## Target
- `next_k_tick_direction`

## Model
- compact DeepLOB-style CNN/LSTM
- latency budget 내 소형 구조만 허용

## Execution Mode
- thresholded classifier
- HFT_RISK_001 필수 앞단
- no-trade region 포함

## Preferred Regimes
- `NORMAL_BALANCED`
- `TIGHT_SPREAD_DEEP_BOOK`

## Blocked Regimes
- `HIGH_TOXICITY`
- `CRISIS`

## Training Protocol
- replay dataset
- strict train/val/test split by event time
- latency measurement 포함

## Evaluation Protocol
- directional accuracy
- calibration
- post-cost pnl
- inference latency
- regime-specific robustness

## Promotion Gate
- HFT_BASE 계열 대비 분명한 precision 개선
- latency budget 충족
- paper replay에서 stability 확보

## Demotion Gate
- latency 초과
- regime overfit
- 비용 반영 후 edge 소멸
