# HFT_DL_002 — Event Sequence LSTM

- pipeline_id: `HFT_DL_002`
- family: `HFT Deep Learning`
- status: `research_candidate`

## Objective
tick / quote / trade 이벤트 시퀀스를 이용해 next short-horizon move를 추정한다.

## Horizon
- `next 5 ticks`

## Inputs
- event timestamps
- trade sign
- size
- quote changes
- spread changes
- imbalance changes

## Target
- `future_return_bps_5tick`
- or `next_tick_direction`

## Model
- lightweight LSTM / GRU

## Execution Mode
- direction-aware thresholding
- confidence-based no-trade

## Preferred Regimes
- `NORMAL_BALANCED`
- `LOW_VOL_TREND`

## Blocked Regimes
- `EVENT_WINDOW`
- `HIGH_TOXICITY`

## Control Plane Usage
- prediction threshold를 policy에서 읽음
- HFT_RISK_001 앞단 필수

## Training Protocol
- event-time replay dataset (not clock-time)
- strict temporal split
- sequence length tuning
- latency measurement 포함

## Evaluation Protocol
- directional accuracy on event sequences
- post-cost pnl with latency penalty
- out-of-distribution regime robustness
- inference latency (must fit HFT budget)

## Promotion Gate
- HFT_BASE_001 대비 분명한 precision 개선
- latency budget 충족
- paper replay stability 확보

## Demotion Gate
- latency 초과
- regime overfit 확인
- 비용 반영 후 edge 소멸

## Notes
DeepLOB(HFT_DL_001)보다 event-time 해석에 가깝고, HFT_BASE_001의 발전형 후보로 본다.
순수 top-of-book이 아닌 event flow 구조를 학습한다는 점에서 보완 관계.
