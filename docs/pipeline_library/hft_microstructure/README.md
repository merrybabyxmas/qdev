# HFT Microstructure Pipeline Library

## 목적
이 문서는 `qdev` 저장소에서 사용하는 HFT / microstructure 계열 파이프라인 후보들을 정의한다.

이 계열은 기존 macro/daily 파이프라인과 다르게 다음 특징을 가진다.

- 입력 데이터는 tick / quote / top-of-book / order flow 중심
- horizon은 next-tick ~ next-5-tick 수준
- 목표는 예측 정확도만이 아니라 execution 품질, fill, toxicity 회피, inventory 통제까지 포함
- 상위 control-plane의 정책(`hft_policy.json`)을 읽어 동적으로 활성/비활성된다
- paper replay / captured session / live paper validation을 통해 승격된다

## 분류
현재 HFT 파이프라인은 다음 다섯 계열로 나눈다.

1. BASE
   - 가장 단순한 마이크로스트럭처 베이스라인
   - 빠른 추론, 높은 설명가능성
2. RISK
   - 독성, 점프, 비유동성, wide spread 회피
3. SDE
   - inventory / quote skew / volatility state 제어
4. DL
   - compact microstructure predictor
5. EXEC / HYB
   - execution 최적화 또는 macro/daily와 연결되는 hybrid control

## 공통 필드
각 파이프라인 md는 아래 필드를 가진다.

- pipeline_id
- family
- status
- objective
- horizon
- inputs
- target
- model
- execution_mode
- preferred_regimes
- blocked_regimes
- control_plane_usage
- training_protocol
- evaluation_protocol
- promotion_gate
- demotion_gate
- artifacts
- notes

## 운영 원칙
- HFT 파이프라인은 단독 챔피언이 아니라 `hft champion` 후보군으로 관리한다.
- macro/daily champion과 직접 같은 leaderboard에서 비교하지 않는다.
- HFT 파이프라인은 반드시 replay 기반 검증을 거쳐야 한다.
- real paper account 검증 전에는 live 승격 금지.
- `WIDE_SPREAD_ILLIQUID`, `HIGH_TOXICITY`, `CRISIS`에서는 대부분 halt 또는 강한 threshold를 적용한다.
