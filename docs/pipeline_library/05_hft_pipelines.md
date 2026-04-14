# HFT Pipelines Catalog

## HFT-specialized Pipelines (Group E)

- 목적: HFT 및 초단기 미시구조 기반 파이프라인 정리
- 핵심 질문: 마이크로초~밀리초 단위의 틱/호가창 데이터에서 어떻게 의미 있는 알파를 추출하고 슬리피지를 통제할 것인가?
- 모델 패밀리: LOB imbalance + logistic, DeepLOB, queue-aware market making, Hawkes intensity 등
- 리스크 및 실패 모드: 통신 지연(Latency), Queue Position 추정 오차, Adverse Selection (독성 주문)
- 이 문서는 자동 생성된 A-D family 카탈로그를 보완하는 보조 HFT 카탈로그다.

### E001. LOB Imbalance + Logistic Baseline
- 입력: H2 (Top-of-Book) + H4 (Order Flow)
- 타깃: next midprice move
- 실행: thresholded taking
- 장점: 연산 속도가 극히 빨라 HFT baseline으로 최적
- 난이도: L2

### E002. Microprice Deviation + Thresholded Execution
- 입력: H2 (Microprice vs Midprice)
- 타깃: 단기 가격 회귀 (Spread capture)
- 실행: Passive quoting with inventory cap
- 장점: 호가창 불균형을 즉각적으로 스프레드 수익으로 전환
- 난이도: L3

### E003. Short-term Volatility Burst Detector
- 입력: H1 (Tick Trades)
- 타깃: Volatility burst event
- 실행: Liquidity taking 중단 / Spread 확대
- 장점: 급등락장(Flash crash 등)에서 Market Making 손실 방어
- 난이도: L3

## Reference Anchors

- [DeepLOB: Deep Convolutional Neural Networks for Limit Order Books](../references/papers/1808.03668_deeplob.pdf)
- [Multi-Horizon Forecasting for Limit Order Books](../references/papers/2105.10430_lob_multi_horizon.pdf)
- [Market Making with Deep Reinforcement Learning from Limit Order Books](../references/papers/2305.15821_market_making_rl_lob.pdf)
- [Deep Learning can Replicate Adaptive Traders in a Limit-Order-Book Financial Market](../references/papers/1811.02880_adaptive_traders_lob.pdf)

## Repo Coverage

- `src/ingestion/websocket_client.py` now supports offline replay and immediate feature emission on trade/quote snapshots.
- `src/backtest/matching_engine.py` provides the queue-aware synthetic execution loop used in smoke tests.
- `src/monitoring/health.py` and `scripts/monitor_health.py` validate stream freshness, broker heartbeat, and kill-switch wiring.
- `tests/test_monitoring.py` exercises a captured replay fixture rather than only synthetic event injection.

## Navigation

- Start from [Pipeline Library](./README.md) for the full map.
- Use [Priority Shortlist](./05_priority_shortlist.md) for HFT-adjacent implementation ordering and gating.
- Use [Candidate Dossier Index](./07_candidate_dossiers.md) when you want the detailed dossier view of the primary shortlist.
