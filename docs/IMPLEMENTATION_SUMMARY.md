# HFT & Quant Trading System Implementation Summary

이 문서는 실거래형 퀀트 트레이딩 명세서 및 HFT(High-Frequency Trading) 확장 명세서를 바탕으로 현재까지 구현된 시스템의 전체 아키텍처와 주요 컴포넌트를 요약합니다.

모든 코드는 단순한 Mock이나 Proxy 스터브(Stub)가 아닌, 실제 동작하는 비즈니스 로직과 연산(Numpy/Pandas/LightGBM/SGD/VectorBT/Alpaca)으로 작성되었습니다.

---

## 1. 아키텍처 개요 (Architecture Overview)

시스템은 대량의 배치(Batch) 백테스트 환경과 마이크로초 단위의 이벤트 기반(Event-driven) 실시간 스트리밍 환경을 모두 지원하도록 모듈화되어 있습니다.

### 디렉터리 구조
- `src/utils/`: Pydantic 기반 설정(`config.py`) 및 Structlog 기반 구조화 로깅(`logger.py`).
- `src/brokers/`: `BrokerInterface` 추상화 및 모의 거래를 위한 `MockBroker`.
- `src/ingestion/`: Alpaca-py 기반 과거 데이터 수집(`loader.py`) 및 다중 종목 실시간 Websocket 스트림 수신(`websocket_client.py`).
- `src/state/`: HFT 환경에서 O(1) 속도로 틱/호가 데이터를 관리하기 위한 고성능 Numpy 기반 링 버퍼(`ring_buffers.py`).
- `src/features/`: 기술적 지표(`builder.py`) 및 실시간 미시구조(Microstructure) 피처(`imbalance.py`).
- `src/models/`: 배치 학습용 HMM/LightGBM 모델과 스트리밍 환경을 위한 온라인 점진적 학습 모델(`sgd_online.py`, `lgbm_online.py`), 다중 종목 랭킹 엔진(`ranker_engine.py`), 5축 시장 상태 감지기(`state_detector.py`).
- `src/signals/`: 감지된 시장 상태에 따라 실행 전략을 결정하는 동적 라우터(`router.py`).
- `src/strategies/`: ML 예측치 기반 포트폴리오 비중 산출 로직(`ml_strategy.py`, `baselines.py`).
- `src/risk/`: 포지션 캡(Position Cap) 및 낙폭(Drawdown) 기반 킬 스위치를 담당하는 리스크 관리자(`manager.py`).
- `src/execution/`: 시장가 이탈 시 기존 주문을 취소하고 재주문하는 실시간 추종 로직(`policy.py`).
- `src/backtest/`: VectorBT 기반 다중 자산 포트폴리오 백테스터(`engine.py`) 및 큐(Queue) 대기/지연(Latency)을 모사하는 HFT 매칭 엔진(`matching_engine.py`).
- `src/live/`: 실시간 스트림, 랭커, 라우터, 실행기를 통합하여 무한 루프로 동작하는 라이브 트레이딩 엔진(`engine.py`).

---

## 2. 핵심 구현 상세 (Key Implementations)

### 2.1 실시간 데이터 및 상태 관리 (Ingestion & State)
- **`MultiSymbolHFTStreamManager`**: `alpaca-py`의 `CryptoDataStream`을 사용하여 여러 종목의 Trade(체결) 및 Quote(호가) 이벤트를 비동기(`asyncio`)로 수신합니다.
- **`TickRingBuffer` / `QuoteRingBuffer`**: Pandas DataFrame의 `append` 오버헤드를 제거하기 위해 Numpy 배열을 링 버퍼 형태로 사용하여 최근 N개의 틱 데이터를 메모리에 고속으로 유지합니다.

### 2.2 5축 미시구조 피처 (5-Axis Microstructure Features)
틱 데이터가 들어올 때마다 다음 5가지 핵심 지표를 즉각 계산합니다.
1. **변동성 (Volatility):** 최근 체결가의 단기 표준편차 (`volatility_burst`).
2. **유동성 (Liquidity):** Bid-Ask Spread (`spread`).
3. **독성 (Toxicity):** 매수/매도 주도 거래량 불균형을 통한 VPIN 프록시 (`toxicity_vpin`).
4. **가격동학 (Price Dynamics):** 잔량 가중 평균가(Volume-weighted mid)인 Microprice와 Midprice 간의 괴리/추세 (`microprice_drift`).
5. **호가 불균형 (Order Book Imbalance):** `(Bid Size - Ask Size) / Total Size` (`obi`).

### 2.3 시장 상태 감지 및 라우팅 (State Detection & Routing)
- **`MarketStateDetector`**: 위 5축 피처를 입력받아 하드 룰 기반 버키팅을 거쳐 시장을 8가지 실전 운영 상태(예: `STABLE_MEAN_REVERSION`, `HIGH_VOL_TOXIC`, `WIDE_SPREAD_ILLIQUID`, `EVENT_SHOCK` 등)로 분류합니다.
- **`PipelineRouter`**: 감지된 상태에 따라 전략을 동적으로 전환합니다.
  - 독성 흐름 / 이벤트 쇼크: 거래 중단 (`HALT`).
  - 안정적 평균회귀: 스프레드 수취 패시브 마켓메이킹 (`PASSIVE_MAKE`).
  - 강한 추세: 상대 호가를 타격하는 공격적 유동성 취득 (`AGGRESSIVE_TAKE`).

### 2.4 온라인 학습 및 랭킹 (Online Learning & Ranking)
- **`OnlineSGDRegressor`**: 틱 데이터가 들어올 때마다 O(1) 시간 복잡도로 `partial_fit`을 수행하여 실시간으로 모델의 가중치를 업데이트합니다.
- **`RealTimeCrossSectionalRanker`**: 여러 종목의 피처를 동시에 평가하여 온라인 모델을 통해 단기 예측 수익률을 산출하고, 이를 기반으로 실시간 크로스섹셔널(Cross-Sectional) 포트폴리오 비중을 생성합니다.

### 2.5 HFT 실행 및 리스크 관리 (Execution & Risk)
- **`ExecutionTracker` (Cancel/Replace)**: 현재 활성화된 지정가 주문(Limit Orders)을 추적합니다. 시장의 Best Bid/Offer가 지정된 허용치(Drift BPS) 이상으로 도망가면, 즉시 기존 주문을 취소(`Cancel`)하고 새로운 호가에 맞춰 재주문(`Replace`)합니다.
- **`RiskManager`**: 개별 종목의 최대 노출 비중(Max Position Cap)을 제한하고, 지정된 손실률(Max Drawdown) 도달 시 모든 비중을 0으로 강제하는 킬 스위치(Kill Switch)를 동작시킵니다.
- **`HFTMatchingEngine`**: 네트워크 지연(Latency)을 모사하여 주문이 일정 시간 이후에 거래소 큐에 도달하도록 처리하며, 이후 시장 호가의 변동에 따라 수동적(Passive) 체결 여부를 결정하는 틱 단위 백테스트 엔진입니다.

---

## 3. 검증 및 테스트 (Verification & Testing)

모든 핵심 로직은 `unittest` 프레임워크를 통해 철저히 검증되었습니다. (`python3 -m unittest discover tests/`)
- **`test_multi_stream.py`**: 다중 종목 링 버퍼 갱신 및 실시간 랭커(`RealTimeCrossSectionalRanker`) 기능 검증.
- **`test_state_router.py`**: 5축 피처 기반 상태 분류기(`MarketStateDetector`) 및 라우터(`PipelineRouter`) 동작 검증.
- **`test_online_hft.py`**: SGD 모델의 점진적 학습(`partial_fit`) 수치 변화 및 `ExecutionTracker`의 Cancel/Replace 로직 정상 동작 검증.
- **`test_system.py`**: 기술적 지표 생성, HMM 레짐 필터, LightGBM 배치 랭커, 전략 비중 할당 및 리스크 캡핑 등 기존 퀀트 파이프라인 컴포넌트의 E2E 검증.

실제 통합 동작을 확인하기 위해 `scripts/test_online_hft.py`와 `scripts/run_live_hft.py` 스크립트가 제공되어 스트리밍 틱 처리, 온라인 학습, 상태 전환, 동적 주문 생성에 이르는 전체 사이클을 시뮬레이션할 수 있습니다.
