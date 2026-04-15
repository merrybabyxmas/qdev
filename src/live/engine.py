import asyncio
import json
import time
from datetime import datetime, timezone
import numpy as np
from src.ingestion.websocket_client import MultiSymbolHFTStreamManager
from src.models.state_detector import MarketStateDetector
from src.models.ranker_engine import RealTimeCrossSectionalRanker
from src.signals.router import PipelineRouter, ExecutionAction
from src.execution.policy import ExecutionTracker
from src.risk.manager import RiskManager
from src.brokers.base import BrokerInterface
from src.models.champion_registry import ChampionRegistry
from src.models.sgd_online import OnlineSGDRegressor
from src.monitoring.control_plane import HFTControlPlane
from src.utils.logger import logger
from src.controlplane.artifacts import CONTROL_PLANE_ROOT

_HFT_STATUS_PATH = CONTROL_PLANE_ROOT / "hft_status.json"
_HFT_TICKS_PATH = CONTROL_PLANE_ROOT / "logs" / "hft_ticks.jsonl"
_HFT_TICKS_MAX_LINES = 5000

class LiveTradingEngine:
    """
    모든 HFT 컴포넌트(스트림 수신 -> 마이크로스트럭처 계산 -> 온라인 학습 -> 랭킹 -> 상태 기반 파이프라인 라우팅 -> 취소/재주문 실행)를
    무한 루프(Event-driven Loop) 안에서 엮는 실제 트레이딩 엔진.
    HFT Execution Plane: Control Plane(스케줄러)이 발행한 hft_policy.json을 주기적으로 읽어 실행 여부를 상위에서 통제받습니다.
    """
    def __init__(self, symbols, broker_engine, is_simulation=False, api_key: str = "mock", secret_key: str = "mock", enable_live_stream: bool = False):
        self.symbols = symbols
        self.broker = broker_engine
        self.enable_live_stream = enable_live_stream

        self.stream_manager = MultiSymbolHFTStreamManager(
            symbols=symbols,
            api_key=api_key,
            secret_key=secret_key,
            enable_live_stream=enable_live_stream,
        )
        self.stream_manager.on_feature_update = self._on_feature_event

        self.detector = MarketStateDetector(high_vol_threshold=0.5, toxic_vpin_threshold=0.8, trend_threshold=0.001)
        self.ranker = RealTimeCrossSectionalRanker(symbols=symbols, target_lookahead=5)

        # Load persisted OnlineSGD state if available
        _sgd_path = CONTROL_PLANE_ROOT / "models" / "sgd_online.pkl"
        if _sgd_path.exists():
            try:
                self.ranker.model = OnlineSGDRegressor.load(str(_sgd_path))
                logger.info("Loaded existing OnlineSGD state from disk.")
            except Exception as _sgd_err:
                logger.warning(f"Could not load SGD state: {_sgd_err}")

        # Load the HFT champion model from the specialized leaderboard
        try:
            self.registry = ChampionRegistry()
            champ = self.registry.get_champion("hft_microstructure")
            logger.info(f"Loaded HFT Champion Model: {champ.get('model_id', 'Default')} (Sharpe: {champ.get('sharpe', 0.0)})")
        except Exception as e:
            logger.warning(f"ChampionRegistry not available: {e}")

        # Control Plane Integration
        try:
            self.control_plane = HFTControlPlane()
            self.current_policy = self.control_plane.read_policy()
        except Exception as e:
            logger.warning(f"HFTControlPlane not available, using default policy: {e}")
            self.control_plane = None
            self.current_policy = {"allow_hft": True, "symbols": {}, "thresholds": {}}
        self.last_policy_sync_time = time.time()
        self.policy_sync_interval = 10.0  # 스케줄러 정책을 10초마다 확인

        self.router = PipelineRouter()
        self.tracker = ExecutionTracker(broker_or_engine=self.broker, cancel_threshold_bps=1.0)
        self.risk_manager = RiskManager(max_position_cap=0.40)

        self.is_simulation = is_simulation
        self.tick_counter = 0
        self._sym_tick_counter: dict = {s: 0 for s in symbols}

        self.cached_equity = 100000.0
        self.cached_positions = {sym: 0.0 for sym in symbols}
        self.last_account_sync_time = 0.0
        self.account_sync_interval = 60.0

    def _sync_account_state(self):
        now = time.time()
        if now - self.last_account_sync_time > self.account_sync_interval:
            try:
                acct = self.broker.get_account()
                self.cached_equity = acct.get("equity", self.cached_equity)

                positions = self.broker.get_positions()
                for sym in self.symbols:
                    self.cached_positions[sym] = positions.get(sym, 0.0)

                self.risk_manager.evaluate_account_risk(self.cached_equity)
                self.last_account_sync_time = now
            except Exception as e:
                logger.error(f"Failed to sync account info: {e}")

    def _sync_macro_policy(self):
        """10초마다 상위 스케줄러가 작성한 hft_policy.json 정책을 로드하여 엔진 동기화"""
        if self.control_plane is None:
            return
        now = time.time()
        if now - self.last_policy_sync_time > self.policy_sync_interval:
            new_policy = self.control_plane.read_policy()
            if self.current_policy.get("allow_hft") != new_policy.get("allow_hft"):
                logger.info(f"Control Plane Policy Changed! allow_hft: {new_policy.get('allow_hft')}")
            self.current_policy = new_policy
            self.last_policy_sync_time = now

    def _on_feature_event(self, symbol: str, feature_event: dict):
        """Websocket 스트림에서 틱마다 콜백됨."""
        self.tick_counter += 1
        t = feature_event["timestamp"]
        b = feature_event["bid"]
        a = feature_event["ask"]

        self._sync_account_state()
        self._sync_macro_policy()

        equity = self.cached_equity
        current_qty = self.cached_positions.get(symbol, 0.0)

        # 1. State Detection
        current_state = self.detector.detect_state(feature_event)

        # 2. Continuous Cross-sectional Ranking & Online Update
        preds, raw_targets = self.ranker.update_and_predict(symbol, feature_event)

        # 3. Apply Portfolio Caps
        target_weights = self.risk_manager.apply_position_caps(raw_targets)

        target_weight_for_sym = target_weights.get(symbol, 0.0)
        pred_for_sym = preds.get(symbol, 0.0)

        if self.tick_counter % 50 == 0:
            logger.info(f"[{symbol}] State: {current_state.name} | Target W: {target_weight_for_sym:.2%} | Raw Pred: {pred_for_sym:+.2f} bps")

        # 4. Context-Aware Pipeline Execution Routing (with macro policy)
        action: ExecutionAction = self.router.route_execution(
            state=current_state,
            prediction=pred_for_sym,
            policy=self.current_policy,
            symbol=symbol
        )

        if action.action == "HALT":
            # Override target weight to 0 to safely exit any open positions when halted
            target_weight_for_sym = 0.0

        # 5. Execution Policy (Cancel/Replace tracking)
        self.tracker.evaluate_cancel_replace(t, symbol, b, a)

        mid_price = feature_event["mid_price"]
        delta_qty = self.risk_manager.calculate_order_qty(symbol, target_weight_for_sym, current_qty, mid_price, equity)

        # 6. Execute newly proposed actions
        active_orders_for_sym = {k: v for k, v in self.tracker.active_orders.items() if v["symbol"] == symbol}

        if action.action == "HALT" or delta_qty == 0.0:
            for oid in list(active_orders_for_sym.keys()):
                if self.is_simulation:
                    self.broker.cancel_order(oid, t)
                else:
                    self.broker.cancel_order(oid)
                self.tracker.untrack_order(oid)

        elif action.action == "PASSIVE_MAKE" and not active_orders_for_sym and delta_qty != 0.0:
            side = "buy" if delta_qty > 0 else "sell"
            qty = abs(delta_qty)
            price = b if side == "buy" else a

            if self.is_simulation:
                oid = self.broker.place_limit_order(symbol, side, price, qty, t)
            else:
                oid = self.broker.place_limit_order(symbol, side, price, qty)
            if oid:
                self.tracker.track_order(oid, symbol, side, price, qty)

        elif action.action == "AGGRESSIVE_TAKE" and not active_orders_for_sym and delta_qty != 0.0:
            side = "buy" if delta_qty > 0 else "sell"
            qty = abs(delta_qty)
            price = a if side == "buy" else b

            if self.is_simulation:
                oid = self.broker.place_limit_order(symbol, side, price, qty, t)
            else:
                oid = self.broker.place_limit_order(symbol, side, price, qty)
            if oid:
                self.tracker.track_order(oid, symbol, side, price, qty)

        # Status file write every 10 ticks per symbol
        self._sym_tick_counter[symbol] = self._sym_tick_counter.get(symbol, 0) + 1
        if self._sym_tick_counter[symbol] % 10 == 0:
            try:
                CONTROL_PLANE_ROOT.mkdir(parents=True, exist_ok=True)
                (CONTROL_PLANE_ROOT / "logs").mkdir(parents=True, exist_ok=True)

                # Build per-symbol entry
                fe = feature_event
                sym_entry = {
                    "price": float((b + a) / 2.0),
                    "bid": float(b),
                    "ask": float(a),
                    "spread": float(a - b),
                    "obi": float(fe.get("obi", 0.0)),
                    "microprice": float(fe.get("microprice", (b + a) / 2.0)),
                    "toxicity_vpin": float(fe.get("toxicity_vpin", 0.0)),
                    "volatility_burst": float(fe.get("volatility_burst", 0.0)),
                    "intensity": float(fe.get("intensity", 0.0)),
                    "market_state": current_state.name,
                    "prediction_bps": float(pred_for_sym),
                    "target_weight": float(target_weight_for_sym),
                    "tick_count": int(self.ranker.tick_counts.get(symbol, 0)),
                }

                # Load existing status to merge symbol entries
                existing_symbols: dict = {}
                if _HFT_STATUS_PATH.exists():
                    try:
                        existing_symbols = json.loads(_HFT_STATUS_PATH.read_text(encoding="utf-8")).get("symbols", {})
                    except Exception:
                        existing_symbols = {}
                existing_symbols[symbol] = sym_entry

                total_ticks = sum(self.ranker.tick_counts.values())
                status_payload = {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "tick_counter": int(self.tick_counter),
                    "symbols": existing_symbols,
                    "model": {
                        "total_updates": int(total_ticks),
                        "n_features": 5,
                    },
                    "broker": {
                        "cash": float(getattr(self.broker, "cash", self.cached_equity)),
                        "equity": float(self.cached_equity),
                        "inventory": float(sum(self.cached_positions.values())),
                        "pnl": float(getattr(self.broker, "pnl", 0.0)),
                        "active_orders": int(len(self.tracker.active_orders)),
                        "mode": "paper" if (not self.is_simulation and getattr(self.broker, 'paper', True)) else ("live" if not self.is_simulation else "simulation"),
                    },
                }
                _HFT_STATUS_PATH.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")

                # Append tick record to JSONL, keep last 5000 lines
                tick_record = {"timestamp": status_payload["updated_at"], "symbol": symbol, **sym_entry}
                existing_lines: list = []
                if _HFT_TICKS_PATH.exists():
                    existing_lines = _HFT_TICKS_PATH.read_text(encoding="utf-8").splitlines()
                existing_lines.append(json.dumps(tick_record))
                if len(existing_lines) > _HFT_TICKS_MAX_LINES:
                    existing_lines = existing_lines[-_HFT_TICKS_MAX_LINES:]
                _HFT_TICKS_PATH.write_text("\n".join(existing_lines) + "\n", encoding="utf-8")
            except Exception as _e:
                logger.debug(f"[HFT status write] skipped: {_e}")

        # Save OnlineSGD state every 500 ticks
        if self.tick_counter % 500 == 0:
            try:
                _sgd_save_path = CONTROL_PLANE_ROOT / "models" / "sgd_online.pkl"
                self.ranker.model.save(str(_sgd_save_path))
                logger.debug(f"[SGD save] persisted at tick {self.tick_counter}")
            except Exception as _sgd_e:
                logger.debug(f"[SGD save] skipped: {_sgd_e}")

        # Simulate Matching Process if running offline/simulation broker
        if self.is_simulation:
            self.broker.process_quote_update(t, b, a)

            # Clean up filled orders from tracker
            for oid in list(self.tracker.active_orders.keys()):
                if oid not in getattr(self.broker, "active_orders", {}):
                    self.tracker.untrack_order(oid)

            self.cached_positions[symbol] = getattr(self.broker, "inventory", self.cached_positions[symbol])

    def _run_simulation(self):
        """시뮬레이션 모드: 합성 랜덤워크 틱을 생성해 스트림 매니저로 주입."""
        SEED_PRICES = {"BTC/USD": 83000.0, "ETH/USD": 1600.0}
        mid = {s: SEED_PRICES.get(s, 1000.0) for s in self.symbols}
        t_ms = time.time() * 1000
        tick_interval = 0.05  # 20 ticks/sec per symbol

        logger.info("Simulation tick loop started.")
        while True:
            for sym in self.symbols:
                # Random walk step
                shock = np.random.normal(0, mid[sym] * 0.0002)
                mid[sym] = max(mid[sym] + shock, 1.0)
                half_spread = mid[sym] * 0.0001
                bid = mid[sym] - half_spread
                ask = mid[sym] + half_spread
                bid_size = round(np.random.exponential(0.5) + 0.1, 4)
                ask_size = round(np.random.exponential(0.5) + 0.1, 4)
                trade_size = round(np.random.exponential(0.1) + 0.01, 4)
                taker_side = "B" if np.random.rand() > 0.5 else "S"

                self.stream_manager.process_quote_snapshot(t_ms, bid, bid_size, ask, ask_size, symbol=sym)
                self.stream_manager.process_trade_snapshot(t_ms, mid[sym], trade_size, taker_side, symbol=sym)

            t_ms += tick_interval * 1000
            time.sleep(tick_interval)

    def start(self):
        """실시간(Live) 트레이딩 루프 시작 (블로킹)"""
        logger.info("=== Starting Continuous Live Trading Engine (Execution Plane) ===")
        if self.control_plane is not None:
            initial_policy = self.control_plane.read_policy()
            logger.info(f"Initial Policy Status - HFT Allowed: {initial_policy.get('allow_hft')}")

        if not self.is_simulation:
            self._sync_account_state()

        if self.enable_live_stream:
            logger.info("Using real Alpaca WebSocket stream.")
            self.stream_manager.run()
        else:
            logger.info("Live stream disabled — using synthetic tick simulation.")
            self._run_simulation()
