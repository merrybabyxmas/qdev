import asyncio
import json
import time
from datetime import datetime, timezone
import numpy as np
import torch
from collections import deque
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
from src.hft.models.sde.avellaneda_stoikov import AvellanedaStoikovMarketMaker
from src.hft.execution.fill_prob import FillProbabilityGate
from src.models.logistic_online import OnlineLogisticDirectionClassifier
from src.hft.models.dl.event_lstm import EventSequenceLSTM
from src.hft.models.dl.deeplob import CompactDeepLOB
from src.utils.logger import logger
from src.controlplane.artifacts import CONTROL_PLANE_ROOT

_HFT_STATUS_PATH = CONTROL_PLANE_ROOT / "hft_status.json"
_HFT_TICKS_PATH = CONTROL_PLANE_ROOT / "logs" / "hft_ticks.jsonl"
_HFT_TICKS_MAX_LINES = 5000

class LiveTradingEngine:
    """
    HFT Execution Plane:
    모든 HFT 파이프라인 (HFT_BASE, HFT_SDE, HFT_DL, HFT_RISK, HFT_HYB)을 종합하여
    초고빈도 틱 데이터를 바탕으로 의사결정 및 주문 실행.
    Control Plane(스케줄러)이 발행한 hft_policy.json을 주기적으로 읽어 실행 여부를 상위에서 통제받습니다.
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

        # Load persisted OnlineLGBM state if available
        from src.models.lgbm_online import OnlineLightGBMRanker
        _lgbm_path = CONTROL_PLANE_ROOT / "models" / "lgbm_online.pkl"
        if _lgbm_path.exists():
            try:
                self.ranker.lgbm = OnlineLightGBMRanker.load(str(_lgbm_path))
                logger.info("Loaded existing OnlineLGBM state from disk.")
            except Exception as _lgbm_err:
                logger.warning(f"Could not load LGBM state: {_lgbm_err}")

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

        # HFT Execution / Policies
        self.tracker = ExecutionTracker(broker_or_engine=self.broker, cancel_threshold_bps=1.0, max_order_age_ms=5000.0)
        self.risk_manager = RiskManager(max_position_cap=0.40)
        self.fill_prob_gate = FillProbabilityGate(min_fill_prob=0.3)
        self.as_market_maker = AvellanedaStoikovMarketMaker(risk_aversion=0.1, time_horizon=1.0)

        # Additional HFT Models (HFT_BASE_002, HFT_DL_001, HFT_DL_002)
        self.logistic_classifier = OnlineLogisticDirectionClassifier()

        # Sequence tracking for DL models (require sequence of ticks, e.g. length=10)
        self.dl_sequence_length = 10
        self.event_lstm = EventSequenceLSTM(input_features=5, sequence_length=self.dl_sequence_length, output_dim=1)
        self.event_lstm.eval()

        self.deeplob = CompactDeepLOB(input_features=5, sequence_length=self.dl_sequence_length, num_classes=3)
        self.deeplob.eval()

        self.event_history = {sym: deque(maxlen=self.dl_sequence_length) for sym in symbols}

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

        # 2. HFT_BASE_001: Online SGD + LGBM Ranker
        sgd_preds, lgbm_preds, raw_targets = self.ranker.update_and_predict(symbol, feature_event)
        target_weights = self.risk_manager.apply_position_caps(raw_targets)

        target_weight_for_sym = target_weights.get(symbol, 0.0)
        pred_for_sym = sgd_preds.get(symbol, 0.0)
        lgbm_pred_for_sym = lgbm_preds.get(symbol, 0.0)

        if self.tick_counter % 50 == 0:
            logger.info(f"[{symbol}] State: {current_state.name} | Target W: {target_weight_for_sym:.2%} | SGD: {pred_for_sym:+.2f} bps | LGBM: {lgbm_pred_for_sym:+.2f} bps")

        # Feature vector for alternative models
        feat_vec = np.array([
            feature_event.get("obi", 0.0),
            feature_event.get("microprice_drift", 0.0),
            feature_event.get("spread", 0.0),
            feature_event.get("toxicity_vpin", 0.0),
            feature_event.get("volatility_burst", 0.0),
        ])

        # 3. HFT_BASE_002: Online Logistic Direction Classifier
        target_label = np.array([1 if pred_for_sym > 0.5 else (-1 if pred_for_sym < -0.5 else 0)])
        self.logistic_classifier.update(feat_vec.reshape(1, -1), target_label)
        logistic_pred_probs = self.logistic_classifier.predict_proba(feat_vec.reshape(1, -1))
        # Convert class probabilities to signed bps signal: UP=+1, DOWN=-1 weighted by confidence
        logistic_pred_bps = float((logistic_pred_probs[0, 2] - logistic_pred_probs[0, 0]) * 2.0)  # range [-2, +2]

        # Ensemble: SGD base + logistic modifier + LGBM blend
        ensemble_pred = pred_for_sym
        if logistic_pred_probs[0, 2] > 0.6:   # High prob of UP
            ensemble_pred += 1.0
        elif logistic_pred_probs[0, 0] > 0.6:  # High prob of DOWN
            ensemble_pred -= 1.0
        if lgbm_pred_for_sym != 0.0:
            ensemble_pred = ensemble_pred * 0.7 + lgbm_pred_for_sym * 0.3

        # 4. HFT_DL_001 / HFT_DL_002: Sequence-based DL models
        lstm_pred_bps = 0.0
        deeplob_pred_bps = 0.0
        self.event_history[symbol].append(feat_vec)
        if len(self.event_history[symbol]) == self.dl_sequence_length:
            seq_tensor = torch.tensor(np.array(self.event_history[symbol]), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                # HFT_DL_002: Event Sequence LSTM
                lstm_pred_bps = self.event_lstm(seq_tensor).item()
                ensemble_pred += lstm_pred_bps * 0.1

                # HFT_DL_001: Compact DeepLOB
                lob_out = self.deeplob(seq_tensor)
                lob_class = torch.argmax(lob_out, dim=1).item()
                deeplob_pred_bps = float(lob_class - 1)  # -1 / 0 / +1
                if lob_class == 2:    # Up
                    ensemble_pred += 0.5
                elif lob_class == 0:  # Down
                    ensemble_pred -= 0.5

        # 5. Context-Aware Pipeline Execution Routing
        action: ExecutionAction = self.router.route_execution(
            state=current_state,
            prediction=ensemble_pred,
            policy=self.current_policy,
            symbol=symbol,
            features=feature_event,
        )

        if action.action == "HALT":
            target_weight_for_sym = 0.0

        # HFT_EXEC_002: Cancel/Replace Tracker
        self.tracker.evaluate_cancel_replace(t, symbol, b, a)

        mid_price = feature_event["mid_price"]
        # Use broker's reported buying power if available (synced every 60s)
        available_cash = getattr(self.broker, "buying_power", None)
        if available_cash is None:
            available_cash = self.cached_equity - sum(
                self.cached_positions.get(s, 0.0) * mid_price for s in self.symbols
            )
        available_cash = max(float(available_cash), 0.0)
        delta_qty = self.risk_manager.calculate_order_qty(
            symbol, target_weight_for_sym, current_qty, mid_price, equity,
            available_cash=available_cash,
            max_order_usd=min(500.0, available_cash * 0.90),
        )

        # 6. Execute newly proposed actions
        active_orders_for_sym = {k: v for k, v in self.tracker.active_orders.items() if v["symbol"] == symbol}

        if action.action == "HALT" or delta_qty == 0.0:
            for oid in list(active_orders_for_sym.keys()):
                try:
                    self.broker.cancel_order(oid, t)
                except TypeError:
                    self.broker.cancel_order(oid)
                self.tracker.untrack_order(oid)

        elif action.action == "PASSIVE_MAKE" and not active_orders_for_sym and delta_qty != 0.0:
            side = "buy" if delta_qty > 0 else "sell"
            qty = abs(delta_qty) * action.size_multiplier

            # HFT_SDE_001: Avellaneda-Stoikov inventory-aware pricing
            opt_bid, opt_ask = self.as_market_maker.calculate_quotes(
                mid_price=mid_price,
                inventory=current_qty,
                volatility=feature_event.get("volatility_burst", 0.01),
                current_time=0.0,
            )
            price = opt_bid if side == "buy" else opt_ask
            price = min(price, b) if side == "buy" else max(price, a)

            # HFT_EXEC_001: Fill Probability Pre-Check
            fill_feat = np.array([feature_event["spread"], 0.1, feature_event["bid_size"], 0.0])
            if self.fill_prob_gate.is_executable(fill_feat):
                try:
                    oid = self.broker.place_limit_order(symbol, side, price, qty, t)
                except TypeError:
                    oid = self.broker.place_limit_order(symbol, side, price, qty)
                if oid:
                    self.tracker.track_order(oid, symbol, side, price, qty, t)

        elif action.action == "AGGRESSIVE_TAKE" and not active_orders_for_sym and delta_qty != 0.0:
            side = "buy" if delta_qty > 0 else "sell"
            qty = abs(delta_qty) * action.size_multiplier
            price = a if side == "buy" else b

            try:
                oid = self.broker.place_limit_order(symbol, side, price, qty, t)
            except TypeError:
                oid = self.broker.place_limit_order(symbol, side, price, qty)
            if oid:
                self.tracker.track_order(oid, symbol, side, price, qty, t)

        # Status file write every 10 ticks per symbol
        self._sym_tick_counter[symbol] = self._sym_tick_counter.get(symbol, 0) + 1
        if self._sym_tick_counter[symbol] % 10 == 0:
            try:
                CONTROL_PLANE_ROOT.mkdir(parents=True, exist_ok=True)
                (CONTROL_PLANE_ROOT / "logs").mkdir(parents=True, exist_ok=True)

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
                    "lgbm_prediction_bps": float(lgbm_pred_for_sym),
                    "logistic_prediction_bps": float(logistic_pred_bps),
                    "lstm_prediction_bps": float(lstm_pred_bps),
                    "deeplob_prediction_bps": float(deeplob_pred_bps),
                    "ensemble_prediction_bps": float(ensemble_pred),
                    "target_weight": float(target_weight_for_sym),
                    "tick_count": int(self.ranker.tick_counts.get(symbol, 0)),
                }

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
                        "lgbm_updates": int(self.ranker.lgbm.total_updates),
                        "lgbm_fitted": bool(self.ranker.lgbm.is_fitted),
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

        # Save model states every 500 ticks
        if self.tick_counter % 500 == 0:
            try:
                _sgd_save_path = CONTROL_PLANE_ROOT / "models" / "sgd_online.pkl"
                self.ranker.model.save(str(_sgd_save_path))
                logger.debug(f"[SGD save] persisted at tick {self.tick_counter}")
            except Exception as _sgd_e:
                logger.debug(f"[SGD save] skipped: {_sgd_e}")
            try:
                if self.ranker.lgbm.is_fitted:
                    _lgbm_save_path = CONTROL_PLANE_ROOT / "models" / "lgbm_online.pkl"
                    self.ranker.lgbm.save(str(_lgbm_save_path))
                    logger.debug(f"[LGBM save] persisted at tick {self.tick_counter}")
            except Exception as _lgbm_e:
                logger.debug(f"[LGBM save] skipped: {_lgbm_e}")

        # Simulate Matching Process if running offline/simulation broker
        if self.is_simulation:
            self.broker.process_quote_update(t, b, a)

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
