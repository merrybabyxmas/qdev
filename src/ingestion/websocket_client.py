import time
import asyncio
from typing import Callable, Any, List

try:  # pragma: no cover - optional dependency guard
    from alpaca.data.live.crypto import CryptoDataStream
except Exception:  # pragma: no cover - optional dependency guard
    CryptoDataStream = None

from src.state.ring_buffers import TickRingBuffer, QuoteRingBuffer
from src.features.microstructure.imbalance import (compute_order_book_imbalance,
                                                  compute_microprice,
                                                  compute_spread,
                                                  compute_trade_intensity,
                                                  compute_toxicity_vpin_proxy,
                                                  compute_volatility_burst)
from src.utils.logger import logger

class MultiSymbolHFTStreamManager:
    """
    다중 종목(Multi-symbol) 이벤트 기반(Websocket) 실시간 HFT 스트림 수신 및 피처 갱신.
    각 종목별로 개별 버퍼(Ring Buffer)를 유지하고, 체결/호가 시마다 개별 피처를 발생시킴.
    또한 오프라인 리플레이(Replay) 기능을 지원함.
    """
    def __init__(
        self,
        api_key: str = "mock",
        secret_key: str = "mock",
        symbol: str = None,
        symbols: List[str] = None,
        enable_live_stream: bool = False,
    ):
        # Support both symbol= (singular, legacy) and symbols= (multi)
        if symbols is not None:
            self.symbols = list(symbols)
        elif symbol is not None:
            self.symbols = [symbol]
        else:
            self.symbols = ["BTC/USD"]
        self.symbol = self.symbols[0]  # legacy compat

        self.stream = None
        if enable_live_stream:
            if CryptoDataStream is None:
                raise ImportError("alpaca-py is not installed; live HFT stream mode is unavailable.")
            self.stream = CryptoDataStream(api_key, secret_key)

        # Per-symbol ring buffers
        self.trade_buffers: dict = {s: TickRingBuffer(capacity=10000) for s in self.symbols}
        self.quote_buffers: dict = {s: QuoteRingBuffer(capacity=1000) for s in self.symbols}

        # Backward-compat aliases for single-symbol usage
        self.trade_buffer = self.trade_buffers[self.symbol]
        self.quote_buffer = self.quote_buffers[self.symbol]

        # Callbacks — on_feature_update(symbol: str, feature_event: dict)
        self.on_feature_update: Callable[[str, dict], None] = None
        self.last_feature_event: dict = {sym: None for sym in self.symbols}
        self.last_event_received_at: float = None
        self.last_trade_at: float = None
        self.last_quote_at: float = None

    def _mark_event_received(self, kind: str) -> None:
        now = time.monotonic()
        self.last_event_received_at = now
        if kind == "trade":
            self.last_trade_at = now
        elif kind == "quote":
            self.last_quote_at = now

    def process_trade_snapshot(self, timestamp_ms: float, price: float, size: float, taker_side: str, symbol: str = None):
        """Process an offline trade event for replay or tests."""
        sym = symbol or self.symbol
        self._mark_event_received("trade")
        side = 1 if str(taker_side).upper().startswith("B") else -1
        if sym in self.trade_buffers:
            self.trade_buffers[sym].append(timestamp_ms, price, size, side)
            self._trigger_features(sym)

    def process_quote_snapshot(self, timestamp_ms: float, bid_price: float, bid_size: float, ask_price: float, ask_size: float, symbol: str = None):
        """Process an offline quote event for replay or tests."""
        sym = symbol or self.symbol
        self._mark_event_received("quote")
        if sym in self.quote_buffers:
            self.quote_buffers[sym].append(timestamp_ms, bid_price, bid_size, ask_price, ask_size)
            self._trigger_features(sym)

    def replay_events(self, events: list):
        """Replay a finite event stream without requiring a live websocket connection."""
        for event in events:
            event_type = event.get("type")
            sym = event.get("symbol", self.symbol)
            if not sym or sym not in self.symbols:
                continue

            if event_type == "quote":
                self.process_quote_snapshot(
                    float(event["timestamp_ms"]),
                    float(event.get("bid", event.get("bid_price"))),
                    float(event.get("bid_size", event.get("bid_size"))),
                    float(event.get("ask", event.get("ask_price"))),
                    float(event.get("ask_size", event.get("ask_size"))),
                    symbol=sym,
                )
            elif event_type == "trade":
                self.process_trade_snapshot(
                    float(event["timestamp_ms"]),
                    float(event["price"]),
                    float(event["size"]),
                    str(event.get("taker_side", "B")),
                    symbol=sym,
                )
            else:
                raise ValueError(f"Unsupported replay event type: {event_type}")

    async def _trade_handler(self, data: Any):
        """체결(Trade) 이벤트 처리기"""
        self._mark_event_received("trade")
        t = data.timestamp.timestamp() * 1000
        p = float(data.price)
        s = float(data.size)
        taker_side = getattr(data, "taker_side", None) or getattr(data, "side", None)
        side = 1 if str(taker_side).upper().startswith("B") else -1
        sym = data.symbol
        if sym in self.trade_buffers:
            self.trade_buffers[sym].append(t, p, s, side)
            logger.debug(f"Trade: {sym} @ {p} ({s})")
            self._trigger_features(sym)

    async def _quote_handler(self, data: Any):
        """호가(Quote / Top of Book) 이벤트 처리기"""
        self._mark_event_received("quote")
        t = data.timestamp.timestamp() * 1000
        bp = float(data.bid_price)
        bs = float(data.bid_size)
        ap = float(data.ask_price)
        as_ = float(data.ask_size)
        sym = data.symbol
        if sym in self.quote_buffers:
            self.quote_buffers[sym].append(t, bp, bs, ap, as_)
            logger.debug(f"Quote: {sym} B {bs}@{bp} - A {as_}@{ap}")
            self._trigger_features(sym)

    def _trigger_features(self, symbol: str):
        """틱, 호가 업데이트 후 즉각적인 마이크로스트럭처 계산"""
        quote = self.quote_buffers[symbol].get_latest()
        if quote[1] == 0.0:  # No quote yet
            return

        timestamp, bp, bs, ap, ask_s = quote

        obi = compute_order_book_imbalance(bs, ask_s)
        microprice = compute_microprice(bp, bs, ap, ask_s)
        spread = compute_spread(bp, ap)

        recent_trades = self.trade_buffers[symbol].get_recent(50)
        intensity = compute_trade_intensity(recent_trades, window_ms=1000.0)
        toxicity = compute_toxicity_vpin_proxy(recent_trades, window_ms=1000.0)
        vol_burst = compute_volatility_burst(recent_trades, window_ms=1000.0)

        feature_event = {
            "timestamp": timestamp,
            "bid": bp,
            "bid_size": bs,
            "ask": ap,
            "ask_size": ask_s,
            "microprice": microprice,
            "obi": obi,
            "spread": spread,
            "intensity": intensity,
            "toxicity_vpin": toxicity,
            "volatility_burst": vol_burst,
            "mid_price": (bp + ap) / 2.0
        }
        self.last_feature_event[symbol] = feature_event
        if self.on_feature_update:
            self.on_feature_update(symbol, feature_event)

    def run(self):
        """스트림 구독 시작 (블로킹 루프)"""
        if self.stream is None:
            raise RuntimeError("Live stream is disabled. Instantiate with enable_live_stream=True to run() against Alpaca.")

        logger.info(f"Starting Websocket stream for {self.symbols}...")
        for sym in self.symbols:
            self.stream.subscribe_trades(self._trade_handler, sym)
            self.stream.subscribe_quotes(self._quote_handler, sym)

        self.stream.run()
