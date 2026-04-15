import time
import asyncio
import numpy as np
from typing import Callable, Any, List

try:  # optional dependency guard
    from alpaca.data.live.crypto import CryptoDataStream
except ImportError:
    CryptoDataStream = None

from src.state.ring_buffers import TickRingBuffer, QuoteRingBuffer
from src.features.microstructure.imbalance import (compute_order_book_imbalance,
                                                  compute_microprice,
                                                  compute_spread,
                                                  compute_trade_intensity,
                                                  compute_toxicity_vpin_proxy,
                                                  compute_volatility_burst,
                                                  compute_jump_proxy)
from src.utils.logger import logger

class MultiSymbolHFTStreamManager:
    def __init__(self, api_key: str = "mock", secret_key: str = "mock", symbols: List[str] = ["BTC/USD", "ETH/USD"], enable_live_stream: bool = True):
        self.symbols = symbols
        self.stream = None

        if enable_live_stream:
            if CryptoDataStream is None:
                raise ImportError("alpaca-py is not installed; live HFT stream mode is unavailable.")
            self.stream = CryptoDataStream(api_key, secret_key)

        self.trade_buffers = {sym: TickRingBuffer(capacity=10000) for sym in symbols}
        self.quote_buffers = {sym: QuoteRingBuffer(capacity=1000) for sym in symbols}

        self.on_feature_update: Callable[[str, dict], None] = None
        self.last_feature_event = {sym: None for sym in symbols}
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

    def process_trade_snapshot(self, symbol: str, timestamp_ms: float, price: float, size: float, taker_side: str):
        self._mark_event_received("trade")
        side = 1 if str(taker_side).upper().startswith("B") else -1
        if symbol in self.trade_buffers:
            self.trade_buffers[symbol].append(timestamp_ms, price, size, side)
            self._trigger_features(symbol)

    def process_quote_snapshot(self, symbol: str, timestamp_ms: float, bid_price: float, bid_size: float, ask_price: float, ask_size: float):
        self._mark_event_received("quote")
        if symbol in self.quote_buffers:
            self.quote_buffers[symbol].append(timestamp_ms, bid_price, bid_size, ask_price, ask_size)
            self._trigger_features(symbol)

    def replay_events(self, events: list[dict[str, Any]]):
        for event in events:
            event_type = event.get("type")
            symbol = event.get("symbol")
            if not symbol or symbol not in self.symbols:
                continue

            if event_type == "quote":
                self.process_quote_snapshot(
                    symbol,
                    float(event["timestamp_ms"]),
                    float(event.get("bid", event.get("bid_price"))),
                    float(event.get("bid_size", event.get("bid_size"))),
                    float(event.get("ask", event.get("ask_price"))),
                    float(event.get("ask_size", event.get("ask_size"))),
                )
            elif event_type == "trade":
                self.process_trade_snapshot(
                    symbol,
                    float(event["timestamp_ms"]),
                    float(event["price"]),
                    float(event["size"]),
                    str(event.get("taker_side", "B")),
                )
            else:
                raise ValueError(f"Unsupported replay event type: {event_type}")

    async def _trade_handler(self, data: Any):
        self._mark_event_received("trade")
        t = data.timestamp.timestamp() * 1000 # ms
        p = float(data.price)
        s = float(data.size)

        taker_side = getattr(data, "taker_side", None)
        if taker_side is None:
            taker_side = getattr(data, "side", None)
        side = 1 if str(taker_side).upper().startswith("B") else -1

        sym = data.symbol

        if sym in self.trade_buffers:
            self.trade_buffers[sym].append(t, p, s, side)
            logger.debug(f"Trade: {sym} @ {p} ({s})")
            self._trigger_features(sym)

    async def _quote_handler(self, data: Any):
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
        jump_proxy = compute_jump_proxy(recent_trades, window_ms=1000.0, jump_threshold=3.0)

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
            "jump_proxy": jump_proxy,
            "mid_price": (bp + ap) / 2.0
        }

        self.last_feature_event[symbol] = feature_event
        if self.on_feature_update:
            self.on_feature_update(symbol, feature_event)

    def run(self):
        if self.stream is None:
            raise RuntimeError("Live stream is disabled. Instantiate with enable_live_stream=True to run() against Alpaca.")

        logger.info(f"Starting Multi-Symbol Websocket stream for {self.symbols}...")
        self.stream.subscribe_trades(self._trade_handler, *self.symbols)
        self.stream.subscribe_quotes(self._quote_handler, *self.symbols)
        self.stream.run()
