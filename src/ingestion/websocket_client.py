import asyncio
import numpy as np
from typing import Callable, Any, List
from alpaca.data.live.crypto import CryptoDataStream
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
    """
    def __init__(self, api_key: str = "mock", secret_key: str = "mock", symbols: List[str] = ["BTC/USD", "ETH/USD"]):
        self.stream = CryptoDataStream(api_key, secret_key)
        self.symbols = symbols

        # State stores per symbol
        self.trade_buffers = {sym: TickRingBuffer(capacity=10000) for sym in symbols}
        self.quote_buffers = {sym: QuoteRingBuffer(capacity=1000) for sym in symbols}

        self.on_feature_update: Callable[[str, dict], None] = None

    async def _trade_handler(self, data: Any):
        t = data.timestamp.timestamp() * 1000 # ms
        p = float(data.price)
        s = float(data.size)
        side = 1 if data.taker_side == 'B' else -1
        sym = data.symbol

        if sym in self.trade_buffers:
            self.trade_buffers[sym].append(t, p, s, side)
            logger.debug(f"Trade: {sym} @ {p} ({s})")
            self._trigger_features(sym)

    async def _quote_handler(self, data: Any):
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
        if not self.on_feature_update:
            return

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
        # Fire event upward to the engine
        self.on_feature_update(symbol, feature_event)

    def run(self):
        logger.info(f"Starting Multi-Symbol Websocket stream for {self.symbols}...")
        self.stream.subscribe_trades(self._trade_handler, *self.symbols)
        self.stream.subscribe_quotes(self._quote_handler, *self.symbols)
        self.stream.run()
