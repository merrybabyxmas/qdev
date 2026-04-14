import asyncio
import numpy as np
from typing import Callable, Any
from alpaca.data.live.crypto import CryptoDataStream
from src.state.ring_buffers import TickRingBuffer, QuoteRingBuffer
from src.features.microstructure.imbalance import compute_order_book_imbalance, compute_microprice, compute_spread, compute_trade_intensity
from src.utils.logger import logger

class HFTStreamManager:
    """
    이벤트 기반(Websocket) 실시간 HFT 스트림 수신 및 피처 갱신.
    온라인 학습과 취소/수정 체결까지 실시간 루프에서 동작하도록 설계됨.
    """
    def __init__(self, api_key: str = "mock", secret_key: str = "mock", symbol: str = "BTC/USD"):
        self.stream = CryptoDataStream(api_key, secret_key)
        self.symbol = symbol

        self.trade_buffer = TickRingBuffer(capacity=10000)
        self.quote_buffer = QuoteRingBuffer(capacity=1000)

        self.on_feature_update: Callable[[dict], None] = None

    async def _trade_handler(self, data: Any):
        t = data.timestamp.timestamp() * 1000 # ms
        p = float(data.price)
        s = float(data.size)
        side = 1 if data.taker_side == 'B' else -1

        self.trade_buffer.append(t, p, s, side)
        logger.debug(f"Trade: {self.symbol} @ {p} ({s})")
        self._trigger_features()

    async def _quote_handler(self, data: Any):
        t = data.timestamp.timestamp() * 1000
        bp = float(data.bid_price)
        bs = float(data.bid_size)
        ap = float(data.ask_price)
        as_ = float(data.ask_size)

        self.quote_buffer.append(t, bp, bs, ap, as_)
        logger.debug(f"Quote: B {bs}@{bp} - A {as_}@{ap}")
        self._trigger_features()

    def _trigger_features(self):
        if not self.on_feature_update:
            return

        quote = self.quote_buffer.get_latest()
        if quote[1] == 0.0:  # No quote yet
            return

        timestamp, bp, bs, ap, ask_s = quote

        obi = compute_order_book_imbalance(bs, ask_s)
        microprice = compute_microprice(bp, bs, ap, ask_s)
        spread = compute_spread(bp, ap)

        recent_trades = self.trade_buffer.get_recent(50)
        intensity = compute_trade_intensity(recent_trades, window_ms=1000.0)

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
            "mid_price": (bp + ap) / 2.0
        }
        self.on_feature_update(feature_event)

    def run(self):
        logger.info(f"Starting Websocket stream for {self.symbol}...")
        self.stream.subscribe_trades(self._trade_handler, self.symbol)
        self.stream.subscribe_quotes(self._quote_handler, self.symbol)
        self.stream.run()
