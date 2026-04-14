import asyncio
from typing import Callable, Any
from alpaca.data.live.crypto import CryptoDataStream
from src.state.ring_buffers import TickRingBuffer, QuoteRingBuffer
from src.features.microstructure.imbalance import compute_order_book_imbalance, compute_microprice, compute_spread
from src.utils.logger import logger

class HFTStreamManager:
    """
    이벤트 기반(Websocket) 실시간 HFT 스트림 수신 및 피처 갱신.
    """
    def __init__(self, api_key: str = "mock", secret_key: str = "mock", symbol: str = "BTC/USD"):
        # Alpaca crypto WS doesn't strictly need valid keys for basic feeds
        self.stream = CryptoDataStream(api_key, secret_key)
        self.symbol = symbol

        # State stores
        self.trade_buffer = TickRingBuffer(capacity=10000)
        self.quote_buffer = QuoteRingBuffer(capacity=1000)

        # Callbacks
        self.on_feature_update: Callable[[dict], None] = None

    async def _trade_handler(self, data: Any):
        """체결(Trade) 이벤트 처리기"""
        t = data.timestamp.timestamp() * 1000 # to ms
        p = float(data.price)
        s = float(data.size)
        side = 1 if data.taker_side == 'B' else -1

        self.trade_buffer.append(t, p, s, side)

        logger.debug(f"Trade event: {self.symbol} @ {p} (size: {s})")
        self._trigger_features()

    async def _quote_handler(self, data: Any):
        """호가(Quote / Top of Book) 이벤트 처리기"""
        t = data.timestamp.timestamp() * 1000
        bp = float(data.bid_price)
        bs = float(data.bid_size)
        ap = float(data.ask_price)
        as_ = float(data.ask_size)

        self.quote_buffer.append(t, bp, bs, ap, as_)

        logger.debug(f"Quote event: B {bs}@{bp} - A {as_}@{ap}")
        self._trigger_features()

    def _trigger_features(self):
        """틱, 호가 업데이트 후 즉각적인 마이크로스트럭처 계산"""
        if not self.on_feature_update:
            return

        quote = self.quote_buffer.get_latest()
        if quote[1] == 0.0:  # No quote yet
            return

        timestamp, bp, bs, ap, ask_s = quote

        # Calculate instant microstructure features
        obi = compute_order_book_imbalance(bs, ask_s)
        microprice = compute_microprice(bp, bs, ap, ask_s)
        spread = compute_spread(bp, ap)

        # Publish Event
        feature_event = {
            "timestamp": timestamp,
            "microprice": microprice,
            "obi": obi,
            "spread": spread,
            "mid_price": (bp + ap) / 2.0
        }
        self.on_feature_update(feature_event)

    def run(self):
        """스트림 구독 시작 (블로킹 루프)"""
        logger.info(f"Starting Websocket stream for {self.symbol}...")
        self.stream.subscribe_trades(self._trade_handler, self.symbol)
        self.stream.subscribe_quotes(self._quote_handler, self.symbol)

        # This will block forever
        self.stream.run()
