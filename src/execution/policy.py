import numpy as np
from src.utils.logger import logger

class ExecutionTracker:
    """
    HFT_EXEC_002: Cancel/Replace Threshold Controller
    주문을 추적하며:
    1. 시장 호가(Drift)가 멀어질 때 취소/재주문 (Cancel/Replace)
    2. 일정 시간이 경과한 Stale Order 삭제 (max_order_age_ms 기반 취소)
    """
    def __init__(self, broker_or_engine, cancel_threshold_bps: float = 2.0, max_order_age_ms: float = 5000.0):
        self.engine = broker_or_engine
        self.cancel_threshold_bps = cancel_threshold_bps / 10000.0
        self.max_order_age_ms = max_order_age_ms

        # {order_id: {"symbol":, "side":, "price":, "size":, "placed_at":}}
        self.active_orders = {}

    def track_order(self, order_id: str, symbol: str, side: str, price: float, size: float, current_time_ms: float = 0.0):
        self.active_orders[order_id] = {
            "symbol": symbol,
            "side": side,
            "price": price,
            "size": size,
            "placed_at": current_time_ms
        }
        logger.debug(f"Tracking active order {order_id} ({side} {size}@{price})")

    def untrack_order(self, order_id: str):
        if order_id in self.active_orders:
            del self.active_orders[order_id]

    def evaluate_cancel_replace(self, current_time_ms: float, symbol: str, current_bid: float, current_ask: float):
        """
        1. max_order_age_ms 경과된 Stale 주문 취소
        2. 호가 변동(Drift) 감지 시 취소/재진입
        """
        orders_to_check = list(self.active_orders.items())

        for oid, order in orders_to_check:
            if order["symbol"] != symbol:
                continue

            # Stale order check (HFT_EXEC_002 추가 요구사항)
            order_age = current_time_ms - order["placed_at"]
            if order_age > self.max_order_age_ms:
                logger.info(f"[Cancel/Replace] Stale order {oid} ({order_age}ms old). Canceling.")
                # Simulator와 Real Broker API 시그니처 고려 (Simulator엔 current_time_ms 전달)
                try:
                    self.engine.cancel_order(oid, current_time_ms)
                except TypeError:
                    self.engine.cancel_order(oid)
                self.untrack_order(oid)
                continue

            side = order["side"]
            my_price = order["price"]

            if side == "buy" and current_bid > my_price:
                drift = (current_bid - my_price) / my_price
                if drift > self.cancel_threshold_bps:
                    logger.debug(f"[Cancel/Replace] Buy limit {my_price} is lagging. Drift={drift:.4%}. Canceling and replacing.")
                    try:
                        self.engine.cancel_order(oid, current_time_ms)
                    except TypeError:
                        self.engine.cancel_order(oid)

                    self.untrack_order(oid)

                    try:
                        new_oid = self.engine.place_limit_order(symbol, "buy", current_bid, order["size"], current_time_ms)
                    except TypeError:
                        new_oid = self.engine.place_limit_order(symbol, "buy", current_bid, order["size"])

                    if new_oid:
                        self.track_order(new_oid, symbol, "buy", current_bid, order["size"], current_time_ms)

            elif side == "sell" and current_ask < my_price:
                drift = (my_price - current_ask) / current_ask
                if drift > self.cancel_threshold_bps:
                    logger.debug(f"[Cancel/Replace] Sell limit {my_price} is lagging. Drift={drift:.4%}. Canceling and replacing.")
                    try:
                        self.engine.cancel_order(oid, current_time_ms)
                    except TypeError:
                        self.engine.cancel_order(oid)

                    self.untrack_order(oid)

                    try:
                        new_oid = self.engine.place_limit_order(symbol, "sell", current_ask, order["size"], current_time_ms)
                    except TypeError:
                        new_oid = self.engine.place_limit_order(symbol, "sell", current_ask, order["size"])

                    if new_oid:
                        self.track_order(new_oid, symbol, "sell", current_ask, order["size"], current_time_ms)
