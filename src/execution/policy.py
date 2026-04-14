import numpy as np
from src.utils.logger import logger

class ExecutionTracker:
    """
    모든 미체결 주문(Active limit orders)을 추적하며,
    시장 호가(Bid/Ask)의 변동(Drift)을 모니터링하여 필요시 취소 및 수정 주문(Cancel/Replace)을 발생시킨다.
    """
    def __init__(self, broker_or_engine, cancel_threshold_bps: float = 2.0):
        self.engine = broker_or_engine # 실거래 BrokerInterface 나 HFTMatchingEngine 주입
        # 취소/수정 허용 오차: 만약 시장 Best Bid가 내가 건 주문보다 이 BPS 이상 도망가면 쫓아간다.
        self.cancel_threshold_bps = cancel_threshold_bps / 10000.0

        # Tracking memory: {order_id: {"symbol":, "side":, "price":, "size":}}
        self.active_orders = {}

    def track_order(self, order_id: str, symbol: str, side: str, price: float, size: float):
        self.active_orders[order_id] = {
            "symbol": symbol,
            "side": side,
            "price": price,
            "size": size
        }
        logger.debug(f"Tracking active order {order_id} ({side} {size}@{price})")

    def untrack_order(self, order_id: str):
        if order_id in self.active_orders:
            del self.active_orders[order_id]

    def evaluate_cancel_replace(self, current_time_ms: float, symbol: str, current_bid: float, current_ask: float):
        """
        매 틱마다 시장 BBO와 내 주문의 거리를 비교하여, 지정된 임계값을 넘으면
        빠르게 주문을 Cancel하고 새로운 가격에 Replace한다 (마켓 메이킹, 유동성 제공 전략에 필수).
        """
        # Dictionary changed size during iteration 방지를 위해 list 변환
        orders_to_check = list(self.active_orders.items())

        for oid, order in orders_to_check:
            if order["symbol"] != symbol:
                continue

            side = order["side"]
            my_price = order["price"]

            # 매수 주문의 경우: 시장 Bid 가격이 내 주문 가격보다 임계치 이상 높아졌다면 (매수 기회 상실 우려)
            if side == "buy" and current_bid > my_price:
                drift = (current_bid - my_price) / my_price
                if drift > self.cancel_threshold_bps:
                    logger.info(f"[Cancel/Replace] Buy limit {my_price} is lagging. Market bid is {current_bid}. Drift={drift:.4%}. Canceling {oid} and replacing.")

                    # 1. 기존 오더 취소
                    self.engine.cancel_order(oid, current_time_ms)
                    self.untrack_order(oid)

                    # 2. 새로운 오더를 현재 호가에 맞춰 재진입
                    new_oid = self.engine.place_limit_order(symbol, "buy", current_bid, order["size"], current_time_ms)
                    self.track_order(new_oid, symbol, "buy", current_bid, order["size"])

            # 매도 주문의 경우: 시장 Ask 가격이 내 주문 가격보다 임계치 이상 낮아졌다면 (매도 기회 상실 우려)
            elif side == "sell" and current_ask < my_price:
                drift = (my_price - current_ask) / current_ask
                if drift > self.cancel_threshold_bps:
                    logger.info(f"[Cancel/Replace] Sell limit {my_price} is lagging. Market ask is {current_ask}. Drift={drift:.4%}. Canceling {oid} and replacing.")

                    # 1. 기존 오더 취소
                    self.engine.cancel_order(oid, current_time_ms)
                    self.untrack_order(oid)

                    # 2. 새로운 오더를 현재 호가에 맞춰 재진입
                    new_oid = self.engine.place_limit_order(symbol, "sell", current_ask, order["size"], current_time_ms)
                    self.track_order(new_oid, symbol, "sell", current_ask, order["size"])
