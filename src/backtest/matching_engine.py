import numpy as np
from src.utils.logger import logger

class HFTMatchingEngine:
    """
    이벤트 기반 호가창 및 체결 모의 시뮬레이터 (Queue-Aware Simulator)
    1틱 단위에서의 패시브 지정가 주문(Limit Order)의 체결 가능성을 근사적으로 계산.
    """
    def __init__(self, latency_ms: float = 10.0, fee_bps: float = 0.0001):
        self.latency_ms = latency_ms
        self.fee_bps = fee_bps
        self.active_orders = {}
        self.pnl = 0.0
        self.inventory = 0.0
        self.cash = 100000.0
        self.order_counter = 0

    def place_limit_order(self, symbol: str, side: str, price: float, size: float, current_time_ms: float) -> str:
        """
        지정가 주문 제출.
        통신 지연(latency) 반영: 현재 시간 + 지연 시간 이후부터 큐(대기열)에 합류한다고 가정.
        """
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        if price <= 0:
            raise ValueError("price must be positive")
        if size <= 0:
            raise ValueError("size must be positive")

        self.order_counter += 1
        order_id = f"O-{self.order_counter}"

        arrival_time = current_time_ms + self.latency_ms

        self.active_orders[order_id] = {
            "symbol": symbol,
            "side": side,
            "price": price,
            "size": size,
            "arrival_time": arrival_time,
            "queue_ahead": 0.0,  # Approximate volume ahead of us
            "filled": 0.0
        }

        logger.debug(f"Order Placed: {order_id} {side} {size} @ {price}, Arrives @ {arrival_time}")
        return order_id

    def cancel_order(self, order_id: str, current_time_ms: float):
        if order_id in self.active_orders:
            cancel_time = current_time_ms + self.latency_ms
            logger.debug(f"Order Cancelled: {order_id} @ {cancel_time}")
            del self.active_orders[order_id]

    def process_quote_update(self, current_time_ms: float, bid: float, ask: float):
        """
        호가창 업데이트 시 즉각적인 체결 평가 (Cross 매칭).
        내가 매수 지정가를 걸었는데 현재 시장 매도 호가(Ask)가 내 가격 이하로 내려오면 즉시 체결.
        """
        filled_orders = []
        for oid, order in self.active_orders.items():
            if current_time_ms < order["arrival_time"]:
                continue

            side = order["side"]
            price = order["price"]
            size = order["size"] - order["filled"]

            fill_price = 0.0
            is_filled = False

            if side == "buy" and ask <= price and ask > 0:
                is_filled = True
                fill_price = ask
            elif side == "sell" and bid >= price and bid > 0:
                is_filled = True
                fill_price = bid

            if is_filled:
                fee = fill_price * size * self.fee_bps
                if side == "buy":
                    self.inventory += size
                    self.cash -= (fill_price * size + fee)
                else:
                    self.inventory -= size
                    self.cash += (fill_price * size - fee)

                order["filled"] += size
                filled_orders.append(oid)
                logger.debug(f"Order Filled: {oid} {side} {size} @ {fill_price}")

        for oid in filled_orders:
            del self.active_orders[oid]

    def get_portfolio_value(self, current_mid: float) -> float:
        return self.cash + (self.inventory * current_mid)

    def get_account(self) -> dict:
        return {"equity": self.cash, "cash": self.cash, "buying_power": self.cash}

    def get_positions(self) -> dict:
        return {}  # single-instrument inventory tracked via self.inventory
