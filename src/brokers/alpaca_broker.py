from typing import Dict, Any, List
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from src.brokers.base import BrokerInterface
from src.utils.logger import logger

class AlpacaBroker(BrokerInterface):
    """
    실제 Alpaca Trading API를 사용하는 브로커 어댑터.
    이 클래스를 통해 실제로 지정가/시장가 주문이 전송되고, 계좌 잔고를 실시간으로 조회합니다.
    """
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.client = None
        self.connected = False

    def connect(self):
        try:
            self.client = TradingClient(self.api_key, self.secret_key, paper=self.paper)
            acct = self.client.get_account()
            self.connected = True
            logger.info(f"Alpaca Broker connected successfully. Environment: {'Paper' if self.paper else 'Live'}. Buying Power: {acct.buying_power}")
        except Exception as e:
            self.connected = False
            logger.error(f"Failed to connect to Alpaca: {e}")
            raise e

    def disconnect(self):
        self.connected = False
        logger.info("Alpaca Broker disconnected.")

    def get_account(self) -> Dict[str, Any]:
        if not self.connected:
            raise ConnectionError("Broker not connected")
        acct = self.client.get_account()
        return {
            "balance": float(acct.cash),
            "equity": float(acct.equity),
            "buying_power": float(acct.buying_power),
            "initial_margin": float(acct.initial_margin)
        }

    def get_positions(self) -> Dict[str, Any]:
        if not self.connected:
            raise ConnectionError("Broker not connected")
        positions = self.client.get_all_positions()
        pos_dict = {}
        for p in positions:
            pos_dict[p.symbol] = float(p.qty)
        return pos_dict

    def get_latest_price(self, symbol: str) -> float:
        """
        REST API로 최신 가격 조회. 실시간 스트리밍 환경에서는 잘 안 쓰이지만 Fallback 용도로 사용.
        """
        # Note: In a pure HFT setup, latest price comes from the WS stream, not REST.
        # This is a basic mock return for the interface requirement.
        logger.warning("get_latest_price called via REST on Alpaca Broker. Use stream data instead.")
        return 0.0

    def place_order(self, order: Dict[str, Any]) -> str:
        """시장가 주문 (Market Order)"""
        if not self.connected:
            raise ConnectionError("Broker not connected")

        symbol = order.get("symbol")
        qty = order.get("qty")
        side = OrderSide.BUY if order.get("side").lower() == "buy" else OrderSide.SELL

        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.GTC
        )

        try:
            res = self.client.submit_order(req)
            logger.info(f"Placed MARKET order {res.id}: {side} {qty} {symbol}")
            return str(res.id)
        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            return ""

    def place_limit_order(self, symbol: str, side: str, price: float, size: float, current_time_ms: float = 0.0) -> str:
        """지정가 주문 (Limit Order) - HFT/Cancel-Replace에 주로 쓰임"""
        if not self.connected:
            raise ConnectionError("Broker not connected")

        alpaca_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        req = LimitOrderRequest(
            symbol=symbol,
            limit_price=round(price, 2), # Alpaca requires valid tick sizes
            qty=size,
            side=alpaca_side,
            time_in_force=TimeInForce.GTC
        )

        try:
            res = self.client.submit_order(req)
            logger.info(f"Placed LIMIT order {res.id}: {side} {size} {symbol} @ {price}")
            return str(res.id)
        except Exception as e:
            logger.error(f"Failed to place limit order: {e}")
            return ""

    def cancel_order(self, order_id: str, current_time_ms: float = 0.0):
        if not self.connected:
            return
        try:
            self.client.cancel_order_by_id(order_id)
            logger.info(f"Canceled order {order_id}")
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")

    def get_open_orders(self) -> List[Dict[str, Any]]:
        if not self.connected:
            return []
        orders = self.client.get_orders(status="open")
        return [{"order_id": str(o.id), "symbol": o.symbol, "side": o.side, "qty": float(o.qty), "price": float(o.limit_price) if o.limit_price else 0.0} for o in orders]

    def get_fills(self) -> List[Dict[str, Any]]:
        # Normally tracked via trade_updates websocket stream.
        # This REST fallback is omitted for brevity in HFT.
        return []

    def heartbeat(self) -> bool:
        if not self.connected or self.client is None:
            return False
        try:
            self.client.get_account()
            return True
        except Exception:
            return False

    def sync_state(self) -> Dict[str, Any]:
        if not self.connected:
            return {}
        try:
            return self.get_account()
        except Exception:
            return {}
