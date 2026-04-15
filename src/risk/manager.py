from typing import Dict
from src.utils.logger import logger

class RiskManager:
    """
    Implements core risk policies focusing on real USD values and constraints.
    - Position Cap (single name max weight)
    - Drawdown Kill Switch (Based on real account equity)
    - Position Sizing (Target % to actual Order Qty)
    """
    def __init__(self, max_position_cap: float = 0.20, max_drawdown: float = 0.15, max_leverage: float = 1.0):
        self.max_position_cap = max_position_cap
        self.max_drawdown = max_drawdown
        self.max_leverage = max_leverage
        self.kill_switch_active = False

        # Real account high water mark tracking
        self.peak_equity = 0.0

    def evaluate_account_risk(self, current_equity: float) -> bool:
        """
        매 루프마다 실제 계좌 잔고를 기반으로 Drawdown 한도를 검사함.
        만약 손실이 너무 커지면 즉각 킬스위치 가동.
        """
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        current_drawdown = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0

        if current_drawdown >= self.max_drawdown:
            logger.error(f"HARD DRAWDOWN KILL SWITCH ACTIVATED: {current_drawdown:.2%} >= {self.max_drawdown:.2%}")
            self.kill_switch_active = True

        return self.kill_switch_active

    def check_drawdown(self, current_drawdown: float):
        if current_drawdown >= self.max_drawdown:
            logger.error(f"DRAWDOWN KILL SWITCH ACTIVATED: {current_drawdown:.2%} >= {self.max_drawdown:.2%}")
            self.kill_switch_active = True
        return self.kill_switch_active

    def apply_position_caps(self, target_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Caps individual positions at max_position_cap.
        목표 포지션 비중이 개별 한도를 초과하는지 검사.
        """
        if self.kill_switch_active:
            logger.warning("Kill switch is active. Forcing all weights to 0.")
            return {sym: 0.0 for sym in target_weights}

        capped_weights = {}
        for sym, weight in target_weights.items():
            if weight > self.max_position_cap:
                logger.debug(f"Capping {sym} weight from {weight:.2%} to {self.max_position_cap:.2%}")
                capped_weights[sym] = self.max_position_cap
            elif weight < -self.max_position_cap:
                # Assuming long-short system, cap absolute negative weights too
                capped_weights[sym] = -self.max_position_cap
            else:
                capped_weights[sym] = weight

        return capped_weights

    def pretrade_check(
        self,
        target_weights: Dict[str, float],
        current_drawdown: float = 0.0,
        stale_data: bool = False,
        max_gross_exposure: float = None,
    ) -> tuple:
        """Basic pre-trade gate for paper/live-safe execution."""
        reasons: list = []

        if stale_data:
            reasons.append("stale_data")

        if self.check_drawdown(current_drawdown):
            reasons.append("drawdown_kill_switch")

        gross_exposure = sum(abs(weight) for weight in target_weights.values())
        if max_gross_exposure is not None and gross_exposure > max_gross_exposure:
            reasons.append(f"gross_exposure>{max_gross_exposure}")

        oversized = [sym for sym, weight in target_weights.items() if abs(weight) > self.max_position_cap]
        if oversized:
            reasons.append(f"position_cap_exceeded:{','.join(sorted(oversized))}")

        if reasons:
            logger.warning(f"Pre-trade gate blocked execution: {reasons}")
            return False, reasons

        return True, reasons

    def calculate_order_qty(self, symbol: str, target_weight: float, current_qty: float, current_price: float, account_equity: float) -> float:
        """
        목표 비중(Weight)을 기반으로, 현재 내 실제 보유 잔고(Qty)를 빼고,
        Alpaca 브로커에 전송할 '신규 주문 필요 수량(Delta Qty)'을 계산.
        """
        if current_price <= 0:
            return 0.0

        if self.kill_switch_active:
            target_weight = 0.0

        target_value_usd = account_equity * target_weight
        target_qty = target_value_usd / current_price

        # 얼마나 사고팔아야 하는가
        delta_qty = target_qty - current_qty

        # HFT나 단기 트레이딩에서는 소수점 반올림(틱 사이즈)이 매우 중요함
        # Crypto의 경우 통상 소수점 4자리 허용
        delta_qty = round(delta_qty, 4)

        return delta_qty
