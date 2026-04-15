import numpy as np
from src.utils.logger import logger

class AvellanedaStoikovMarketMaker:
    """
    HFT_SDE_001: Avellaneda-Stoikov (AS) Model
    Inventory-Aware Quote Skew Controller.

    이 모델은 단순히 Midprice 주변으로 대칭적인 Spread를 제시하는 것이 아니라,
    Market Maker의 재고(Inventory) 상태와 시장 변동성(Volatility)을 고려해
    예약 가격(Reservation Price)을 계산하고 최적의 비대칭 Spread를 계산합니다.
    """
    def __init__(self, risk_aversion: float = 0.1, time_horizon: float = 1.0,
                 order_arrival_intensity: float = 1.5, order_density: float = 1.5):
        """
        :param risk_aversion: 감마(gamma), 리스크 회피 성향
        :param time_horizon: T, 남은 거래 세션 시간 (통상 1로 정규화)
        :param order_arrival_intensity: A, 시장의 기준 주문 도달 강도
        :param order_density: k, 호가 깊이 파라미터 (주문이 호가창 깊숙이 갈수록 체결확률이 지수적으로 감소하는 정도)
        """
        self.gamma = risk_aversion
        self.T = time_horizon
        self.A = order_arrival_intensity
        self.k = order_density

    def calculate_quotes(self, mid_price: float, inventory: float, volatility: float, current_time: float) -> tuple:
        """
        최적의 Bid, Ask 가격을 계산합니다.

        :param mid_price: 현재 시장 중간가 (Midprice)
        :param inventory: 현재 보유 재고 (Inventory q)
        :param volatility: 단기 변동성 (Sigma sigma)
        :param current_time: 현재 시간 (t)
        :return: (optimal_bid, optimal_ask)
        """
        # 시간 경과 반영
        time_left = max(0.0001, self.T - current_time)

        # 1. Reservation Price 계산 (재고 편향 반영)
        # q > 0 이면 (롱 포지션), r < mid_price 가 되어 Ask를 낮춰 재고를 털어내려 함
        # q < 0 이면 (숏 포지션), r > mid_price 가 되어 Bid를 높여 재고를 확보하려 함
        reservation_price = mid_price - (inventory * self.gamma * (volatility ** 2) * time_left)

        # 2. Optimal Spread 계산 (재고 위험 프리미엄 반영)
        # 변동성이 높고 남은 시간이 많을수록 스프레드가 벌어짐
        spread = self.gamma * (volatility ** 2) * time_left + (2 / self.gamma) * np.log(1 + (self.gamma / self.k))

        # 3. 최적 호가 계산
        optimal_bid = reservation_price - (spread / 2)
        optimal_ask = reservation_price + (spread / 2)

        logger.debug(f"AS Model | Mid: {mid_price:.2f} | Inv: {inventory} | Vol: {volatility:.4f} -> Res: {reservation_price:.2f}, Bid: {optimal_bid:.2f}, Ask: {optimal_ask:.2f}")

        return optimal_bid, optimal_ask
