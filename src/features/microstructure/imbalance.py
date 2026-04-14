import numpy as np

def compute_order_book_imbalance(bid_size: float, ask_size: float) -> float:
    """
    Order Book Imbalance (OBI) 계산:
    (Bid Size - Ask Size) / (Bid Size + Ask Size)
    값이 +1에 가까우면 매수 잔량 우세, -1에 가까우면 매도 잔량 우세
    """
    total = bid_size + ask_size
    if total == 0:
        return 0.0
    return (bid_size - ask_size) / total

def compute_microprice(bid_price: float, bid_size: float, ask_price: float, ask_size: float) -> float:
    """
    Microprice 계산:
    잔량 불균형에 가중치를 둔 현재 예상 적정가 (Volume-weighted mid price)
    Bid 비중이 크면 Ask 방향으로, Ask 비중이 크면 Bid 방향으로 기울어짐
    """
    total = bid_size + ask_size
    if total == 0:
        return (bid_price + ask_price) / 2.0

    # Imbalance weighted mid
    return (bid_price * ask_size + ask_price * bid_size) / total

def compute_spread(bid_price: float, ask_price: float) -> float:
    """단순 명목 스프레드"""
    return ask_price - bid_price

def compute_trade_intensity(ticks: np.ndarray, window_ms: float = 1000.0) -> float:
    """
    최근 지정된 window 내의 거래량(Trade Intensity)를 누적.
    ticks: numpy array [timestamp, price, size, side]
    """
    if len(ticks) == 0:
        return 0.0

    latest_time = ticks[-1, 0]
    cutoff_time = latest_time - window_ms

    # 시간 필터링
    recent_ticks = ticks[ticks[:, 0] >= cutoff_time]

    if len(recent_ticks) == 0:
        return 0.0

    # 총 거래량 (크기)
    intensity = np.sum(recent_ticks[:, 2])
    return intensity
