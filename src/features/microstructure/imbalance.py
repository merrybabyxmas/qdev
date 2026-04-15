import numpy as np

def compute_order_book_imbalance(bid_size: float, ask_size: float) -> float:
    total = bid_size + ask_size
    if total == 0:
        return 0.0
    return (bid_size - ask_size) / total

def compute_microprice(bid_price: float, bid_size: float, ask_price: float, ask_size: float) -> float:
    total = bid_size + ask_size
    if total == 0:
        return (bid_price + ask_price) / 2.0
    return (bid_price * ask_size + ask_price * bid_size) / total

def compute_spread(bid_price: float, ask_price: float) -> float:
    return ask_price - bid_price

def compute_trade_intensity(ticks: np.ndarray, window_ms: float = 1000.0) -> float:
    if len(ticks) == 0:
        return 0.0
    latest_time = ticks[-1, 0]
    cutoff_time = latest_time - window_ms
    recent_ticks = ticks[ticks[:, 0] >= cutoff_time]
    if len(recent_ticks) == 0:
        return 0.0
    return np.sum(recent_ticks[:, 2])

def compute_toxicity_vpin_proxy(ticks: np.ndarray, window_ms: float = 1000.0) -> float:
    """
    간이 VPIN (Volume-Synchronized Probability of Informed Trading) 대용치.
    최근 윈도우 내의 매수 주도 거래량(Buy Volume)과 매도 주도 거래량(Sell Volume)의 불균형을 통해 독성(Toxicity) 측정.
    값이 1.0에 가까우면 한 방향(독성)으로의 일방적 흐름.
    """
    if len(ticks) == 0:
        return 0.0
    latest_time = ticks[-1, 0]
    cutoff_time = latest_time - window_ms
    recent_ticks = ticks[ticks[:, 0] >= cutoff_time]

    if len(recent_ticks) == 0:
        return 0.0

    # ticks: [timestamp, price, size, side (1=buy, -1=sell)]
    buy_vol = np.sum(recent_ticks[recent_ticks[:, 3] > 0, 2])
    sell_vol = np.sum(recent_ticks[recent_ticks[:, 3] < 0, 2])

    total_vol = buy_vol + sell_vol
    if total_vol == 0:
        return 0.0

    return abs(buy_vol - sell_vol) / total_vol

def compute_volatility_burst(ticks: np.ndarray, window_ms: float = 1000.0) -> float:
    """단기 가격 변동성 (표준편차)"""
    if len(ticks) < 2:
        return 0.0
    latest_time = ticks[-1, 0]
    cutoff_time = latest_time - window_ms
    recent_ticks = ticks[ticks[:, 0] >= cutoff_time]

    if len(recent_ticks) < 2:
        return 0.0

    prices = recent_ticks[:, 1]
    return float(np.std(prices))
