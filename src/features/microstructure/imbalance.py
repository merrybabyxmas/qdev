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
    if len(ticks) == 0:
        return 0.0
    latest_time = ticks[-1, 0]
    cutoff_time = latest_time - window_ms
    recent_ticks = ticks[ticks[:, 0] >= cutoff_time]

    if len(recent_ticks) == 0:
        return 0.0

    buy_vol = np.sum(recent_ticks[recent_ticks[:, 3] > 0, 2])
    sell_vol = np.sum(recent_ticks[recent_ticks[:, 3] < 0, 2])

    total_vol = buy_vol + sell_vol
    if total_vol == 0:
        return 0.0

    return abs(buy_vol - sell_vol) / total_vol

def compute_volatility_burst(ticks: np.ndarray, window_ms: float = 1000.0) -> float:
    if len(ticks) < 2:
        return 0.0
    latest_time = ticks[-1, 0]
    cutoff_time = latest_time - window_ms
    recent_ticks = ticks[ticks[:, 0] >= cutoff_time]

    if len(recent_ticks) < 2:
        return 0.0

    prices = recent_ticks[:, 1]
    return float(np.std(prices))

def compute_jump_proxy(ticks: np.ndarray, window_ms: float = 1000.0, jump_threshold: float = 3.0) -> float:
    """
    HFT_SDE_002: Jump-Risk Filter (Crash/Burst proxy).
    최근 window_ms 내 가격의 최대 변화량이 해당 기간의 평균 변동성의 N배(jump_threshold)를 초과하는지 여부(Jump Severity)를 계산.
    0.0이면 점프 없음, 값이 클수록 심각한 꼬리(tail) 위험 점프가 발생했음을 의미.
    """
    if len(ticks) < 3:
        return 0.0

    latest_time = ticks[-1, 0]
    cutoff_time = latest_time - window_ms
    recent_ticks = ticks[ticks[:, 0] >= cutoff_time]

    if len(recent_ticks) < 3:
        return 0.0

    prices = recent_ticks[:, 1]
    returns = np.diff(prices) / prices[:-1]

    if len(returns) < 2:
        return 0.0

    std_ret = np.std(returns)
    if std_ret == 0:
        return 0.0

    max_abs_ret = np.max(np.abs(returns))

    # Z-score of the largest move
    z_score = max_abs_ret / std_ret
    if z_score > jump_threshold:
        # Returns a severity magnitude above the threshold
        return float(z_score - jump_threshold)

    return 0.0
