import numpy as np

class TickRingBuffer:
    """
    고성능(In-memory) O(1) 틱 데이터 관리를 위한 Ring Buffer.
    Pandas의 빈번한 append()를 피하고 Numpy 배열 기반으로 고정 길이의 최근 상태를 메모리에 유지.
    """
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        # [timestamp, price, size, side (1=buy, -1=sell)]
        self.buffer = np.zeros((capacity, 4), dtype=np.float64)
        self.index = 0
        self.size = 0

    def append(self, timestamp: float, price: float, volume: float, side: int):
        self.buffer[self.index] = [timestamp, price, volume, float(side)]
        self.index = (self.index + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def get_recent(self, count: int) -> np.ndarray:
        """가장 최근에 추가된 count 개의 데이터를 시간순으로 반환"""
        if count > self.size:
            count = self.size
        if count == 0:
            return np.empty((0, 4))

        start_idx = (self.index - count) % self.capacity
        if start_idx < self.index:
            return self.buffer[start_idx:self.index]
        else:
            return np.vstack((self.buffer[start_idx:], self.buffer[:self.index]))

class QuoteRingBuffer:
    """
    Top-of-Book(BBA) 호가 관리를 위한 버퍼.
    """
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        # [timestamp, bid_price, bid_size, ask_price, ask_size]
        self.buffer = np.zeros((capacity, 5), dtype=np.float64)
        self.index = 0
        self.size = 0

    def append(self, timestamp: float, bid: float, bid_size: float, ask: float, ask_size: float):
        self.buffer[self.index] = [timestamp, bid, bid_size, ask, ask_size]
        self.index = (self.index + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def get_latest(self) -> np.ndarray:
        if self.size == 0:
            return np.zeros(5)
        idx = (self.index - 1) % self.capacity
        return self.buffer[idx]
