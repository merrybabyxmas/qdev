import asyncio
import logging
import os
from src.live.engine import LiveTradingEngine
from src.backtest.matching_engine import HFTMatchingEngine
from src.brokers.alpaca_broker import AlpacaBroker
from src.utils.logger import logger

logging.getLogger("src").setLevel(logging.INFO)

def main():
    symbols = ["BTC/USD", "ETH/USD"]

    # 1. 모드 설정 (.env 또는 환경 변수)
    # IS_SIMULATION 이 False이면 실제 AlpacaBroker를 연결하여 실제 잔고 및 주문을 넣음
    is_simulation = os.getenv("IS_SIMULATION", "True").lower() == "true"

    logger.info(f"Initializing HFT Engine for {symbols} (Simulation: {is_simulation})...")

    if is_simulation:
        logger.info("Using HFTMatchingEngine (Simulated Matching)...")
        broker_engine = HFTMatchingEngine(latency_ms=10.0, fee_bps=0.0001)
    else:
        logger.info("Using Actual AlpacaBroker (Real Trading/Paper)...")
        api_key = os.getenv("BROKER_API_KEY", "YOUR_API_KEY")
        secret_key = os.getenv("BROKER_SECRET_KEY", "YOUR_SECRET_KEY")
        is_paper = os.getenv("BROKER_PAPER", "True").lower() == "true"

        broker_engine = AlpacaBroker(api_key, secret_key, paper=is_paper)
        broker_engine.connect()

    # 2. 엔진 인스턴스화
    live_engine = LiveTradingEngine(symbols=symbols, broker_engine=broker_engine, is_simulation=is_simulation)

    try:
        live_engine.start() # Blocks forever receiving websockets
    except KeyboardInterrupt:
        logger.info("Live engine stopped by user.")
    finally:
        if not is_simulation:
            broker_engine.disconnect()

if __name__ == "__main__":
    main()
