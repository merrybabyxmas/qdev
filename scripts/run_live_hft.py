import asyncio
import logging
from src.live.engine import LiveTradingEngine
from src.backtest.matching_engine import HFTMatchingEngine
from src.utils.logger import logger

# For demonstration and local testing, we use the simulated matching engine
# to avoid actual live orders but run the exact same asyncio stream logic.
logging.getLogger("src").setLevel(logging.INFO)

def main():
    symbols = ["BTC/USD", "ETH/USD"]

    logger.info(f"Initializing Live HFT Engine for {symbols}...")

    # In production: broker_engine = AlpacaBroker()
    # Here, we use HFTMatchingEngine to simulate order fills against the live stream
    simulated_broker = HFTMatchingEngine(latency_ms=10.0, fee_bps=0.0001)

    live_engine = LiveTradingEngine(symbols=symbols, broker_engine=simulated_broker, is_simulation=True)

    try:
        live_engine.start() # Blocks forever receiving websockets
    except KeyboardInterrupt:
        logger.info("Live engine stopped by user.")

if __name__ == "__main__":
    main()
