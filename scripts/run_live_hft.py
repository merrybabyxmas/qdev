import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from src.live.engine import LiveTradingEngine
from src.backtest.matching_engine import HFTMatchingEngine
from src.utils.logger import logger

logging.getLogger("src").setLevel(logging.INFO)

# Load .env from project root
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

def main():
    symbols = ["BTC/USD", "ETH/USD"]

    # Mode setup (.env or environment variable)
    # IS_SIMULATION=False enables real AlpacaBroker with real/paper account
    is_simulation = os.getenv("IS_SIMULATION", "True").lower() == "true"

    api_key = os.environ.get("BROKER_API_KEY", "mock")
    secret_key = os.environ.get("BROKER_SECRET_KEY", "mock")
    enable_live = api_key != "mock"

    logger.info(f"Initializing HFT Engine for {symbols} | simulation={is_simulation} | live_stream={enable_live}...")

    if is_simulation:
        logger.info("Using HFTMatchingEngine (Simulated Matching)...")
        broker_engine = HFTMatchingEngine(latency_ms=10.0, fee_bps=0.0001)
    else:
        logger.info("Using Actual AlpacaBroker (Real Trading/Paper)...")
        try:
            from src.brokers.alpaca_broker import AlpacaBroker
            is_paper = os.getenv("BROKER_PAPER", "True").lower() == "true"
            broker_engine = AlpacaBroker(api_key, secret_key, paper=is_paper)
            broker_engine.connect()
        except ImportError:
            logger.warning("AlpacaBroker not available, falling back to HFTMatchingEngine")
            broker_engine = HFTMatchingEngine(latency_ms=10.0, fee_bps=0.0001)

    live_engine = LiveTradingEngine(
        symbols=symbols,
        broker_engine=broker_engine,
        is_simulation=is_simulation,
        api_key=api_key,
        secret_key=secret_key,
        enable_live_stream=enable_live,
    )

    try:
        live_engine.start()
    except KeyboardInterrupt:
        logger.info("Live engine stopped by user.")
    finally:
        if not is_simulation and hasattr(broker_engine, "disconnect"):
            broker_engine.disconnect()

if __name__ == "__main__":
    main()
