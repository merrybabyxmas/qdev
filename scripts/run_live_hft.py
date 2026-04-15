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

    api_key = os.environ.get("BROKER_API_KEY", "mock")
    secret_key = os.environ.get("BROKER_SECRET_KEY", "mock")
    enable_live = api_key != "mock"

    logger.info(f"Initializing Live HFT Engine for {symbols} | live_stream={enable_live}...")

    # Order execution is always simulated (is_simulation=True) — no real orders placed.
    # Data feed uses real Alpaca WebSocket when credentials are available.
    simulated_broker = HFTMatchingEngine(latency_ms=10.0, fee_bps=0.0001)

    live_engine = LiveTradingEngine(
        symbols=symbols,
        broker_engine=simulated_broker,
        is_simulation=True,
        api_key=api_key,
        secret_key=secret_key,
        enable_live_stream=enable_live,
    )

    try:
        live_engine.start()
    except KeyboardInterrupt:
        logger.info("Live engine stopped by user.")

if __name__ == "__main__":
    main()
