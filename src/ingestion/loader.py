import hashlib
from datetime import datetime

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency guard
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame
except Exception:  # pragma: no cover - optional dependency guard
    CryptoHistoricalDataClient = None
    CryptoBarsRequest = None
    TimeFrame = None

from src.utils.logger import logger


def _build_synthetic_ohlcv(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Deterministic fallback path for offline testing."""
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
    except Exception as exc:
        logger.error(f"Invalid date range for synthetic fallback: {exc}")
        return pd.DataFrame()

    if end < start:
        logger.warning(f"End date precedes start date for {symbol}; returning empty frame.")
        return pd.DataFrame()

    dates = pd.date_range(start=start, end=end, freq="D")
    if dates.empty:
        return pd.DataFrame()

    seed_bytes = hashlib.sha256(f"{symbol}|{start_date}|{end_date}".encode("utf-8")).digest()
    seed = int.from_bytes(seed_bytes[:8], "big", signed=False) % (2**32 - 1)
    rng = np.random.default_rng(seed)

    base_price = 100.0 + rng.uniform(-20.0, 20.0)
    daily_returns = rng.normal(0.0008, 0.02, size=len(dates))
    close = base_price * np.exp(np.cumsum(daily_returns))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0.004, 0.0015, size=len(dates)))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.uniform(1_000, 10_000, size=len(dates))

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    df.index.name = "date"
    df.attrs["data_source"] = "synthetic"
    logger.warning(f"Using synthetic fallback data for {symbol} ({len(df)} rows).")
    return df


def fetch_data_alpaca(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch OHLCV data using alpaca-py. We use Crypto as it often doesn't require strict API keys for basic historical data on Alpaca, making it reliable for testing."""
    if CryptoHistoricalDataClient is None or CryptoBarsRequest is None or TimeFrame is None:
        logger.warning("alpaca-py is unavailable; falling back to synthetic data.")
        return _build_synthetic_ohlcv(symbol, start_date, end_date)

    try:
        client = CryptoHistoricalDataClient()
        request_params = CryptoBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=datetime.strptime(start_date, "%Y-%m-%d"),
            end=datetime.strptime(end_date, "%Y-%m-%d")
        )

        bars = client.get_crypto_bars(request_params)
        df = bars.df

        if df.empty:
            logger.warning(f"No data fetched for {symbol}")
            return df

        # Ensure standard OHLCV format
        # Alpaca crypto df index is usually multi-level: (symbol, timestamp)
        df = df.reset_index()
        df = df.rename(columns={'timestamp': 'date'})
        df = df.set_index('date')
        df = df[['open', 'high', 'low', 'close', 'volume']]

        logger.info(f"Successfully fetched {len(df)} rows for {symbol} from Alpaca")
        df.attrs["data_source"] = "alpaca"
        return df

    except Exception as e:
        logger.warning(f"Error fetching data from Alpaca for {symbol}: {e}. Falling back to synthetic data.")
        df = _build_synthetic_ohlcv(symbol, start_date, end_date)
        df.attrs["data_source"] = "synthetic"
        return df
