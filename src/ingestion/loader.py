import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from src.utils.logger import logger

def fetch_data_alpaca(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch OHLCV data using alpaca-py. We use Crypto as it often doesn't require strict API keys for basic historical data on Alpaca, making it reliable for testing."""
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
        return df

    except Exception as e:
        logger.error(f"Error fetching data from Alpaca: {e}")
        return pd.DataFrame()
