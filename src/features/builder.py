import pandas as pd
from src.utils.logger import logger

def build_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds technical indicators I2 (SMA, EMA, RSI, MACD) and Price bundle I1.
    """
    required_columns = {"open", "high", "low", "close", "volume"}
    if df.empty or not required_columns.issubset(df.columns):
        missing = sorted(required_columns - set(df.columns))
        logger.warning(f"DataFrame is empty or missing required columns: {missing}")
        return df

    df = df.copy()

    # Simple Moving Average
    df['SMA_20'] = df['close'].rolling(window=20).mean()

    # Exponential Moving Average
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # Returns
    df['return_1d'] = df['close'].pct_change()
    df['return_5d'] = df['close'].pct_change(5)

    # Volatility
    df['volatility_20d'] = df['return_1d'].rolling(window=20).std()

    # Drop NAs
    df = df.dropna()
    logger.info(f"Built technical features. Shape: {df.shape}")
    return df
