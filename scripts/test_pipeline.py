from _bootstrap import ensure_project_root

ensure_project_root()

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.ingestion.loader import fetch_data_alpaca
from src.features.builder import build_technical_features
from src.models.hmm import SimpleHMMRegimeDetector
from src.models.lgbm import LightGBMRanker
from src.strategies.ml_strategy import MLStrategy
from src.risk.manager import RiskManager
from src.backtest.engine import BacktestEngine

def run():
    print("1. Fetching data and generating features...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    symbols = ["BTC/USD", "ETH/USD"]

    data_dict = {}
    for sym in symbols:
        df = fetch_data_alpaca(sym, start_date, end_date)
        if not df.empty:
            # Shift close to get target return for training
            df['target_return'] = df['close'].pct_change().shift(-1)
            df_feat = build_technical_features(df)
            data_dict[sym] = df_feat

    if not data_dict:
        print("No data fetched, stopping.")
        return

    # Combine data for model training
    train_df = pd.concat(data_dict.values()).dropna(subset=['target_return'])

    print("2. Fitting Models (HMM & LightGBM)...")
    hmm = SimpleHMMRegimeDetector()
    hmm.fit(train_df)

    lgbm = LightGBMRanker()
    lgbm.fit(train_df)

    print("3. Generating Signals & Target Weights...")
    strategy = MLStrategy(symbols)

    # We predict the signal on the latest available data point
    latest_preds = {}
    price_series_dict = {}

    for sym, df in data_dict.items():
        price_series_dict[sym] = df['close']
        latest_row = df.iloc[[-1]]

        regime = hmm.predict(latest_row)[0]
        # In a real pipeline, regime might gate the model prediction. Here we just print it.
        print(f"[{sym}] Detected Regime: {regime}")

        pred = lgbm.predict(latest_row)[0]
        latest_preds[sym] = pred

    target_weights = strategy.generate_weights(latest_preds)

    print("4. Applying Risk Management...")
    risk_manager = RiskManager(max_position_cap=0.40)
    capped_weights = risk_manager.apply_position_caps(target_weights)

    print("5. Running Multi-Asset Backtest...")
    price_df = pd.DataFrame(price_series_dict).dropna()

    # Forward-fill target weights over the history for a simple simulation
    weights_df = pd.DataFrame([capped_weights] * len(price_df), index=price_df.index)

    bt_engine = BacktestEngine()
    portfolio = bt_engine.run(price_df, weights_df)

    print("\n=== Backtest Results ===")
    print(portfolio.stats()[['Total Return [%]', 'Max Drawdown [%]', 'Sharpe Ratio']])

if __name__ == "__main__":
    run()
