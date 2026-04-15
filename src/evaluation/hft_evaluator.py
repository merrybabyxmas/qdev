from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd


_HFT_TICKS_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "control_plane"
    / "logs"
    / "hft_ticks.jsonl"
)
_HFT_STATUS_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "control_plane"
    / "hft_status.json"
)

_PERIODS_PER_YEAR = 252 * 6.5 * 3600  # ~tick-level annualisation (rough; seconds in trading year)


def _load_ticks(path: Path, lookback: int = 10_000) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-lookback:]:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in ("price", "spread", "obi", "prediction_bps", "target_weight"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["timestamp", "price"]).reset_index(drop=True)


def _compute_symbol_metrics(df: pd.DataFrame, pred_col: str = "prediction_bps") -> dict:
    """Compute performance metrics for a single symbol's tick series."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    if n < 10:
        return {}

    # Tick-level price change (fractional)
    price_next = df["price"].shift(-1)
    price_change = (price_next - df["price"]) / df["price"]  # last row → NaN
    price_change_bps = price_change * 10_000

    tw = df.get("target_weight", pd.Series(0.0, index=df.index)).fillna(0.0)
    pred_bps = df[pred_col].fillna(0.0) if pred_col in df.columns else pd.Series(0.0, index=df.index)

    # Tick P&L as fraction of equity
    tick_pnl = (tw * price_change).dropna()

    # Hit rate: sign of prediction matches sign of actual price change
    valid = price_change_bps.dropna()
    pred_valid = pred_bps.loc[valid.index]
    hit = ((pred_valid > 0) & (valid > 0)) | ((pred_valid < 0) & (valid < 0))
    hit_rate = float(hit.mean()) if len(hit) > 0 else 0.0

    # MAE in bps
    mae_bps = float((pred_valid - valid).abs().mean()) if len(valid) > 0 else 0.0

    # Equity curve and drawdown
    equity = (1.0 + tick_pnl.fillna(0.0)).cumprod()
    total_return_pct = float(equity.iloc[-1] - 1.0) * 100.0 if not equity.empty else 0.0

    running_max = equity.cummax()
    drawdown = (equity / running_max - 1.0)
    max_dd_pct = float(drawdown.min() * 100.0) if not drawdown.empty else 0.0

    # Sharpe (tick-level, annualised crudely by sqrt of tick count per year)
    # Use seconds-based annualisation if timestamps available
    if n >= 2:
        total_seconds = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds()
        ticks_per_year = (n / max(total_seconds, 1)) * 252 * 6.5 * 3600
    else:
        ticks_per_year = _PERIODS_PER_YEAR

    mean_r = float(tick_pnl.mean()) if len(tick_pnl) > 0 else 0.0
    std_r = float(tick_pnl.std(ddof=1)) if len(tick_pnl) > 1 else 0.0
    sharpe = (mean_r / std_r * sqrt(ticks_per_year)) if std_r > 0 else 0.0
    sharpe = float(np.clip(sharpe, -10.0, 10.0))

    downside = tick_pnl[tick_pnl < 0]
    sortino_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = (mean_r / sortino_std * sqrt(ticks_per_year)) if sortino_std > 0 else 0.0
    sortino = float(np.clip(sortino, -10.0, 10.0))

    win_rate_pct = float((tick_pnl > 0).mean() * 100.0)
    avg_spread_bps = float(
        ((df.get("spread", pd.Series(0.0, index=df.index)).fillna(0.0))
         / df["price"].replace(0, np.nan) * 10_000).mean()
    ) if "spread" in df.columns else 0.0

    return {
        "tick_count": n,
        "total_return_pct": total_return_pct,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": max_dd_pct,
        "win_rate_pct": win_rate_pct,
        "hit_rate_pct": hit_rate * 100.0,
        "mae_bps": mae_bps,
        "avg_spread_bps": avg_spread_bps,
    }


def _summary_dict(metrics: dict) -> dict:
    """Build a test_summary-compatible sub-dict from HFT metrics."""
    return {
        "total_return_pct": metrics.get("total_return_pct", 0.0),
        "annualized_return_pct": metrics.get("total_return_pct", 0.0),  # proxy
        "annualized_vol_pct": 0.0,
        "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
        "sortino_ratio": metrics.get("sortino_ratio", 0.0),
        "max_drawdown_pct": metrics.get("max_drawdown_pct", 0.0),
        "win_rate_pct": metrics.get("win_rate_pct", 0.0),
        "avg_turnover": 0.0,
        "total_turnover": 0.0,
        "avg_gross_exposure": 0.0,
        "cost_drag_pct": 0.0,
        "active_days": 0,
        "final_equity": 1.0 + metrics.get("total_return_pct", 0.0) / 100.0,
    }


def _hft_final_score(metrics: dict) -> float:
    """
    HFT-specific composite score (0–100 scale, independent of macro leaderboard).
    Components:
      - hit_rate (0-100) × 0.40
      - sharpe clipped [-3, 3] rescaled to [0,100] × 0.35
      - return clipped [-5, 5] rescaled to [0,100] × 0.15
      - drawdown clipped [-10, 0] rescaled to [0,100] (higher=shallower) × 0.10
    """
    hit = float(np.clip(metrics.get("hit_rate_pct", 50.0), 0.0, 100.0))
    sharpe_raw = float(np.clip(metrics.get("sharpe_ratio", 0.0), -3.0, 3.0))
    sharpe_scaled = (sharpe_raw + 3.0) / 6.0 * 100.0
    ret_raw = float(np.clip(metrics.get("total_return_pct", 0.0), -5.0, 5.0))
    ret_scaled = (ret_raw + 5.0) / 10.0 * 100.0
    dd_raw = float(np.clip(metrics.get("max_drawdown_pct", 0.0), -10.0, 0.0))
    dd_scaled = (dd_raw + 10.0) / 10.0 * 100.0

    return round(
        0.40 * hit
        + 0.35 * sharpe_scaled
        + 0.15 * ret_scaled
        + 0.10 * dd_scaled,
        4,
    )


def build_hft_leaderboard_rows(
    ticks_path: Path | None = None,
    status_path: Path | None = None,
    lookback: int = 10_000,
) -> pd.DataFrame:
    """
    Read hft_ticks.jsonl and hft_status.json to produce leaderboard rows
    compatible with the existing leaderboard.csv schema (one row per symbol +
    one combined "ALL" row).

    Returns an empty DataFrame if insufficient tick data is available.
    """
    ticks_path = Path(ticks_path) if ticks_path else _HFT_TICKS_PATH
    status_path = Path(status_path) if status_path else _HFT_STATUS_PATH

    df = _load_ticks(ticks_path, lookback=lookback)
    if df.empty or len(df) < 20:
        return pd.DataFrame()

    total_updates = 0
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            total_updates = int(status.get("model", {}).get("total_updates", 0))
        except Exception:
            pass

    rows: list[dict] = []

    has_lgbm = "lgbm_prediction_bps" in df.columns

    symbols = df["symbol"].unique() if "symbol" in df.columns else ["ALL"]
    for sym in symbols:
        sym_df = df[df["symbol"] == sym] if "symbol" in df.columns else df

        for model_tag, pred_col, model_label in [
            ("SGD", "prediction_bps", "Online SGD"),
            *([("LGBM", "lgbm_prediction_bps", "Online LGBM")] if has_lgbm else []),
        ]:
            metrics = _compute_symbol_metrics(sym_df, pred_col=pred_col)
            if not metrics:
                continue

            summary = _summary_dict(metrics)
            final_score = _hft_final_score(metrics)

            row = {
                "pipeline_id": f"HFT_{model_tag}_{sym.replace('/', '_')}",
                "name": f"{model_label} — {sym}",
                "family": "HFT Microstructure",
                "doc_status": "live",
                "implementation_mode": "direct",
                "base_model": model_tag.lower() + "_online",
                "feature_profile": "microstructure_5axis",
                "allocation_mode": "hft_weight",
                "overlays": "[]",
                "risk_cap": 0.5,
                "max_drawdown": 0.15,
                "decision": "live",
                "feature_count": 5,
                "runtime_seconds": 0.0,
                "notes": (
                    f"Live {model_label} | {metrics['tick_count']:,} ticks"
                    + (f" | updates: {total_updates:,}" if model_tag == "SGD" else "")
                ),
                "source_doc": "live_engine",
                "status": "ok",
                "train_summary.total_return_pct": summary["total_return_pct"],
                "train_summary.annualized_return_pct": summary["annualized_return_pct"],
                "train_summary.annualized_vol_pct": 0.0,
                "train_summary.sharpe_ratio": summary["sharpe_ratio"],
                "train_summary.sortino_ratio": summary["sortino_ratio"],
                "train_summary.max_drawdown_pct": summary["max_drawdown_pct"],
                "train_summary.win_rate_pct": summary["win_rate_pct"],
                "train_summary.avg_turnover": 0.0,
                "train_summary.total_turnover": 0.0,
                "train_summary.avg_gross_exposure": 0.0,
                "train_summary.cost_drag_pct": 0.0,
                "train_summary.active_days": 0,
                "train_summary.final_equity": summary["final_equity"],
                "validation_summary.total_return_pct": summary["total_return_pct"],
                "validation_summary.annualized_return_pct": summary["annualized_return_pct"],
                "validation_summary.annualized_vol_pct": 0.0,
                "validation_summary.sharpe_ratio": summary["sharpe_ratio"],
                "validation_summary.sortino_ratio": summary["sortino_ratio"],
                "validation_summary.max_drawdown_pct": summary["max_drawdown_pct"],
                "validation_summary.win_rate_pct": summary["win_rate_pct"],
                "validation_summary.avg_turnover": 0.0,
                "validation_summary.total_turnover": 0.0,
                "validation_summary.avg_gross_exposure": 0.0,
                "validation_summary.cost_drag_pct": 0.0,
                "validation_summary.active_days": 0,
                "validation_summary.final_equity": summary["final_equity"],
                "test_summary.total_return_pct": summary["total_return_pct"],
                "test_summary.annualized_return_pct": summary["annualized_return_pct"],
                "test_summary.annualized_vol_pct": 0.0,
                "test_summary.sharpe_ratio": summary["sharpe_ratio"],
                "test_summary.sortino_ratio": summary["sortino_ratio"],
                "test_summary.max_drawdown_pct": summary["max_drawdown_pct"],
                "test_summary.win_rate_pct": summary["win_rate_pct"],
                "test_summary.avg_turnover": 0.0,
                "test_summary.total_turnover": 0.0,
                "test_summary.avg_gross_exposure": 0.0,
                "test_summary.cost_drag_pct": 0.0,
                "test_summary.active_days": 0,
                "test_summary.final_equity": summary["final_equity"],
                "hft.hit_rate_pct": metrics["hit_rate_pct"],
                "hft.mae_bps": metrics["mae_bps"],
                "hft.tick_count": metrics["tick_count"],
                "hft.avg_spread_bps": metrics["avg_spread_bps"],
                "final_score": final_score,
                "score_rank": 0,
                "promotion_candidate": False,
            }
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)
