from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from src.features.builder import build_technical_features
from src.ingestion.loader import fetch_data_alpaca
from src.utils.logger import logger

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = ROOT / "artifacts" / "experiments" / "datasets"
DEFAULT_SYMBOLS = ("BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "DOGE/USD")


def _stable_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return sha256(encoded).hexdigest()[:12]


def _normalize_symbol(symbol: str) -> str:
    return symbol.replace("/", "_").replace(" ", "_")


def _normalize_sort_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """
    sort_values 전에 date/symbol 정렬 키 dtype을 안정적으로 맞춘다.
    unordered categorical로 인한 pandas sort 오류를 방지하기 위한 정규화 단계.
    """
    frame = frame.copy()

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].astype("string")

    return frame


def _add_symbol_rollups(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    group["date"] = pd.to_datetime(group["date"], errors="coerce")
    group = group.dropna(subset=["date"])
    group = group.sort_values("date", kind="mergesort").copy()

    group["return_mean_20d"] = group["return_1d"].rolling(20, min_periods=5).mean()
    group["return_std_20d"] = group["return_1d"].rolling(20, min_periods=5).std()
    group["volume_mean_20d"] = group["volume"].rolling(20, min_periods=5).mean()
    group["volume_std_20d"] = group["volume"].rolling(20, min_periods=5).std()
    group["corr_to_market_20d"] = group["return_1d"].rolling(20, min_periods=5).corr(group["market_return_1d"])
    group["tail_risk_20d"] = group["return_1d"].rolling(20, min_periods=5).quantile(0.05)
    group["market_beta_20d"] = group["return_1d"].rolling(20, min_periods=5).cov(group["market_return_1d"])
    return group


def _add_cross_section_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _normalize_sort_keys(frame)
    frame = frame.dropna(subset=["date"])
    frame = frame.sort_values(["date", "symbol"], kind="mergesort").copy()

    by_date = frame.groupby("date", sort=True)

    frame["market_return_1d"] = by_date["return_1d"].transform("mean")
    frame["market_return_5d"] = by_date["return_5d"].transform("mean")
    frame["market_volatility_20d"] = by_date["volatility_20d"].transform("mean")
    frame["market_dispersion_1d"] = by_date["return_1d"].transform("std").fillna(0.0)
    frame["relative_return_1d"] = frame["return_1d"] - frame["market_return_1d"]
    frame["relative_return_5d"] = frame["return_5d"] - frame["market_return_5d"]
    frame["return_rank"] = by_date["return_1d"].rank(pct=True, method="average")
    frame["vol_rank"] = by_date["volatility_20d"].rank(pct=True, method="average")
    frame["volume_rank"] = by_date["volume"].rank(pct=True, method="average")
    frame["inverse_vol"] = 1.0 / (frame["volatility_20d"].abs() + 1e-8)
    frame["shock_score"] = frame["relative_return_1d"].abs() / (frame["volatility_20d"].abs() + 1e-8)
    frame["jump_flag"] = (frame["shock_score"] > 2.5).astype(float)
    frame["momentum_proxy"] = 0.5 * frame["return_5d"] + 0.5 * frame["relative_return_5d"]
    frame["factor_proxy"] = 0.5 * frame["return_rank"] + 0.5 * (1.0 - frame["vol_rank"])

    frame = (
        frame.groupby("symbol", sort=False)
        .apply(_add_symbol_rollups, include_groups=False)
        .reset_index(level=0)
    )

    frame = _normalize_sort_keys(frame)
    frame = frame.dropna(subset=["date"])
    return frame.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)


def _add_symbol_lags(frame: pd.DataFrame, lags: Iterable[int] = (1, 5)) -> pd.DataFrame:
    frame = _normalize_sort_keys(frame)
    frame = frame.dropna(subset=["date"])
    frame = frame.sort_values(["symbol", "date"], kind="mergesort").copy()

    base_cols = [
        "return_1d",
        "return_5d",
        "volatility_20d",
        "market_return_1d",
        "relative_return_1d",
        "momentum_proxy",
        "shock_score",
        "factor_proxy",
        "corr_to_market_20d",
        "tail_risk_20d",
    ]
    grouped = frame.groupby("symbol", sort=False)
    for lag in lags:
        for column in base_cols:
            frame[f"{column}_lag_{lag}"] = grouped[column].shift(lag)

    frame = _normalize_sort_keys(frame)
    frame = frame.dropna(subset=["date"])
    return frame.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)


@dataclass(frozen=True)
class DatasetSpec:
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    start_date: str = "2025-01-01"
    end_date: str = "2026-04-13"
    target_horizon: int = 1
    feature_version: str = "technical_cross_section_v1"
    source: str = "alpaca_or_synthetic"
    train_fraction: float = 0.6
    validation_fraction: float = 0.2

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbols", tuple(self.symbols))

    def fingerprint(self) -> str:
        payload = asdict(self)
        payload["symbols"] = list(self.symbols)
        return _stable_hash(payload)


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_dates: list[pd.Timestamp] = field(default_factory=list)
    validation_dates: list[pd.Timestamp] = field(default_factory=list)
    test_dates: list[pd.Timestamp] = field(default_factory=list)


@dataclass
class DatasetBundle:
    spec: DatasetSpec
    version: str
    frame: pd.DataFrame
    root: Path
    manifest: dict[str, object] = field(default_factory=dict)

    def save(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        frame = self.frame.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        frame.to_csv(self.root / "panel.csv", index=False)

        manifest = self.manifest or self._build_manifest()
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return self.root

    def _build_manifest(self) -> dict[str, object]:
        frame = self.frame
        dates = pd.to_datetime(frame["date"])
        source_counts = frame["data_source"].value_counts(dropna=False).to_dict() if "data_source" in frame.columns else {}
        return {
            "version": self.version,
            "spec": asdict(self.spec),
            "symbol_count": int(frame["symbol"].nunique()),
            "row_count": int(len(frame)),
            "date_min": dates.min().isoformat() if not dates.empty else None,
            "date_max": dates.max().isoformat() if not dates.empty else None,
            "columns": list(frame.columns),
            "source_counts": source_counts,
        }

    @classmethod
    def load(cls, root: Path) -> "DatasetBundle":
        manifest_path = root / "manifest.json"
        panel_path = root / "panel.csv"
        if not manifest_path.exists() or not panel_path.exists():
            raise FileNotFoundError(f"Dataset bundle is incomplete at {root}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(panel_path, parse_dates=["date"])
        spec = DatasetSpec(**manifest["spec"])
        return cls(spec=spec, version=str(manifest["version"]), frame=frame, root=root, manifest=manifest)

    def split(self) -> DatasetSplit:
        dates = sorted(pd.to_datetime(self.frame["date"]).dropna().unique())
        if len(dates) < 5:
            raise ValueError("Not enough dates to build a train/validation/test split")

        train_end = max(1, int(len(dates) * self.spec.train_fraction))
        validation_end = max(train_end + 1, int(len(dates) * (self.spec.train_fraction + self.spec.validation_fraction)))
        validation_end = min(validation_end, len(dates) - 1)

        train_dates = list(dates[:train_end])
        validation_dates = list(dates[train_end:validation_end])
        test_dates = list(dates[validation_end:])
        if not validation_dates:
            validation_dates = [dates[min(train_end, len(dates) - 1)]]
        if not test_dates:
            test_dates = [dates[-1]]

        frame = self.frame.copy()
        train = frame[frame["date"].isin(train_dates)].copy()
        validation = frame[frame["date"].isin(validation_dates)].copy()
        test = frame[frame["date"].isin(test_dates)].copy()

        return DatasetSplit(train=train, validation=validation, test=test, train_dates=train_dates, validation_dates=validation_dates, test_dates=test_dates)


def _build_symbol_frame(
    symbol: str,
    start_date: str,
    end_date: str,
    target_horizon: int,
    data_loader: Callable[[str, str, str], pd.DataFrame],
) -> pd.DataFrame:
    raw = data_loader(symbol, start_date, end_date)
    if raw.empty:
        raise ValueError(f"No data returned for {symbol}")

    raw = raw.copy()
    data_source = raw.attrs.get("data_source", "unknown")
    features = build_technical_features(raw)
    if features.empty:
        raise ValueError(f"Feature builder returned empty frame for {symbol}")

    frame = features.copy()
    frame["date"] = pd.to_datetime(frame.index)
    frame["symbol"] = symbol
    frame["data_source"] = data_source
    frame["target_return"] = frame["return_1d"].shift(-target_horizon)
    frame = frame.reset_index(drop=True)
    return frame


def _assemble_panel(
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    target_horizon: int,
    data_loader: Callable[[str, str, str], pd.DataFrame],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        frame = _build_symbol_frame(symbol, start_date, end_date, target_horizon, data_loader)
        frames.append(frame)

    panel = pd.concat(frames, axis=0, ignore_index=True)
    panel = _add_cross_section_features(panel)
    panel = _add_symbol_lags(panel)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(subset=["target_return"]).reset_index(drop=True)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel


def build_dataset_bundle(
    spec: DatasetSpec,
    *,
    cache_root: Path | None = None,
    data_loader: Callable[[str, str, str], pd.DataFrame] = fetch_data_alpaca,
    refresh: bool = False,
) -> DatasetBundle:
    cache_root = cache_root or DEFAULT_DATASET_ROOT
    cache_root = Path(cache_root)
    bundle_root = cache_root / spec.fingerprint()

    if bundle_root.exists() and not refresh:
        logger.info("Loading cached dataset bundle", root=str(bundle_root))
        return DatasetBundle.load(bundle_root)

    logger.info(
        "Building dataset bundle",
        symbols=list(spec.symbols),
        start_date=spec.start_date,
        end_date=spec.end_date,
        target_horizon=spec.target_horizon,
        root=str(bundle_root),
    )
    frame = _assemble_panel(spec.symbols, spec.start_date, spec.end_date, spec.target_horizon, data_loader)
    bundle = DatasetBundle(spec=spec, version=spec.fingerprint(), frame=frame, root=bundle_root)
    bundle.manifest = bundle._build_manifest()
    bundle.save()
    return bundle