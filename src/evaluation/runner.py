from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.evaluation.dataset import DatasetBundle, DatasetSplit
from src.evaluation.metrics import PerformanceSummary, classify_candidate, summarize_performance
from src.evaluation.registry import ExperimentSpec, get_feature_columns
from src.models.hmm import SimpleHMMRegimeDetector
from src.models.lgbm import LightGBMRanker
from src.models.linear import BayesianLinearReturnForecaster
from src.models.linear import LinearReturnForecaster
from src.risk.manager import RiskManager
from src.utils.logger import logger


def _normalize_positive(scores: pd.Series, mode: str) -> pd.Series:
    scores = scores.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    positive = scores[scores > 0.0]
    if scores.empty or positive.empty:
        return pd.Series(0.0, index=scores.index)

    if mode == "all_equal":
        return pd.Series(1.0 / len(scores), index=scores.index)
    if mode == "equal_positive":
        weights = pd.Series(0.0, index=scores.index)
        weights.loc[positive.index] = 1.0 / len(positive)
        return weights
    if mode == "proportional_positive":
        total = float(positive.sum())
        if total <= 0:
            return pd.Series(0.0, index=scores.index)
        weights = pd.Series(0.0, index=scores.index)
        weights.loc[positive.index] = positive / total
        return weights
    raise ValueError(f"Unsupported allocation mode: {mode}")


def _score_base_model(
    spec: ExperimentSpec,
    train: pd.DataFrame,
    frame: pd.DataFrame,
    feature_columns: list[str],
):
    if spec.base_model == "equal_weight":
        return pd.Series(1.0, index=frame.index, dtype=float), {}

    if spec.base_model == "random_walk":
        scores = frame.groupby("symbol", sort=False)["return_1d"].shift(1).fillna(0.0)
        return scores.astype(float), {}

    if spec.base_model == "linear":
        model = LinearReturnForecaster(feature_columns=feature_columns)
        model.fit(train, target="target_return")
        return pd.Series(model.predict(frame), index=frame.index, dtype=float), {"model": model}

    if spec.base_model == "bayesian_linear":
        model = BayesianLinearReturnForecaster(feature_columns=feature_columns)
        model.fit(train, target="target_return")
        mean, lower, upper = model.predict_interval(frame)
        return pd.Series(mean, index=frame.index, dtype=float), {
            "model": model,
            "bayes_lower": pd.Series(lower, index=frame.index, dtype=float),
            "bayes_upper": pd.Series(upper, index=frame.index, dtype=float),
        }

    if spec.base_model == "lightgbm":
        model = LightGBMRanker()
        model.features = list(feature_columns)
        model.fit(train, target="target_return")
        return pd.Series(model.predict(frame), index=frame.index, dtype=float), {"model": model}

    if spec.base_model == "hmm_router":
        model = SimpleHMMRegimeDetector()
        model.features = ["return_1d", "volatility_20d"]
        model.fit(train)
        train_regimes = pd.Series(model.predict(train), index=train.index, dtype=int)
        regime_returns = train.assign(regime=train_regimes)["target_return"].groupby(train_regimes).mean()
        if regime_returns.empty or regime_returns.dropna().empty:
            active_regime = 0
        else:
            active_regime = int(regime_returns.idxmax())
        regimes = pd.Series(model.predict(frame), index=frame.index, dtype=int)
        scores = (regimes == active_regime).astype(float)
        return scores, {"model": model, "regime_model": model, "active_regime": active_regime, "regimes": regimes}

    raise ValueError(f"Unsupported base model: {spec.base_model}")


def _apply_overlays(
    spec: ExperimentSpec,
    frame: pd.DataFrame,
    aux: dict[str, object],
    train: pd.DataFrame,
) -> pd.Series:
    score = frame["score"].astype(float).copy()

    if "confidence_gate" in spec.overlays:
        lower = aux.get("bayes_lower")
        if isinstance(lower, pd.Series) and len(lower) == len(score):
            score = score.where(lower > 0.0, 0.0)

    if "regime_gate" in spec.overlays:
        regime_model = aux.get("regime_model")
        active_regime = aux.get("active_regime")
        if regime_model is None:
            regime_model = SimpleHMMRegimeDetector()
            regime_model.features = ["return_1d", "volatility_20d"]
            regime_model.fit(train)
            train_regimes = pd.Series(regime_model.predict(train), index=train.index, dtype=int)
            regime_returns = train.assign(regime=train_regimes)["target_return"].groupby(train_regimes).mean()
            active_regime = int(regime_returns.idxmax()) if not regime_returns.empty else 0
            aux["regime_model"] = regime_model
            aux["active_regime"] = active_regime
        regimes = pd.Series(regime_model.predict(frame), index=frame.index, dtype=int)
        score = score.where(regimes == int(active_regime), 0.0)

    if "volatility_timing" in spec.overlays:
        vol_ref = float(train["volatility_20d"].median()) if "volatility_20d" in train.columns else 1.0
        scale = vol_ref / (frame["volatility_20d"].abs() + 1e-8)
        score = score * scale.clip(lower=0.0, upper=1.5)

    if "inverse_vol_allocation" in spec.overlays:
        inv_ref = float(train["inverse_vol"].median()) if "inverse_vol" in train.columns else 1.0
        scale = frame["inverse_vol"] / (inv_ref + 1e-8)
        score = score * scale.clip(lower=0.0, upper=2.0)

    if "jump_filter" in spec.overlays:
        shock_threshold = float(train["shock_score"].quantile(0.90)) if "shock_score" in train.columns else 2.5
        score = score.where(frame["shock_score"] < shock_threshold, 0.0)

    if "correlation_penalty" in spec.overlays:
        corr_threshold = float(train["corr_to_market_20d"].abs().quantile(0.75)) if "corr_to_market_20d" in train.columns else 0.5
        correlation_penalty = (corr_threshold / (frame["corr_to_market_20d"].abs() + 1e-8)).clip(lower=0.25, upper=1.0)
        score = score * correlation_penalty

    if "cvar_overlay" in spec.overlays:
        tail_threshold = float(train["tail_risk_20d"].abs().quantile(0.90)) if "tail_risk_20d" in train.columns else 0.05
        tail_severity = frame["tail_risk_20d"].abs() + 1e-8
        cvar_scale = (tail_threshold / tail_severity).clip(lower=0.2, upper=1.0)
        score = score * cvar_scale

    return score.astype(float)


def _build_weights_frame(
    spec: ExperimentSpec,
    scored_frame: pd.DataFrame,
    risk_manager: RiskManager,
) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for date, day_frame in scored_frame.groupby("date", sort=True):
        if spec.allocation_mode == "all_equal":
            weights = pd.Series(1.0 / len(day_frame), index=day_frame["symbol"].tolist(), dtype=float)
        else:
            weights = _normalize_positive(day_frame.set_index("symbol")["score"], spec.allocation_mode)

        capped = risk_manager.apply_position_caps(weights.to_dict())
        rows.append(pd.Series(capped, name=pd.to_datetime(date)))

    if not rows:
        return pd.DataFrame(columns=sorted(scored_frame["symbol"].unique()))

    weights_frame = pd.DataFrame(rows).sort_index().fillna(0.0)
    weights_frame = weights_frame.reindex(columns=sorted(scored_frame["symbol"].unique()), fill_value=0.0)
    return weights_frame


def _simulate_segment(
    price_returns: pd.DataFrame,
    weights_frame: pd.DataFrame,
    *,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.001,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    aligned_returns = price_returns.reindex(weights_frame.index).fillna(0.0)
    aligned_weights = weights_frame.reindex(aligned_returns.index).fillna(0.0)
    lagged_weights = aligned_weights.shift(1).fillna(0.0)
    gross_returns = (lagged_weights * aligned_returns).sum(axis=1)
    turnover = aligned_weights.diff().abs().sum(axis=1).fillna(aligned_weights.abs().sum(axis=1))
    cost_drag = turnover * (fee_rate + slippage_rate)
    net_returns = gross_returns - cost_drag
    gross_exposure = aligned_weights.abs().sum(axis=1)
    return net_returns, turnover, gross_exposure, cost_drag


@dataclass
class ExperimentOutcome:
    spec: ExperimentSpec
    train_summary: PerformanceSummary
    validation_summary: PerformanceSummary
    test_summary: PerformanceSummary
    decision: str
    feature_count: int
    runtime_seconds: float
    notes: str = ""
    status: str = "ok"

    def to_dict(self) -> dict[str, object]:
        return {
            "pipeline_id": self.spec.pipeline_id,
            "name": self.spec.name,
            "family": self.spec.family,
            "doc_status": self.spec.doc_status,
            "implementation_mode": self.spec.implementation_mode,
            "base_model": self.spec.base_model,
            "feature_profile": self.spec.feature_profile,
            "allocation_mode": self.spec.allocation_mode,
            "overlays": list(self.spec.overlays),
            "risk_cap": self.spec.risk_cap,
            "max_drawdown": self.spec.max_drawdown,
            "train_summary": self.train_summary.to_dict(),
            "validation_summary": self.validation_summary.to_dict(),
            "test_summary": self.test_summary.to_dict(),
            "decision": self.decision,
            "feature_count": self.feature_count,
            "runtime_seconds": self.runtime_seconds,
            "notes": self.notes or self.spec.note,
            "status": self.status,
            "source_doc": self.spec.source_doc,
        }


@dataclass
class BatchRunResult:
    dataset: DatasetBundle
    split: DatasetSplit
    outcomes: list[ExperimentOutcome]
    baseline_pipeline_id: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def outcome_map(self) -> dict[str, ExperimentOutcome]:
        return {outcome.spec.pipeline_id: outcome for outcome in self.outcomes}

    def baseline_outcome(self) -> ExperimentOutcome:
        for outcome in self.outcomes:
            if outcome.spec.pipeline_id == self.baseline_pipeline_id:
                return outcome
        raise KeyError(self.baseline_pipeline_id)


class CandidateBatchRunner:
    def __init__(self, dataset: DatasetBundle, *, output_dir: Path | None = None):
        self.dataset = dataset
        self.output_dir = Path(output_dir) if output_dir is not None else None

    def run(
        self,
        specs: Iterable[ExperimentSpec],
        *,
        baseline_pipeline_id: str = "F001",
    ) -> BatchRunResult:
        split = self.dataset.split()
        outcomes: list[ExperimentOutcome] = []
        price_returns = self.dataset.frame.pivot(index="date", columns="symbol", values="close").sort_index().pct_change().fillna(0.0)

        for spec in specs:
            outcome = self._run_single(spec, split, price_returns, baseline_pipeline_id=baseline_pipeline_id)
            outcomes.append(outcome)

        return BatchRunResult(dataset=self.dataset, split=split, outcomes=outcomes, baseline_pipeline_id=baseline_pipeline_id)

    def _run_single(
        self,
        spec: ExperimentSpec,
        split: DatasetSplit,
        price_returns: pd.DataFrame,
        *,
        baseline_pipeline_id: str,
    ) -> ExperimentOutcome:
        import time

        start = time.perf_counter()
        feature_columns = get_feature_columns(spec.feature_profile, split.train)
        if not feature_columns and spec.base_model not in {"equal_weight", "random_walk", "hmm_router"}:
            raise ValueError(f"No features available for {spec.pipeline_id} ({spec.feature_profile})")

        base_train_scores, aux = _score_base_model(spec, split.train, split.train, feature_columns)
        train_frame = split.train.copy()
        train_frame["score"] = base_train_scores
        train_frame["score"] = _apply_overlays(spec, train_frame, aux, split.train)

        validation_base_scores, validation_aux = _score_base_model(spec, split.train, split.validation, feature_columns)
        validation_frame = split.validation.copy()
        validation_frame["score"] = validation_base_scores
        validation_frame["score"] = _apply_overlays(spec, validation_frame, validation_aux | aux, split.train)

        test_base_scores, test_aux = _score_base_model(spec, split.train, split.test, feature_columns)
        test_frame = split.test.copy()
        test_frame["score"] = test_base_scores
        test_frame["score"] = _apply_overlays(spec, test_frame, test_aux | aux, split.train)

        risk_manager = RiskManager(max_position_cap=spec.risk_cap, max_drawdown=spec.max_drawdown)
        train_weights = _build_weights_frame(spec, train_frame, risk_manager)
        validation_weights = _build_weights_frame(spec, validation_frame, risk_manager)
        test_weights = _build_weights_frame(spec, test_frame, risk_manager)

        train_returns, train_turnover, train_gross, train_cost = _simulate_segment(
            price_returns.reindex(train_weights.index).fillna(0.0),
            train_weights,
        )
        validation_returns, validation_turnover, validation_gross, validation_cost = _simulate_segment(
            price_returns.reindex(validation_weights.index).fillna(0.0),
            validation_weights,
        )
        test_returns, test_turnover, test_gross, test_cost = _simulate_segment(
            price_returns.reindex(test_weights.index).fillna(0.0),
            test_weights,
        )

        train_summary = summarize_performance(train_returns, train_turnover, train_gross, cost_drag=train_cost)
        validation_summary = summarize_performance(validation_returns, validation_turnover, validation_gross, cost_drag=validation_cost)
        test_summary = summarize_performance(test_returns, test_turnover, test_gross, cost_drag=test_cost)

        baseline_outcome = None
        decision = "reference"
        if spec.pipeline_id != baseline_pipeline_id:
            baseline_outcome = None
        if spec.pipeline_id != baseline_pipeline_id:
            # Decision is assigned after the full run once the baseline is available.
            decision = "pending"

        runtime = time.perf_counter() - start
        notes = spec.note
        if spec.base_model == "bayesian_linear" and "confidence_gate" in spec.overlays:
            notes = f"{notes} Bayesian confidence gate active."
        return ExperimentOutcome(
            spec=spec,
            train_summary=train_summary,
            validation_summary=validation_summary,
            test_summary=test_summary,
            decision=decision,
            feature_count=len(feature_columns),
            runtime_seconds=runtime,
            notes=notes,
        )


def finalize_decisions(result: BatchRunResult) -> BatchRunResult:
    baseline = None
    for outcome in result.outcomes:
        if outcome.spec.pipeline_id == result.baseline_pipeline_id:
            baseline = outcome
            break

    if baseline is None:
        raise KeyError(f"Baseline {result.baseline_pipeline_id!r} not found in outcomes")

    for outcome in result.outcomes:
        if outcome.spec.pipeline_id == result.baseline_pipeline_id:
            outcome.decision = "reference"
            continue
        outcome.decision = classify_candidate(outcome.test_summary, baseline.test_summary)
    return result


def results_to_frame(result: BatchRunResult) -> pd.DataFrame:
    rows = [outcome.to_dict() for outcome in result.outcomes]
    frame = pd.json_normalize(rows)
    return frame.sort_values(["family", "pipeline_id"]).reset_index(drop=True)
