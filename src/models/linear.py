from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from src.utils.logger import logger


def _ensure_2d(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


@dataclass
class LinearModelState:
    feature_columns: list[str]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    intercept: float
    coefficients: np.ndarray
    residual_variance: float


class _BaseLinearForecaster:
    def __init__(self, feature_columns: Iterable[str] | None = None):
        self.feature_columns = list(feature_columns or [])
        self.feature_mean_: np.ndarray | None = None
        self.feature_scale_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.coefficients_: np.ndarray | None = None
        self.residual_variance_: float = 1e-8
        self.is_fitted = False

    def _resolve_feature_columns(self, df: pd.DataFrame) -> list[str]:
        if self.feature_columns:
            missing = [col for col in self.feature_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing feature columns: {missing}")
            return list(self.feature_columns)

        numeric = []
        for column in df.columns:
            if column in {"date", "symbol", "target_return"}:
                continue
            if pd.api.types.is_numeric_dtype(df[column]):
                numeric.append(column)

        if not numeric:
            raise ValueError("No numeric feature columns available for linear forecaster")

        self.feature_columns = numeric
        return numeric

    def _prepare_matrix(self, df: pd.DataFrame) -> np.ndarray:
        columns = self._resolve_feature_columns(df)
        matrix = df[columns].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        return _ensure_2d(matrix)

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        if self.feature_mean_ is None or self.feature_scale_ is None:
            raise RuntimeError("Linear forecaster is not fitted")
        return (matrix - self.feature_mean_) / self.feature_scale_

    def _design_matrix(self, df: pd.DataFrame) -> np.ndarray:
        matrix = self._prepare_matrix(df)
        normalized = self._normalize(matrix)
        return np.column_stack([np.ones(len(normalized)), normalized])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.coefficients_ is None:
            logger.warning("Linear forecaster is not fitted. Returning zeros.")
            return np.zeros(len(df), dtype=float)
        if df.empty:
            return np.empty(0, dtype=float)
        matrix = self._design_matrix(df)
        return matrix @ np.r_[self.intercept_, self.coefficients_]

    def save(self, path: str | Path) -> None:
        payload = LinearModelState(
            feature_columns=list(self.feature_columns),
            feature_mean=self.feature_mean_ if self.feature_mean_ is not None else np.array([]),
            feature_scale=self.feature_scale_ if self.feature_scale_ is not None else np.array([]),
            intercept=float(self.intercept_),
            coefficients=self.coefficients_ if self.coefficients_ is not None else np.array([]),
            residual_variance=float(self.residual_variance_),
        )
        joblib.dump(payload, Path(path))

    @classmethod
    def _restore_state(cls, path: str | Path) -> LinearModelState:
        payload = joblib.load(Path(path))
        if not isinstance(payload, LinearModelState):
            raise TypeError(f"Unexpected linear model payload type: {type(payload)!r}")
        return payload


class LinearReturnForecaster(_BaseLinearForecaster):
    """
    Plain linear regression baseline for next-period return forecasting.
    """

    def fit(self, df: pd.DataFrame, target: str = "target_return") -> None:
        if target not in df.columns:
            raise ValueError(f"Target column {target!r} missing from training frame")
        if df.empty:
            raise ValueError("Training frame cannot be empty")

        matrix = self._prepare_matrix(df)
        target_values = df[target].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        self.feature_mean_ = matrix.mean(axis=0)
        self.feature_scale_ = matrix.std(axis=0)
        self.feature_scale_[self.feature_scale_ == 0.0] = 1.0

        normalized = (matrix - self.feature_mean_) / self.feature_scale_
        design = np.column_stack([np.ones(len(normalized)), normalized])
        coefficients, *_ = np.linalg.lstsq(design, target_values, rcond=None)

        self.intercept_ = float(coefficients[0])
        self.coefficients_ = coefficients[1:]
        residuals = target_values - design @ coefficients
        self.residual_variance_ = float(np.var(residuals)) if len(residuals) else 1e-8
        self.is_fitted = True
        logger.info("LinearReturnForecaster fitted", feature_count=len(self.feature_columns))

    @classmethod
    def load(cls, path: str | Path) -> "LinearReturnForecaster":
        payload = cls._restore_state(path)
        obj = cls(feature_columns=payload.feature_columns)
        obj.feature_mean_ = payload.feature_mean
        obj.feature_scale_ = payload.feature_scale
        obj.intercept_ = float(payload.intercept)
        obj.coefficients_ = payload.coefficients
        obj.residual_variance_ = float(payload.residual_variance)
        obj.is_fitted = True
        return obj


class BayesianLinearReturnForecaster(_BaseLinearForecaster):
    """
    Conjugate Bayesian linear forecaster with a Gaussian prior.

    This is a lightweight proxy for the Bayesian return forecaster and Bayesian
    factor regression families in the shortlist.
    """

    def __init__(
        self,
        feature_columns: Iterable[str] | None = None,
        *,
        prior_precision: float = 1.0,
        noise_precision: float = 1.0,
    ):
        super().__init__(feature_columns=feature_columns)
        self.prior_precision = float(prior_precision)
        self.noise_precision = float(noise_precision)
        self.posterior_cov_: np.ndarray | None = None

    def fit(self, df: pd.DataFrame, target: str = "target_return") -> None:
        if target not in df.columns:
            raise ValueError(f"Target column {target!r} missing from training frame")
        if df.empty:
            raise ValueError("Training frame cannot be empty")

        matrix = self._prepare_matrix(df)
        target_values = df[target].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        self.feature_mean_ = matrix.mean(axis=0)
        self.feature_scale_ = matrix.std(axis=0)
        self.feature_scale_[self.feature_scale_ == 0.0] = 1.0

        normalized = (matrix - self.feature_mean_) / self.feature_scale_
        design = np.column_stack([np.ones(len(normalized)), normalized])

        prior = np.eye(design.shape[1]) * self.prior_precision
        prior[0, 0] = 1e-6  # keep intercept loosely regularized
        precision = prior + self.noise_precision * design.T @ design
        posterior_cov = np.linalg.pinv(precision)
        posterior_mean = self.noise_precision * posterior_cov @ design.T @ target_values

        self.intercept_ = float(posterior_mean[0])
        self.coefficients_ = posterior_mean[1:]
        self.posterior_cov_ = posterior_cov
        residuals = target_values - design @ posterior_mean
        self.residual_variance_ = float(np.var(residuals)) if len(residuals) else 1e-8
        self.is_fitted = True
        logger.info("BayesianLinearReturnForecaster fitted", feature_count=len(self.feature_columns))

    def predict_interval(self, df: pd.DataFrame, z: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_fitted or self.coefficients_ is None or self.posterior_cov_ is None:
            logger.warning("Bayesian linear forecaster is not fitted. Returning zeros.")
            zeros = np.zeros(len(df), dtype=float)
            return zeros, zeros, zeros
        if df.empty:
            empty = np.empty(0, dtype=float)
            return empty, empty, empty

        design = self._design_matrix(df)
        mean = design @ np.r_[self.intercept_, self.coefficients_]
        predictive_var = self.residual_variance_ * (
            1.0 + np.einsum("ij,jk,ik->i", design, self.posterior_cov_, design)
        )
        predictive_std = np.sqrt(np.maximum(predictive_var, 1e-12))
        lower = mean - z * predictive_std
        upper = mean + z * predictive_std
        return mean, lower, upper

    def save(self, path: str | Path) -> None:
        payload = {
            "state": LinearModelState(
                feature_columns=list(self.feature_columns),
                feature_mean=self.feature_mean_ if self.feature_mean_ is not None else np.array([]),
                feature_scale=self.feature_scale_ if self.feature_scale_ is not None else np.array([]),
                intercept=float(self.intercept_),
                coefficients=self.coefficients_ if self.coefficients_ is not None else np.array([]),
                residual_variance=float(self.residual_variance_),
            ),
            "prior_precision": self.prior_precision,
            "noise_precision": self.noise_precision,
            "posterior_cov": self.posterior_cov_,
        }
        joblib.dump(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "BayesianLinearReturnForecaster":
        payload = joblib.load(Path(path))
        if not isinstance(payload, dict) or "state" not in payload:
            raise TypeError(f"Unexpected Bayesian linear payload type: {type(payload)!r}")

        state = payload["state"]
        if not isinstance(state, LinearModelState):
            raise TypeError(f"Unexpected state payload type: {type(state)!r}")

        obj = cls(
            feature_columns=state.feature_columns,
            prior_precision=float(payload.get("prior_precision", 1.0)),
            noise_precision=float(payload.get("noise_precision", 1.0)),
        )
        obj.feature_mean_ = state.feature_mean
        obj.feature_scale_ = state.feature_scale
        obj.intercept_ = float(state.intercept)
        obj.coefficients_ = state.coefficients
        obj.residual_variance_ = float(state.residual_variance)
        posterior_cov = payload.get("posterior_cov")
        obj.posterior_cov_ = posterior_cov if posterior_cov is not None else None
        obj.is_fitted = True
        return obj
