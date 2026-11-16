"""Evaluation metrics and diagnostic analysis for regression models.

Implements standard regression metrics (MAE, MSE, RMSE, R2, Explained Variance)
along with percentage-based metrics robust to zero rentals (WAPE, sMAPE)
and comprehensive residual diagnostics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import config


def calculate_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Weighted Absolute Percentage Error (WAPE).

    WAPE = (sum(|y_true - y_pred|) / sum(y_true)) * 100
    Robust against zero values in individual observations.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        float: WAPE percentage.
    """
    total_actual = float(np.sum(y_true))
    if total_actual == 0:
        return 0.0
    total_abs_error = float(np.sum(np.abs(y_true - y_pred)))
    return (total_abs_error / total_actual) * 100.0


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error (sMAPE).

    sMAPE = mean( 200 * |y_true - y_pred| / (|y_true| + |y_pred| + eps) )
    Bounded between 0% and 200%, avoiding zero division.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.
        epsilon: Small constant to avoid 0/0.

    Returns:
        float: sMAPE percentage.
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator > epsilon
    if not np.any(mask):
        return 0.0
    smape_vals = np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
    return float(np.mean(smape_vals) * 100.0)


def evaluate_predictions(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Calculate all standard and specialized regression evaluation metrics.

    Args:
        y_true: Actual demand values.
        y_pred: Predicted demand values.

    Returns:
        dict[str, float]: Comprehensive evaluation metrics dictionary.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    # Scikit-learn metrics
    mae = float(mean_absolute_error(y_t, y_p))
    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_t, y_p))
    evs = float(explained_variance_score(y_t, y_p))

    # Percentage metrics
    wape = float(calculate_wape(y_t, y_p))
    smape = float(calculate_smape(y_t, y_p))

    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "WAPE": round(wape, 4),
        "sMAPE": round(smape, 4),
        "Explained_Variance": round(evs, 4),
    }


def compute_residual_diagnostics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Compute detailed residual analysis statistics.

    Residual = y_true - y_pred. Positive residual indicates underprediction;
    negative indicates overprediction.

    Args:
        y_true: Actual demand values.
        y_pred: Predicted demand values.

    Returns:
        dict[str, Any]: Statistical properties of prediction residuals.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    residuals = y_t - y_p

    return {
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
        "median_residual": float(np.median(residuals)),
        "max_underprediction": float(np.max(residuals)),
        "max_overprediction": float(np.abs(np.min(residuals))),
        "skewness": float(stats.skew(residuals)),
        "kurtosis": float(stats.kurtosis(residuals)),
        "residuals": residuals,
    }


def get_metric_explanations() -> dict[str, str]:
    """Return human-readable plain language descriptions of evaluation metrics."""
    return config.METRIC_DEFINITIONS.copy()

