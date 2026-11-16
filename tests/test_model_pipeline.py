"""Unit tests for feature engineering, model training, and evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.evaluation import calculate_smape, calculate_wape, evaluate_predictions
from src.feature_engineering import add_calendar_features, add_weather_features
from src.model_training import post_process_predictions


@pytest.fixture
def dummy_features_df() -> pd.DataFrame:
    """Fixture with basic raw features."""
    return pd.DataFrame(
        {
            "Date": ["2018-05-10", "2018-05-10", "2018-05-10"],
            "Hour": [8, 14, 18],
            "Temperature(°C)": [15.0, 24.0, 20.0],
            "Humidity(%)": [60.0, 45.0, 50.0],
            "Wind speed (m/s)": [1.5, 2.0, 1.2],
            "Visibility (10m)": [2000, 2000, 1800],
            "Dew point temperature(°C)": [7.0, 11.0, 9.0],
            "Solar Radiation (MJ/m2)": [0.8, 2.5, 1.2],
            "Rainfall(mm)": [0.0, 0.0, 0.0],
            "Snowfall (cm)": [0.0, 0.0, 0.0],
            "Seasons": ["Spring", "Spring", "Spring"],
            "Holiday": ["No Holiday", "No Holiday", "No Holiday"],
            "Functioning Day": ["Yes", "Yes", "No"],
        }
    )


def test_add_calendar_features(dummy_features_df: pd.DataFrame) -> None:
    """Verify generation of cyclical encodings and rush hour indicators."""
    res = add_calendar_features(dummy_features_df)
    assert "Hour_Sin" in res.columns
    assert "Hour_Cos" in res.columns
    assert "IsPeakHour" in res.columns
    assert "IsMorningRush" in res.columns
    assert "IsEveningRush" in res.columns
    assert res.loc[0, "IsMorningRush"] == 1
    assert res.loc[1, "IsMorningRush"] == 0
    assert res.loc[2, "IsEveningRush"] == 1


def test_add_weather_features(dummy_features_df: pd.DataFrame) -> None:
    """Verify weather categorization and discomfort index calculation."""
    res = add_weather_features(dummy_features_df)
    assert "Weather Category" in res.columns
    assert "Temp_Humidity" in res.columns
    assert "Discomfort_Index" in res.columns
    assert res.loc[0, "Weather Category"] == "Clear"


def test_post_process_predictions(dummy_features_df: pd.DataFrame) -> None:
    """Verify non-negative clipping and closure enforcement."""
    raw_preds = np.array([-25.0, 150.0, 400.0])
    processed = post_process_predictions(raw_preds, dummy_features_df)
    assert processed[0] == 0.0  # Clipped from negative
    assert processed[1] == 150.0
    assert processed[2] == 0.0  # Functioning Day == 'No'


def test_evaluation_metrics_zero_safety() -> None:
    """Verify metrics handle zero actuals and predictions safely."""
    y_true = np.array([0.0, 100.0, 200.0, 0.0])
    y_pred = np.array([0.0, 110.0, 190.0, 5.0])

    metrics = evaluate_predictions(y_true, y_pred)
    assert metrics["MAE"] >= 0
    assert metrics["RMSE"] >= 0
    assert metrics["WAPE"] >= 0
    assert metrics["sMAPE"] >= 0
    assert not np.isnan(metrics["WAPE"])
    assert not np.isnan(metrics["sMAPE"])

