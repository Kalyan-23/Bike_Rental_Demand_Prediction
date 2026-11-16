"""Feature engineering module for Seoul Bike Sharing demand prediction.

Implements calendar features, cyclical transformations, interaction terms,
domain-informed weather indices, and non-leaking lag/rolling statistics.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer calendar, temporal, and cyclical features from Date and Hour.

    Args:
        df: Input DataFrame containing Date and Hour.

    Returns:
        pd.DataFrame: DataFrame with engineered calendar features.
    """
    data = df.copy()
    date_dt = pd.to_datetime(data[config.DATE_COL], format="mixed", errors="coerce")

    data["Year"] = date_dt.dt.year.astype(int)
    data["Month"] = date_dt.dt.month.astype(int)
    data["Day"] = date_dt.dt.day.astype(int)
    data["DayOfWeek"] = date_dt.dt.dayofweek.astype(int)
    data["DayName"] = date_dt.dt.day_name()
    data["DayOfYear"] = date_dt.dt.dayofyear.astype(int)
    data["IsWeekend"] = (data["DayOfWeek"] >= 5).astype(int)

    # Hourly rush-hour indicators
    data["IsPeakHour"] = data["Hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    data["IsMorningRush"] = data["Hour"].isin([7, 8, 9]).astype(int)
    data["IsEveningRush"] = data["Hour"].isin([17, 18, 19]).astype(int)
    data["IsNight"] = data["Hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    data["IsBusinessHour"] = ((data["IsWeekend"] == 0) & (data["Hour"].between(9, 17))).astype(int)

    # Cyclical sin/cos encodings for periodic variables
    data["Hour_Sin"] = np.sin(2 * np.pi * data["Hour"] / 24.0)
    data["Hour_Cos"] = np.cos(2 * np.pi * data["Hour"] / 24.0)

    data["Month_Sin"] = np.sin(2 * np.pi * data["Month"] / 12.0)
    data["Month_Cos"] = np.cos(2 * np.pi * data["Month"] / 12.0)

    data["DayOfWeek_Sin"] = np.sin(2 * np.pi * data["DayOfWeek"] / 7.0)
    data["DayOfWeek_Cos"] = np.cos(2 * np.pi * data["DayOfWeek"] / 7.0)

    return data


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer domain-informed weather indices and interaction terms.

    Args:
        df: Input DataFrame containing meteorological fields.

    Returns:
        pd.DataFrame: DataFrame with enriched weather features.
    """
    data = df.copy()

    rainfall = data["Rainfall(mm)"].fillna(0)
    snowfall = data["Snowfall (cm)"].fillna(0)
    visibility = data["Visibility (10m)"].fillna(2000)
    temp = data["Temperature(°C)"].fillna(15.0)
    humidity = data["Humidity(%)"].fillna(50.0)
    wind = data["Wind speed (m/s)"].fillna(1.5)

    data["Is_Raining"] = (rainfall > 0).astype(int)
    data["Is_Snowing"] = (snowfall > 0).astype(int)
    data["High_Wind"] = (wind > 3.0).astype(int)

    # Weather Category classification
    data["Weather Category"] = np.select(
        [
            snowfall > 0,
            rainfall > 0,
            visibility < 500,
        ],
        [
            "Snow",
            "Rain",
            "Foggy",
        ],
        default="Clear",
    )

    # Temperature and humidity interaction
    data["Temp_Humidity"] = temp * humidity

    # Thom's Discomfort Index (Korean Meteorological Standard):
    # DI = 0.81 * Ta + 0.01 * RH * (0.99 * Ta - 14.3) + 46.3
    data["Discomfort_Index"] = 0.81 * temp + 0.01 * humidity * (0.99 * temp - 14.3) + 46.3

    # Severe weather indicator
    data["Severe_Weather"] = (
        (rainfall > 5.0) | (snowfall > 2.0) | (temp < -10.0) | (temp > 35.0) | (wind > 5.0)
    ).astype(int)

    return data


def add_lag_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 24),
    rolling_windows: tuple[int, ...] = (3, 24),
) -> pd.DataFrame:
    """Construct auto-regressive lag and rolling mean features without target leakage.

    Each lag and rolling window is strictly shifted by at least 1 hour, ensuring
    no future or concurrent target observations are leaked into the predictor set.

    Args:
        df: Chronologically sorted DataFrame containing TARGET_COL.
        lags: Tuple of hourly lags to compute.
        rolling_windows: Tuple of rolling window lengths in hours.

    Returns:
        pd.DataFrame: DataFrame with lag and rolling features.
    """
    data = df.copy()
    if config.TARGET_COL not in data.columns:
        logger.warning("Target column '%s' not present; skipping lag features.", config.TARGET_COL)
        return data

    target_series = data[config.TARGET_COL]

    for lag in lags:
        data[f"Lag_{lag}h"] = target_series.shift(lag)

    for window in rolling_windows:
        # Shifted by 1 so the current hour is never included in the rolling window
        data[f"Rolling_Mean_{window}h"] = target_series.shift(1).rolling(window=window).mean()

    return data


def build_full_features(
    df: pd.DataFrame,
    include_lags: bool = False,
    drop_lag_nans: bool = True,
) -> pd.DataFrame:
    """Generate the complete set of engineered features.

    Args:
        df: Raw or partially cleaned DataFrame.
        include_lags: If True, engineer auto-regressive lag features (Mode 2).
                      If False, generate exogenous weather & calendar features only (Mode 1).
        drop_lag_nans: If True and include_lags is True, drop rows where lags are NaN.

    Returns:
        pd.DataFrame: Fully engineered DataFrame.
    """
    data = add_calendar_features(df)
    data = add_weather_features(data)

    if include_lags:
        data = add_lag_features(data)
        if drop_lag_nans:
            lag_cols = config.LAG_FEATURES
            existing_lag_cols = [c for c in lag_cols if c in data.columns]
            if existing_lag_cols:
                data = data.dropna(subset=existing_lag_cols).reset_index(drop=True)

    return data
