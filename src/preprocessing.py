"""Data preprocessing and leakage-free transformation pipeline.

Ensures all learned transformations (imputation, encoding, scaling) are
fitted solely on training observations, preventing temporal and target leakage.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config

logger = logging.getLogger(__name__)


def clean_raw_data(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict[str, Any]]:
    """Clean raw dataset: drop duplicates, parse dates, validate target.

    Args:
        raw_df: The uncleaned pandas DataFrame.

    Returns:
        Tuple[pd.DataFrame, dict[str, Any]]: Cleaned data and audit metadata.
    """
    df = raw_df.copy()
    original_rows = len(df)

    # 1. Duplicate check and removal
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    # 2. Date parsing and validation
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL], dayfirst=True, errors="coerce")
    invalid_dates = int(df[config.DATE_COL].isna().sum())
    df = df.dropna(subset=[config.DATE_COL]).copy()

    # 3. Numeric conversion for numerical columns
    for col in config.RAW_NUMERICAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Target validation (non-null and non-negative)
    target_series = pd.to_numeric(df[config.TARGET_COL], errors="coerce")
    invalid_target = int(target_series.isna().sum() + (target_series < 0).sum())
    df = df[target_series.notna() & (target_series >= 0)].copy()

    # 5. Outlier analysis via Tukey IQR (Audit only; genuine surges preserved)
    outlier_audit: dict[str, dict[str, float | int]] = {}
    audit_cols = [config.TARGET_COL, "Temperature(°C)", "Humidity(%)", "Rainfall(mm)", "Snowfall (cm)"]
    for col in audit_cols:
        if col in df.columns:
            valid_vals = df[col].dropna()
            q1 = float(valid_vals.quantile(0.25))
            q3 = float(valid_vals.quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_count = int(((valid_vals < lower_bound) | (valid_vals > upper_bound)).sum())
            outlier_audit[col] = {
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outlier_count,
            }

    # Sort strictly by date and hour to maintain true temporal sequence
    df = df.sort_values([config.DATE_COL, "Hour"]).reset_index(drop=True)

    cleaning_metadata = {
        "original_rows": original_rows,
        "duplicate_rows_removed": duplicate_count,
        "invalid_dates_removed": invalid_dates,
        "invalid_target_rows_removed": invalid_target,
        "final_rows": len(df),
        "outlier_audit": outlier_audit,
    }

    return df, cleaning_metadata


def split_data_chronologically(
    df: pd.DataFrame,
    split_ratio: float = config.CHRONOLOGICAL_SPLIT_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a strict chronological split of the dataset.

    Ensures that past data is used for training and future data is used for testing,
    faithfully simulating production deployment without temporal leakage.

    Args:
        df: Chronologically sorted pandas DataFrame.
        split_ratio: Ratio for training set (e.g., 0.8 for 80% train, 20% test).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    sorted_df = df.sort_values([config.DATE_COL, "Hour"]).reset_index(drop=True)
    split_index = int(len(sorted_df) * split_ratio)
    train_df = sorted_df.iloc[:split_index].copy()
    test_df = sorted_df.iloc[split_index:].copy()
    return train_df, test_df


def split_data_randomly(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = config.RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a random 80/20 train/test split.

    Used specifically to reproduce and evaluate against the historical baseline.

    Args:
        df: Pandas DataFrame.
        test_size: Proportion of test observations.
        random_state: Random seed.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df.copy(), test_df.copy()


def build_preprocessor_pipeline(
    numerical_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer for preprocessing.

    Numeric pipeline: SimpleImputer(median) + optional StandardScaler.
    Categorical pipeline: SimpleImputer(most_frequent) + OneHotEncoder(ignore unknown).

    All imputers and encoders are fitted during training only, avoiding data leakage.

    Args:
        numerical_features: List of numerical column names.
        categorical_features: List of categorical column names.
        scale_numeric: Whether to apply StandardScaler to numerical features.

    Returns:
        ColumnTransformer ready to fit on training data.
    """
    num_steps: list[Any] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=num_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor

