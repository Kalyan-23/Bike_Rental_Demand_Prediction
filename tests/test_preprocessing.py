"""Unit tests for data preprocessing and splitting pipelines."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

import config
from src.preprocessing import clean_raw_data, split_data_chronologically, split_data_randomly


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Fixture providing mock raw dataset observations."""
    return pd.DataFrame(
        {
            "Date": ["01/12/2017", "01/12/2017", "02/12/2017", "02/12/2017", "03/12/2017"],
            "Rented Bike Count": [150, 220, 310, 400, 50],
            "Hour": [8, 18, 8, 18, 12],
            "Temperature(°C)": [-2.5, -0.8, 1.2, 0.5, -1.0],
            "Humidity(%)": [55, 60, 48, 52, 70],
            "Wind speed (m/s)": [1.2, 1.5, 2.0, 1.8, 0.9],
            "Visibility (10m)": [1800, 1600, 2000, 2000, 1500],
            "Dew point temperature(°C)": [-10.2, -7.5, -8.1, -8.0, -5.5],
            "Solar Radiation (MJ/m2)": [0.4, 0.0, 0.8, 0.0, 0.5],
            "Rainfall(mm)": [0.0, 0.0, 0.0, 0.0, 0.5],
            "Snowfall (cm)": [0.0, 0.0, 0.0, 0.0, 0.0],
            "Seasons": ["Winter", "Winter", "Winter", "Winter", "Winter"],
            "Holiday": ["No Holiday", "No Holiday", "No Holiday", "No Holiday", "Holiday"],
            "Functioning Day": ["Yes", "Yes", "Yes", "Yes", "Yes"],
        }
    )


def test_clean_raw_data_valid(sample_raw_df: pd.DataFrame) -> None:
    """Test standard cleaning without corrupt rows."""
    cleaned, meta = clean_raw_data(sample_raw_df)
    assert len(cleaned) == 5
    assert meta["duplicate_rows_removed"] == 0
    assert meta["invalid_dates_removed"] == 0
    assert meta["invalid_target_rows_removed"] == 0
    assert "outlier_audit" in meta


def test_clean_raw_data_handles_duplicates(sample_raw_df: pd.DataFrame) -> None:
    """Test duplicate detection and dropping."""
    dup_df = pd.concat([sample_raw_df, sample_raw_df.iloc[[0]]], ignore_index=True)
    assert len(dup_df) == 6
    cleaned, meta = clean_raw_data(dup_df)
    assert len(cleaned) == 5
    assert meta["duplicate_rows_removed"] == 1


def test_clean_raw_data_drops_invalid_target(sample_raw_df: pd.DataFrame) -> None:
    """Test removal of negative target rows."""
    corrupt_df = sample_raw_df.copy()
    corrupt_df.loc[0, "Rented Bike Count"] = -50
    cleaned, meta = clean_raw_data(corrupt_df)
    assert len(cleaned) == 4
    assert meta["invalid_target_rows_removed"] == 1


def test_chronological_split_strict_ordering(sample_raw_df: pd.DataFrame) -> None:
    """Verify chronological split guarantees training dates <= test dates."""
    cleaned, _ = clean_raw_data(sample_raw_df)
    train_df, test_df = split_data_chronologically(cleaned, split_ratio=0.6)
    assert len(train_df) == 3
    assert len(test_df) == 2

    # Check temporal boundary
    max_train_date = train_df[config.DATE_COL].max()
    min_test_date = test_df[config.DATE_COL].min()
    assert max_train_date <= min_test_date


def test_random_split_reproducibility(sample_raw_df: pd.DataFrame) -> None:
    """Test random split consistency with fixed seed."""
    cleaned, _ = clean_raw_data(sample_raw_df)
    tr1, te1 = split_data_randomly(cleaned, test_size=0.4, random_state=42)
    tr2, te2 = split_data_randomly(cleaned, test_size=0.4, random_state=42)
    pd.testing.assert_frame_equal(tr1, tr2)
    pd.testing.assert_frame_equal(te1, te2)

