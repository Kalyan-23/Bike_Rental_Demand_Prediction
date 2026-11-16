"""Unit tests for inference and prediction validation."""

from __future__ import annotations

import pytest

import config
from src.prediction import DemandPredictor, categorize_demand, validate_raw_input


def test_categorize_demand_quartiles() -> None:
    """Verify mapping of rental values to empirical demand quartiles."""
    assert categorize_demand(50)[0] == "Low"
    assert categorize_demand(200)[0] == "Moderate"
    assert categorize_demand(400)[0] == "Moderate"
    assert categorize_demand(550)[0] == "High"
    assert categorize_demand(850)[0] == "High"
    assert categorize_demand(1100)[0] == "Very High"
    assert categorize_demand(2500)[0] == "Very High"


def test_validate_raw_input_valid() -> None:
    """Verify input sanitizer passes valid atmospheric inputs."""
    valid_input = {
        "Date": "2018-06-15",
        "Hour": 17,
        "Temperature(°C)": 24.5,
        "Humidity(%)": 55.0,
        "Wind speed (m/s)": 2.1,
        "Visibility (10m)": 2000.0,
        "Dew point temperature(°C)": 14.0,
        "Solar Radiation (MJ/m2)": 1.5,
        "Rainfall(mm)": 0.0,
        "Snowfall (cm)": 0.0,
        "Seasons": "Summer",
        "Holiday": "No Holiday",
        "Functioning Day": "Yes",
    }
    sanitized = validate_raw_input(valid_input)
    assert sanitized["Hour"] == 17
    assert sanitized["Temperature(°C)"] == 24.5
    assert sanitized["Seasons"] == "Summer"


def test_validate_raw_input_catches_invalid_ranges() -> None:
    """Verify input sanitizer rejects impossible physical values."""
    with pytest.raises(ValueError, match="Hour"):
        validate_raw_input({"Hour": 28})

    with pytest.raises(ValueError, match="Temperature"):
        validate_raw_input({"Hour": 12, "Temperature(°C)": 85.0})

    with pytest.raises(ValueError, match="Humidity"):
        validate_raw_input({"Hour": 12, "Temperature(°C)": 20.0, "Humidity(%)": 125.0})

    with pytest.raises(ValueError, match="Seasons"):
        validate_raw_input(
            {"Hour": 12, "Temperature(°C)": 20.0, "Humidity(%)": 50.0, "Seasons": "Tropical"}
        )


def test_predictor_single_and_functioning_day() -> None:
    """Verify end-to-end predictor produces bounded forecasts and respects system closure."""
    predictor = DemandPredictor()

    sample_active = {
        "Date": "2018-05-20",
        "Hour": 18,
        "Temperature(°C)": 22.0,
        "Humidity(%)": 50.0,
        "Wind speed (m/s)": 1.5,
        "Visibility (10m)": 2000.0,
        "Dew point temperature(°C)": 11.0,
        "Solar Radiation (MJ/m2)": 1.0,
        "Rainfall(mm)": 0.0,
        "Snowfall (cm)": 0.0,
        "Seasons": "Spring",
        "Holiday": "No Holiday",
        "Functioning Day": "Yes",
    }
    res_active = predictor.predict_single(sample_active)
    assert res_active["predicted_count"] >= 0
    assert res_active["demand_category"] in ["Low", "Moderate", "High", "Very High"]
    assert res_active["lower_bound_90pct"] >= 0

    # Test Functioning Day == 'No' must return exactly 0
    sample_closed = sample_active.copy()
    sample_closed["Functioning Day"] = "No"
    res_closed = predictor.predict_single(sample_closed)
    assert res_closed["predicted_count"] == 0

