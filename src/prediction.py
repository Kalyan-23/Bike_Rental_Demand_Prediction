"""Inference and real-time prediction pipeline.

Provides structured input validation, automatic feature generation,
demand category classification based on empirical quartiles, and
uncertainty bounds estimation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import config
from src.feature_engineering import add_calendar_features, add_weather_features
from src.model_training import load_model_bundle, post_process_predictions

logger = logging.getLogger(__name__)


def categorize_demand(rental_count: float) -> Tuple[str, str, str]:
    """Map a numerical rental count to an empirical demand category.

    Categories are derived from historical dataset quartiles:
      - Low: 0 - 200 bikes (Bottom 25%)
      - Moderate: 201 - 550 bikes (25th to 50th percentile)
      - High: 551 - 1,100 bikes (50th to 75th percentile)
      - Very High: > 1,100 bikes (Top 25% surge)

    Args:
        rental_count: Predicted bike rental count.

    Returns:
        Tuple[str, str, str]: (Category Name, Hex Color, Description)
    """
    count = max(0.0, float(rental_count))
    for category, (low, high) in config.DEMAND_THRESHOLDS.items():
        if low <= count < high:
            color = config.DEMAND_CATEGORY_COLORS.get(category, "#38bdf8")
            if category == "Low":
                desc = "Off-peak or unfavorable conditions; minimal fleet needed."
            elif category == "Moderate":
                desc = "Steady, standard demand; normal fleet availability recommended."
            elif category == "High":
                desc = "Strong demand; high station turnover expected."
            else:
                desc = "Surge rush demand; proactive rebalancing and maximum fleet deployment required."
            return category, color, desc

    return "Very High", config.DEMAND_CATEGORY_COLORS["Very High"], "Surge rush demand."


def validate_raw_input(raw_inputs: dict[str, Any]) -> dict[str, Any]:
    """Validate and sanitize user prediction inputs.

    Args:
        raw_inputs: Dictionary of input values from the user or API.

    Returns:
        dict[str, Any]: Sanitized and validated inputs.

    Raises:
        ValueError: If any input is out of realistic physical boundaries.
    """
    sanitized = raw_inputs.copy()

    # Numeric range validations
    hour = int(sanitized.get("Hour", 12))
    if not (0 <= hour <= 23):
        raise ValueError(f"Hour must be an integer between 0 and 23. Received: {hour}")
    sanitized["Hour"] = hour

    temp = float(sanitized.get("Temperature(°C)", 20.0))
    if not (-30.0 <= temp <= 55.0):
        raise ValueError(f"Temperature must be between -30°C and 55°C. Received: {temp}")
    sanitized["Temperature(°C)"] = temp

    humidity = float(sanitized.get("Humidity(%)", 50.0))
    if not (0.0 <= humidity <= 100.0):
        raise ValueError(f"Humidity must be between 0% and 100%. Received: {humidity}")
    sanitized["Humidity(%)"] = humidity

    wind = float(sanitized.get("Wind speed (m/s)", 1.5))
    if not (0.0 <= wind <= 40.0):
        raise ValueError(f"Wind speed must be between 0 and 40 m/s. Received: {wind}")
    sanitized["Wind speed (m/s)"] = wind

    visibility = float(sanitized.get("Visibility (10m)", 2000.0))
    if not (0.0 <= visibility <= 3000.0):
        raise ValueError(f"Visibility must be between 0 and 3000 (10m). Received: {visibility}")
    sanitized["Visibility (10m)"] = visibility

    dew = float(sanitized.get("Dew point temperature(°C)", 10.0))
    if not (-40.0 <= dew <= 40.0):
        raise ValueError(f"Dew point must be between -40°C and 40°C. Received: {dew}")
    sanitized["Dew point temperature(°C)"] = dew

    solar = float(sanitized.get("Solar Radiation (MJ/m2)", 0.0))
    if not (0.0 <= solar <= 10.0):
        raise ValueError(f"Solar radiation must be between 0 and 10 MJ/m2. Received: {solar}")
    sanitized["Solar Radiation (MJ/m2)"] = solar

    rain = float(sanitized.get("Rainfall(mm)", 0.0))
    if not (0.0 <= rain <= 200.0):
        raise ValueError(f"Rainfall must be between 0 and 200 mm. Received: {rain}")
    sanitized["Rainfall(mm)"] = rain

    snow = float(sanitized.get("Snowfall (cm)", 0.0))
    if not (0.0 <= snow <= 100.0):
        raise ValueError(f"Snowfall must be between 0 and 100 cm. Received: {snow}")
    sanitized["Snowfall (cm)"] = snow

    # Categorical validations
    season = str(sanitized.get("Seasons", "Spring"))
    if season not in config.SEASONS:
        raise ValueError(f"Seasons must be one of {config.SEASONS}. Received: {season}")
    sanitized["Seasons"] = season

    holiday = str(sanitized.get("Holiday", "No Holiday"))
    if holiday not in config.HOLIDAYS:
        raise ValueError(f"Holiday must be one of {config.HOLIDAYS}. Received: {holiday}")
    sanitized["Holiday"] = holiday

    func_day = str(sanitized.get("Functioning Day", "Yes"))
    if func_day not in config.FUNCTIONING_DAYS:
        raise ValueError(f"Functioning Day must be one of {config.FUNCTIONING_DAYS}. Received: {func_day}")
    sanitized["Functioning Day"] = func_day

    # Date validation
    date_val = sanitized.get("Date", "2018-05-15")
    try:
        parsed_dt = pd.to_datetime(date_val, format="mixed", errors="coerce")
        sanitized["Date"] = parsed_dt.strftime("%Y-%m-%d")
    except Exception as exc:
        raise ValueError(f"Invalid Date '{date_val}': {exc}") from exc

    return sanitized


class DemandPredictor:
    """Production predictor wrapper with feature pipeline and confidence bounds."""

    def __init__(self, model_bundle_path: Optional[str | Path] = None):
        """Initialize predictor by loading the serialized model bundle."""
        self.bundle = load_model_bundle(model_bundle_path)
        self.pipeline = self.bundle["pipeline"]
        self.model_name = self.bundle.get("model_name", "Best Model")
        self.mode = self.bundle.get("mode", "Mode 1: Weather Scenario")
        self.metrics = self.bundle.get("metrics", {})
        self.rmse = float(self.metrics.get("RMSE", 150.0))
        self.mae = float(self.metrics.get("MAE", 90.0))

    def predict_single(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        """Generate prediction and diagnostic insights for a single input scenario.

        Args:
            raw_input: Dictionary of weather, time, and calendar conditions.

        Returns:
            dict containing predicted count, demand category, bounds, and inputs.
        """
        validated = validate_raw_input(raw_input)
        df_input = pd.DataFrame([validated])

        # Engineer calendar and weather features
        df_feat = add_calendar_features(df_input)
        df_feat = add_weather_features(df_feat)

        # If Mode 2 lags are needed and supplied in raw_input, include them
        if "Lag_1h" in validated:
            df_feat["Lag_1h"] = float(validated["Lag_1h"])
            df_feat["Lag_2h"] = float(validated.get("Lag_2h", validated["Lag_1h"]))
            df_feat["Lag_24h"] = float(validated.get("Lag_24h", validated["Lag_1h"]))
            df_feat["Rolling_Mean_3h"] = float(validated.get("Rolling_Mean_3h", validated["Lag_1h"]))
            df_feat["Rolling_Mean_24h"] = float(validated.get("Rolling_Mean_24h", validated["Lag_1h"]))

        # Execute pipeline prediction
        raw_pred = self.pipeline.predict(df_feat)
        pred_value = float(post_process_predictions(raw_pred, df_input)[0])

        # Categorize demand
        category, color, description = categorize_demand(pred_value)

        # Approximate 90% confidence interval (+/- 1.645 * RMSE)
        lower_bound = max(0.0, round(pred_value - 1.645 * self.rmse, 0))
        upper_bound = round(pred_value + 1.645 * self.rmse, 0)

        return {
            "predicted_count": int(round(pred_value)),
            "exact_prediction": round(pred_value, 2),
            "demand_category": category,
            "category_color": color,
            "category_description": description,
            "lower_bound_90pct": int(lower_bound),
            "upper_bound_90pct": int(upper_bound),
            "model_used": self.model_name,
            "model_mode": self.mode,
            "model_mae": self.mae,
            "model_rmse": self.rmse,
            "functioning_day": validated["Functioning Day"],
            "input_summary": {
                "Date": validated["Date"],
                "Hour": validated["Hour"],
                "Temperature": f"{validated['Temperature(°C)']}°C",
                "Humidity": f"{validated['Humidity(%)']}%",
                "Weather": df_feat.iloc[0]["Weather Category"],
                "Season": validated["Seasons"],
                "Holiday": validated["Holiday"],
                "Is Peak Hour": bool(df_feat.iloc[0]["IsPeakHour"]),
                "Feels Like Index": f"{round(df_feat.iloc[0]['Discomfort_Index'], 1)}",
            },
        }

    def predict_batch(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Batch predict demand on an entire DataFrame of observations.

        Args:
            df_raw: DataFrame containing required raw input columns.

        Returns:
            pd.DataFrame: Original DataFrame augmented with predictions and categories.
        """
        df_feat = add_calendar_features(df_raw)
        df_feat = add_weather_features(df_feat)
        raw_preds = self.pipeline.predict(df_feat)
        processed = post_process_predictions(raw_preds, df_raw)

        result_df = df_raw.copy()
        result_df["Predicted_Rental_Count"] = np.round(processed, 1)
        result_df["Demand_Category"] = [categorize_demand(p)[0] for p in processed]
        return result_df
