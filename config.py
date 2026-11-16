"""Central configuration module for the Seoul Bike Sharing Prediction System."""

from __future__ import annotations

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
VIS_DIR = OUTPUT_DIR / "visualizations"
TESTS_DIR = BASE_DIR / "tests"

# Ensure all primary directories exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUT_DIR, VIS_DIR, TESTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File paths
DEFAULT_DATASET_PATH = DATA_DIR / "SeoulBikeData.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_COMPARISON_PATH = OUTPUT_DIR / "Model_Comparison.csv"
BASELINE_COMPARISON_PATH = OUTPUT_DIR / "Baseline_Model_Comparison.csv"
ANALYSIS_SUMMARY_PATH = OUTPUT_DIR / "Analysis_Summary.csv"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "Feature_Importance.csv"
TEST_PREDICTIONS_PATH = OUTPUT_DIR / "Test_Predictions.csv"
POWER_BI_DATA_PATH = OUTPUT_DIR / "SeoulBike_Updated_For_PowerBI.csv"
CLEANED_DATA_PATH = OUTPUT_DIR / "SeoulBike_Cleaned.csv"
METADATA_PATH = OUTPUT_DIR / "run_metadata.json"

# Column specifications
DATE_COL = "Date"
TARGET_COL = "Rented Bike Count"

RAW_NUMERICAL_COLS = [
    "Hour",
    "Temperature(°C)",
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Dew point temperature(°C)",
    "Solar Radiation (MJ/m2)",
    "Rainfall(mm)",
    "Snowfall (cm)",
]

RAW_CATEGORICAL_COLS = [
    "Seasons",
    "Holiday",
    "Functioning Day",
]

# Engineered numerical features for Mode 1 (Weather & Calendar Scenario)
ENGINEERED_NUMERICAL_COLS = [
    "Hour",
    "Hour_Sin",
    "Hour_Cos",
    "Month",
    "Month_Sin",
    "Month_Cos",
    "DayOfWeek",
    "DayOfWeek_Sin",
    "DayOfWeek_Cos",
    "IsWeekend",
    "DayOfYear",
    "IsPeakHour",
    "IsMorningRush",
    "IsEveningRush",
    "IsNight",
    "Temperature(°C)",
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Dew point temperature(°C)",
    "Solar Radiation (MJ/m2)",
    "Rainfall(mm)",
    "Snowfall (cm)",
    "Temp_Humidity",
    "Discomfort_Index",
    "Is_Raining",
    "Is_Snowing",
    "High_Wind",
]

# Engineered categorical features
ENGINEERED_CATEGORICAL_COLS = [
    "Seasons",
    "Holiday",
    "Functioning Day",
    "Weather Category",
]

# Lag-based features for Mode 2 (Historical Data Assisted Forecasting)
LAG_FEATURES = [
    "Lag_1h",
    "Lag_2h",
    "Lag_24h",
    "Rolling_Mean_3h",
    "Rolling_Mean_24h",
]

# Categorical domain values
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]
HOLIDAYS = ["No Holiday", "Holiday"]
FUNCTIONING_DAYS = ["Yes", "No"]
WEATHER_CATEGORIES = ["Clear", "Rain", "Snow", "Foggy"]

# Realistic empirical demand category thresholds (derived from dataset quartiles: 25th=191, 50th=505, 75th=1065)
DEMAND_THRESHOLDS = {
    "Low": (0.0, 200.0),
    "Moderate": (200.0, 550.0),
    "High": (550.0, 1100.0),
    "Very High": (1100.0, float("inf")),
}

DEMAND_CATEGORY_COLORS = {
    "Low": "#38bdf8",       # Sky cyan
    "Moderate": "#34d399",  # Mint green
    "High": "#fbbf24",      # Amber yellow
    "Very High": "#f87171", # Coral red
}

# Machine Learning Split and Tuning
RANDOM_STATE = 42
CHRONOLOGICAL_SPLIT_RATIO = 0.8  # First 80% for train/validation, final 20% for untouched test
CV_SPLITS = 4

# Metric descriptions
METRIC_DEFINITIONS = {
    "MAE": "Mean Absolute Error: Average absolute magnitude of rental prediction errors in bikes.",
    "MSE": "Mean Squared Error: Average squared error, penalizing larger deviations heavily.",
    "RMSE": "Root Mean Squared Error: Square root of MSE, in the same units as bike rentals (sensitive to surges).",
    "R2": "Coefficient of Determination (R²): Proportion of demand variance explained by the model (1.0 is perfect).",
    "WAPE": "Weighted Absolute Percentage Error: Total error sum divided by total actual demand. Robust against zero-division.",
    "sMAPE": "Symmetric Mean Absolute Percentage Error: Bounded percentage error metric robust to scale variations.",
    "Explained Variance": "Explained Variance Score: Measures the proportion of variance explained by model errors.",
}

# UI Theme styling constants
THEME_CONFIG = {
    "primaryColor": "#00d2ff",
    "backgroundColor": "#0b0f19",
    "secondaryBackgroundColor": "#131b2e",
    "cardBackgroundColor": "#1a233a",
    "textColor": "#e2e8f0",
    "accentColor": "#38bdf8",
    "font": "sans-serif",
}

