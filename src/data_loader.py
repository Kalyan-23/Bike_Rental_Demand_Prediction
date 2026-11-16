"""Data loader module for Seoul Bike Sharing dataset.

Provides robust reading, encoding detection, and schema validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

import config

logger = logging.getLogger(__name__)


def load_raw_data(
    filepath: Optional[Path | str] = None,
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    """Load the raw Seoul Bike dataset with robust encoding detection.

    Args:
        filepath: Optional path to the CSV file. Defaults to DEFAULT_DATASET_PATH.
        encoding: Optional specific encoding to use. If None, candidate encodings are tried.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If essential columns are missing or file cannot be parsed.
    """
    path = Path(filepath) if filepath else config.DEFAULT_DATASET_PATH
    if not path.exists():
        # Fallback check in Project/ folder if root data/ is not yet populated
        fallback = config.BASE_DIR / "Project" / "SeoulBikeData.csv"
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(
                f"Seoul Bike dataset not found at '{path}'. "
                f"Please ensure SeoulBikeData.csv is placed in '{config.DATA_DIR}'."
            )

    encodings_to_try = [encoding] if encoding else ["latin-1", "cp1252", "iso-8859-1", "utf-8"]
    df: Optional[pd.DataFrame] = None
    last_error: Optional[Exception] = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path, encoding=enc)
            logger.info("Successfully loaded dataset from %s using encoding %s", path, enc)
            break
        except (UnicodeDecodeError, OSError, pd.errors.ParserError) as exc:
            last_error = exc
            continue

    if df is None:
        raise ValueError(f"Failed to decode dataset at '{path}'. Error: {last_error}")

    # Standardize column names (trim whitespace, normalize degree signs if needed)
    df.columns = [col.strip() for col in df.columns]

    # Normalize temperature degree signs if needed
    rename_dict = {}
    for col in df.columns:
        if "Temperature" in col and "Dew" not in col and col != "Temperature(°C)":
            rename_dict[col] = "Temperature(°C)"
        elif "Dew point temperature" in col and col != "Dew point temperature(°C)":
            rename_dict[col] = "Dew point temperature(°C)"
    if rename_dict:
        df = df.rename(columns=rename_dict)

    # Validate essential columns
    required_cols = {config.DATE_COL, config.TARGET_COL, "Hour", "Seasons", "Holiday", "Functioning Day"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    return df


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Validate dataset structure, types, missing values, and integrity.

    Args:
        df: Pandas DataFrame to validate.

    Returns:
        dict containing validation details.
    """
    total_rows = len(df)
    missing_by_col = df.isna().sum().to_dict()
    total_missing = sum(missing_by_col.values())
    duplicate_rows = int(df.duplicated().sum())

    date_parseable = False
    date_min = None
    date_max = None
    if config.DATE_COL in df.columns:
        try:
            parsed_dates = pd.to_datetime(df[config.DATE_COL], dayfirst=True, errors="coerce")
            valid_dates = parsed_dates.dropna()
            if not valid_dates.empty:
                date_parseable = True
                date_min = str(valid_dates.min().date())
                date_max = str(valid_dates.max().date())
        except Exception:
            pass

    has_target = config.TARGET_COL in df.columns
    target_stats = {}
    if has_target:
        target_series = pd.to_numeric(df[config.TARGET_COL], errors="coerce")
        target_stats = {
            "min": float(target_series.min()) if not target_series.empty else None,
            "max": float(target_series.max()) if not target_series.empty else None,
            "mean": float(target_series.mean()) if not target_series.empty else None,
            "median": float(target_series.median()) if not target_series.empty else None,
            "negative_count": int((target_series < 0).sum()),
        }

    return {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "columns": df.columns.tolist(),
        "total_missing": total_missing,
        "missing_by_column": missing_by_col,
        "duplicate_rows": duplicate_rows,
        "date_parseable": date_parseable,
        "date_range": {"min": date_min, "max": date_max},
        "target_stats": target_stats,
        "is_valid": total_rows > 0 and date_parseable and has_target and target_stats.get("negative_count", 0) == 0,
    }

