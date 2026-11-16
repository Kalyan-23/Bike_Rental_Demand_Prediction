"""End-to-end Seoul Bike Sharing demand prediction and intelligence workflow.

Runs complete pipeline: data validation, cleaning, feature engineering,
baseline replication, chronological training/evaluation, and Power BI artifact generation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import config
from src.data_loader import load_raw_data, validate_dataset
from src.evaluation import get_metric_explanations
from src.feature_engineering import build_full_features
from src.model_training import (
    extract_feature_importance,
    run_baseline_comparison,
    save_model_bundle,
    train_models_pipeline,
)
from src.preprocessing import clean_raw_data, split_data_chronologically
from src.visualization import export_static_visualizations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=config.DEFAULT_DATASET_PATH,
        help="Path to input SeoulBikeData.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.OUTPUT_DIR,
        help="Directory to save generated outputs",
    )
    parser.add_argument(
        "--mode",
        choices=["mode1", "mode2", "both"],
        default="mode1",
        help="Prediction mode: mode1 (Weather & Calendar scenario) or mode2 (Lag-assisted) or both",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter optimization via RandomizedSearchCV with TimeSeriesSplit",
    )
    return parser.parse_args()


def generate_analysis_summary(
    df_cleaned: pd.DataFrame,
    comparison_df: pd.DataFrame,
    cleaning_meta: dict[str, Any],
    best_model_name: str,
) -> pd.DataFrame:
    """Compile business KPIs and feature correlations into Analysis_Summary.csv."""
    target_series = df_cleaned[config.TARGET_COL]

    hourly = df_cleaned.groupby("Hour")[config.TARGET_COL].mean()
    season = df_cleaned.groupby("Seasons")[config.TARGET_COL].mean()

    # Numerical correlations with target
    num_df = df_cleaned.select_dtypes(include=[np.number])
    correlations = num_df.corr()[config.TARGET_COL].drop(config.TARGET_COL).sort_values(ascending=False).head(10)

    best_row = comparison_df.iloc[0]

    summary_rows = [
        {"Metric": "Total rentals (Full Year)", "Value": float(target_series.sum())},
        {"Metric": "Average rentals per hour", "Value": round(float(target_series.mean()), 2)},
        {"Metric": "Median rentals per hour", "Value": round(float(target_series.median()), 2)},
        {"Metric": "Maximum single-hour rentals", "Value": float(target_series.max())},
        {"Metric": "Minimum single-hour rentals", "Value": float(target_series.min())},
        {"Metric": "Peak rental hour", "Value": int(hourly.idxmax())},
        {"Metric": "Lowest rental hour", "Value": int(hourly.idxmin())},
        {"Metric": "Highest demand season", "Value": str(season.idxmax())},
        {"Metric": "Lowest demand season", "Value": str(season.idxmin())},
        {"Metric": "Cleaned rows processed", "Value": cleaning_meta["final_rows"]},
        {"Metric": "Best model (Chronological Split)", "Value": best_model_name},
        {"Metric": "Best model test MAE", "Value": best_row["MAE"]},
        {"Metric": "Best model test RMSE", "Value": best_row["RMSE"]},
        {"Metric": "Best model test R2", "Value": best_row["R2"]},
        {"Metric": "Best model test WAPE (%)", "Value": best_row.get("WAPE", 0.0)},
        {"Metric": "Best model test sMAPE (%)", "Value": best_row.get("sMAPE", 0.0)},
    ]

    for feat_name, corr_val in correlations.items():
        summary_rows.append({"Metric": f"Correlation: {feat_name}", "Value": round(float(corr_val), 4)})

    return pd.DataFrame(summary_rows)


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    mode: str = "mode1",
    tune_hyperparameters: bool = False,
) -> None:
    """Execute the full upgraded pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SEOUL BIKE SHARING DEMAND PREDICTION & INTELLIGENT ANALYTICS")
    print("=" * 70)

    # 1. Load data
    logger.info("Loading raw dataset from %s...", input_path)
    raw_df = load_raw_data(input_path)
    val_report = validate_dataset(raw_df)
    logger.info("Raw dataset validated: %d rows, %d columns.", len(raw_df), len(raw_df.columns))

    # 2. Clean data
    logger.info("Performing data cleaning and outlier audit...")
    cleaned_df, cleaning_meta = clean_raw_data(raw_df)
    
    # Enrich cleaned dataset with calendar and weather features (preserving original SeoulBike_Cleaned.csv schema)
    from src.feature_engineering import add_calendar_features, add_weather_features
    cleaned_df = add_calendar_features(cleaned_df)
    cleaned_df = add_weather_features(cleaned_df)
    cleaned_df.to_csv(output_dir / "SeoulBike_Cleaned.csv", index=False)

    # 3. Descriptive statistics
    descriptive_df = cleaned_df.describe(include="all").transpose().reset_index().rename(columns={"index": "Column"})
    descriptive_df.to_csv(output_dir / "Descriptive_Statistics.csv", index=False)

    # 4. Run historical baseline comparison (reproduced on random 80/20 split)
    logger.info("Replicating historical baseline comparison on random split...")
    baseline_df = run_baseline_comparison(cleaned_df)
    baseline_df.to_csv(output_dir / "Baseline_Model_Comparison.csv", index=False)
    logger.info("Historical baseline results saved to Baseline_Model_Comparison.csv")

    # 5. Feature Engineering
    include_lags = (mode in ["mode2", "both"])
    logger.info("Engineering features (Mode: %s, include_lags: %s)...", mode, include_lags)
    featured_df = build_full_features(cleaned_df, include_lags=include_lags)

    # 6. Chronological Train/Test Split (Strictly non-leaking)
    logger.info("Executing strict chronological split (80%% train, 20%% test)...")
    train_df, test_df = split_data_chronologically(featured_df, split_ratio=config.CHRONOLOGICAL_SPLIT_RATIO)
    logger.info("Train set: %d rows (%s to %s)", len(train_df), train_df[config.DATE_COL].min(), train_df[config.DATE_COL].max())
    logger.info("Test set:  %d rows (%s to %s)", len(test_df), test_df[config.DATE_COL].min(), test_df[config.DATE_COL].max())

    num_cols = config.ENGINEERED_NUMERICAL_COLS.copy()
    if include_lags:
        num_cols.extend(config.LAG_FEATURES)
    cat_cols = config.ENGINEERED_CATEGORICAL_COLS.copy()

    # Filter to existing columns
    num_cols = [c for c in num_cols if c in train_df.columns]
    cat_cols = [c for c in cat_cols if c in train_df.columns]

    # 7. Model Training & Evaluation
    logger.info("Training candidate regression models on chronological partition...")
    (
        comparison_df,
        fitted_pipelines,
        best_model_name,
        test_predictions_df,
        feature_importance_df,
    ) = train_models_pipeline(
        train_df=train_df,
        test_df=test_df,
        numerical_features=num_cols,
        categorical_features=cat_cols,
        tune_hyperparameters=tune_hyperparameters,
    )

    comparison_df.to_csv(output_dir / "Model_Comparison.csv", index=False)
    test_predictions_df.to_csv(output_dir / "Test_Predictions.csv", index=False)
    feature_importance_df.to_csv(output_dir / "Feature_Importance.csv", index=False)

    # 8. Save best model bundle
    best_pipe = fitted_pipelines[best_model_name]
    best_metrics = comparison_df.iloc[0].to_dict()
    saved_model_path = save_model_bundle(
        pipeline=best_pipe,
        model_name=best_model_name,
        metrics=best_metrics,
        mode=f"Mode: {mode.upper()}",
        numerical_features=num_cols,
        categorical_features=cat_cols,
    )
    # Also save a copy directly in output for backwards compatibility
    joblib.dump(best_pipe, output_dir / "best_model.joblib")

    # 9. Generate Power BI Updated CSV
    power_bi_df = featured_df.copy()
    power_bi_df[config.DATE_COL] = pd.to_datetime(power_bi_df[config.DATE_COL]).dt.strftime("%Y-%m-%d")
    power_bi_df["Data Split"] = "Train"
    power_bi_df.loc[test_predictions_df["RowIndex"], "Data Split"] = "Test"
    power_bi_df["Predicted_Demand"] = np.nan
    power_bi_df.loc[test_predictions_df["RowIndex"], "Predicted_Demand"] = test_predictions_df["Predicted"].values
    power_bi_df["Prediction_Residual"] = np.nan
    power_bi_df.loc[test_predictions_df["RowIndex"], "Prediction_Residual"] = test_predictions_df["Residual"].values
    power_bi_df.to_csv(output_dir / "SeoulBike_Updated_For_PowerBI.csv", index=False)

    # 10. Generate Analysis Summary
    summary_df = generate_analysis_summary(cleaned_df, comparison_df, cleaning_meta, best_model_name)
    summary_df.to_csv(output_dir / "Analysis_Summary.csv", index=False)

    # 11. Run Metadata
    metadata = {
        "input_dataset": str(input_path),
        "total_rows": len(cleaned_df),
        "split_strategy": "Chronological (First 80% Train, Final 20% Unseen Test)",
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "prediction_mode": mode,
        "best_model": best_model_name,
        "best_model_metrics": best_metrics,
        "cleaning_audit": cleaning_meta,
        "artifacts_generated": [
            "Model_Comparison.csv",
            "Baseline_Model_Comparison.csv",
            "Analysis_Summary.csv",
            "Feature_Importance.csv",
            "Test_Predictions.csv",
            "SeoulBike_Cleaned.csv",
            "SeoulBike_Updated_For_PowerBI.csv",
            "Descriptive_Statistics.csv",
            "best_model.joblib",
        ],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    # 12. Export static PNG figures for Power BI and documentation
    logger.info("Exporting static figures to %s...", vis_dir)
    export_static_visualizations(cleaned_df, test_predictions_df, feature_importance_df, vis_dir)

    # Console output report
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Best Chronological Model: {best_model_name}")
    print(f"Test MAE:  {best_metrics['MAE']}")
    print(f"Test RMSE: {best_metrics['RMSE']}")
    print(f"Test R²:   {best_metrics['R2']}")
    print(f"Test WAPE: {best_metrics.get('WAPE', 0.0)}%")
    print(f"Test sMAPE:{best_metrics.get('sMAPE', 0.0)}%")
    print("=" * 70)
    print("\nCHRONOLOGICAL MODEL COMPARISON:")
    print(comparison_df[["Model", "MAE", "RMSE", "R2", "WAPE", "sMAPE", "Training_Time_s"]].to_string(index=False))
    print("\nHISTORICAL BASELINE REPLICATION (RANDOM 80/20 SPLIT):")
    print(baseline_df[["Model", "MAE", "RMSE", "R2", "WAPE", "Split_Type"]].to_string(index=False))
    print(f"\nAll artifacts written to: {output_dir.resolve()}")
    print("=" * 70)


def main() -> None:
    """CLI entry point."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    args = parse_args()
    run_pipeline(
        input_path=args.input.resolve(),
        output_dir=args.output.resolve(),
        mode=args.mode,
        tune_hyperparameters=args.tune,
    )


if __name__ == "__main__":
    main()

