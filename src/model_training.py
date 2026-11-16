"""Model training, hyperparameter optimization, and pipeline serialization module.

Supports chronological validation with TimeSeriesSplit, ensemble algorithms
(LightGBM, CatBoost, XGBoost, Extra Trees, Random Forest, HistGradientBoosting,
GradientBoosting, Ridge, DecisionTree), leakage-free preprocessing, and
honest baseline replication.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

import config
from src.evaluation import evaluate_predictions
from src.preprocessing import build_preprocessor_pipeline

logger = logging.getLogger(__name__)


def get_model_candidates(random_state: int = config.RANDOM_STATE) -> dict[str, Any]:
    """Retrieve initialized candidate regression algorithms.

    Args:
        random_state: Reproducibility seed.

    Returns:
        dict[str, Any]: Dictionary mapping model names to estimator instances.
    """
    return {
        "LightGBM": LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            verbose=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=350,
            learning_rate=0.05,
            depth=6,
            random_seed=random_state,
            verbose=0,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.05,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            random_state=random_state,
        ),
        "Ridge Regression": Ridge(
            alpha=10.0,
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=12,
            min_samples_leaf=5,
            random_state=random_state,
        ),
    }


def get_tuning_distributions() -> dict[str, dict[str, list[Any]]]:
    """Define parameter search distributions for hyperparameter optimization."""
    return {
        "LightGBM": {
            "model__n_estimators": [150, 250, 350],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__num_leaves": [20, 31, 50],
            "model__subsample": [0.8, 1.0],
        },
        "Extra Trees": {
            "model__n_estimators": [150, 250, 350],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 1.0],
        },
        "Random Forest": {
            "model__n_estimators": [150, 250, 350],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 1.0],
        },
        "XGBoost": {
            "model__n_estimators": [150, 250],
            "model__max_depth": [4, 6, 8],
            "model__learning_rate": [0.03, 0.05, 0.1],
        },
    }


def post_process_predictions(
    raw_predictions: np.ndarray,
    data_df: pd.DataFrame,
) -> np.ndarray:
    """Apply domain rules to model predictions.

    1. Rental counts cannot be negative (clip at 0).
    2. When Functioning Day == 'No', rental system is closed (enforce 0).

    Args:
        raw_predictions: NumPy array of continuous model predictions.
        data_df: Corresponding DataFrame with 'Functioning Day' column.

    Returns:
        np.ndarray: Post-processed valid rental predictions.
    """
    preds = np.clip(raw_predictions, 0.0, None)
    if "Functioning Day" in data_df.columns:
        is_closed = (data_df["Functioning Day"] == "No").to_numpy()
        preds[is_closed] = 0.0
    return preds


def train_models_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numerical_features: list[str],
    categorical_features: list[str],
    model_names: Optional[list[str]] = None,
    tune_hyperparameters: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[pd.DataFrame, dict[str, Pipeline], str, pd.DataFrame, pd.DataFrame]:
    """Train multiple regression models on training data, evaluate on untouched test data.

    Args:
        train_df: Chronologically earlier training observations.
        test_df: Chronologically later untouched test observations.
        numerical_features: List of numerical predictor names.
        categorical_features: List of categorical predictor names.
        model_names: Optional subset of model names to train. If None, trains standard suite.
        tune_hyperparameters: Whether to run RandomizedSearchCV with TimeSeriesSplit on train_df.
        progress_callback: Optional callback func(pct_complete, status_text) for UI progress bars.

    Returns:
        Tuple:
            - comparison_df: Summary table with all metrics sorted by RMSE.
            - fitted_pipelines: Dict of name -> fitted sklearn Pipeline.
            - best_model_name: Name of model achieving lowest test RMSE.
            - test_predictions_df: Detailed row-level actual vs predicted test values.
            - feature_importance_df: Ranked feature importances for best model.
    """
    all_candidates = get_model_candidates()
    models_to_run = (
        {k: v for k, v in all_candidates.items() if k in model_names}
        if model_names
        else all_candidates
    )

    tuning_dist = get_tuning_distributions() if tune_hyperparameters else {}

    X_train = train_df[numerical_features + categorical_features].copy()
    y_train = train_df[config.TARGET_COL].to_numpy(dtype=float)

    X_test = test_df[numerical_features + categorical_features].copy()
    y_test = test_df[config.TARGET_COL].to_numpy(dtype=float)

    comparison_results: list[dict[str, Any]] = []
    fitted_pipelines: dict[str, Pipeline] = {}
    test_predictions_by_model: dict[str, np.ndarray] = {}

    total_models = len(models_to_run)
    for idx, (name, estimator) in enumerate(models_to_run.items(), start=1):
        if progress_callback:
            progress_callback(
                (idx - 1) / total_models,
                f"Training {name} ({idx}/{total_models})...",
            )

        # Build clean preprocessor fitted solely on training partition
        scale_numeric = (name == "Ridge Regression")
        preprocessor = build_preprocessor_pipeline(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            scale_numeric=scale_numeric,
        )

        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])

        best_params_str = "Default"
        start_time = time.time()

        if tune_hyperparameters and name in tuning_dist:
            tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS)
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=tuning_dist[name],
                n_iter=6,
                cv=tscv,
                scoring="neg_root_mean_squared_error",
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            pipeline = search.best_estimator_
            best_params_str = str(search.best_params_)
        else:
            pipeline.fit(X_train, y_train)

        elapsed_time = round(time.time() - start_time, 2)

        # Inference on untouched chronological test set
        raw_preds = pipeline.predict(X_test)
        preds = post_process_predictions(raw_preds, test_df)

        metrics = evaluate_predictions(y_test, preds)
        metrics["Model"] = name
        metrics["Training_Time_s"] = elapsed_time
        metrics["Hyperparameters"] = best_params_str

        comparison_results.append(metrics)
        fitted_pipelines[name] = pipeline
        test_predictions_by_model[name] = preds

    if progress_callback:
        progress_callback(1.0, "Model evaluation complete!")

    # Sort comparison table by RMSE ascending (best model first)
    comparison_df = pd.DataFrame(comparison_results).sort_values("RMSE").reset_index(drop=True)
    best_model_name = str(comparison_df.iloc[0]["Model"])
    best_pipeline = fitted_pipelines[best_model_name]
    best_preds = test_predictions_by_model[best_model_name]

    # Build test predictions DataFrame
    test_predictions_df = pd.DataFrame(
        {
            "RowIndex": test_df.index,
            "Date": test_df[config.DATE_COL].astype(str),
            "Hour": test_df["Hour"],
            "Actual": y_test,
            "Predicted": np.round(best_preds, 1),
            "Residual": np.round(y_test - best_preds, 1),
            "Model": best_model_name,
            "Data Split": "Test",
        }
    )

    # Extract Feature Importance
    feature_importance_df = extract_feature_importance(best_pipeline)

    return (
        comparison_df,
        fitted_pipelines,
        best_model_name,
        test_predictions_df,
        feature_importance_df,
    )


def extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Extract and rank feature importances or coefficients from a fitted pipeline.

    Args:
        pipeline: Fitted scikit-learn Pipeline with 'preprocessor' and 'model' steps.

    Returns:
        pd.DataFrame: DataFrame containing 'Feature' and 'Importance' ranked descending.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # Extract transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    importances: Optional[np.ndarray] = None
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importances = np.abs(np.asarray(model.coef_, dtype=float))

    if importances is not None and len(importances) == len(feature_names):
        # Clean up feature name prefixes (e.g., 'num__', 'cat__')
        clean_names = [name.split("__")[-1] for name in feature_names]
        df_imp = pd.DataFrame({"Feature": clean_names, "Importance": importances})
        # Normalize to percentage sum for intuitive display
        total_imp = df_imp["Importance"].sum()
        if total_imp > 0:
            df_imp["Relative_Importance_Pct"] = np.round((df_imp["Importance"] / total_imp) * 100, 2)
        df_imp = df_imp.sort_values("Importance", ascending=False).reset_index(drop=True)
        return df_imp

    return pd.DataFrame({"Feature": [], "Importance": []})


def run_baseline_comparison(
    cleaned_df: pd.DataFrame,
) -> pd.DataFrame:
    """Replicate the exact historical baseline model comparison.

    Uses the original random 80/20 train_test_split and original feature set to
    reproduce the baseline metrics honestly, saving them separately.

    Args:
        cleaned_df: Cleaned dataset.

    Returns:
        pd.DataFrame: Baseline model comparison metrics.
    """
    from sklearn.model_selection import train_test_split

    from src.feature_engineering import build_full_features

    df_feat = build_full_features(cleaned_df, include_lags=False)

    legacy_features = [
        "Hour",
        "Temperature(°C)",
        "Humidity(%)",
        "Wind speed (m/s)",
        "Visibility (10m)",
        "Dew point temperature(°C)",
        "Solar Radiation (MJ/m2)",
        "Rainfall(mm)",
        "Snowfall (cm)",
        "Year",
        "Month",
        "Day",
        "DayOfWeek",
        "IsWeekend",
        "IsPeakHour",
        "DayOfYear",
    ]
    legacy_cats = ["Seasons", "Holiday", "Functioning Day", "Weather Category"]

    X = df_feat[legacy_features + legacy_cats]
    y = df_feat[config.TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE
    )

    baseline_models = {
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=config.RANDOM_STATE, n_jobs=-1
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=config.RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.05, max_depth=3, random_state=config.RANDOM_STATE
        ),
        "Ridge Regression": Ridge(alpha=10.0, random_state=config.RANDOM_STATE),
    }

    baseline_results = []
    for name, estimator in baseline_models.items():
        preprocessor = build_preprocessor_pipeline(
            numerical_features=legacy_features,
            categorical_features=legacy_cats,
            scale_numeric=(name == "Ridge Regression"),
        )
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = evaluate_predictions(y_test, preds)
        metrics["Model"] = name
        metrics["Split_Type"] = "Random 80/20 (Historical Baseline)"
        baseline_results.append(metrics)

    df_base = pd.DataFrame(baseline_results).sort_values("RMSE").reset_index(drop=True)
    return df_base


def save_model_bundle(
    pipeline: Pipeline,
    model_name: str,
    metrics: dict[str, Any],
    mode: str,
    numerical_features: list[str],
    categorical_features: list[str],
    save_path: Optional[str | Path] = None,
) -> Path:
    """Save the fitted model pipeline along with training metadata to disk.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        model_name: Name of algorithm.
        metrics: Test set evaluation metrics.
        mode: Prediction mode ('Mode 1: Weather Scenario' or 'Mode 2: Lag-Assisted').
        numerical_features: List of numerical features expected by preprocessor.
        categorical_features: List of categorical features expected by preprocessor.
        save_path: Destination file path. Defaults to config.BEST_MODEL_PATH.

    Returns:
        Path: Location of saved joblib bundle.
    """
    path = Path(save_path) if save_path else config.BEST_MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "pipeline": pipeline,
        "model_name": model_name,
        "metrics": metrics,
        "mode": mode,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "demand_thresholds": config.DEMAND_THRESHOLDS,
        "saved_timestamp": pd.Timestamp.now().isoformat(),
    }

    joblib.dump(bundle, path)
    logger.info("Saved fitted model bundle to %s", path)
    return path


def load_model_bundle(load_path: Optional[str | Path] = None) -> dict[str, Any]:
    """Load the persisted model bundle and metadata from disk.

    Args:
        load_path: Path to serialized joblib file. Defaults to config.BEST_MODEL_PATH.

    Returns:
        dict[str, Any]: Model bundle containing pipeline, metadata, and features.
    """
    path = Path(load_path) if load_path else config.BEST_MODEL_PATH
    if not path.exists():
        # Fallback check inside Project/output/best_model.joblib
        fallback = config.BASE_DIR / "Project" / "output" / "best_model.joblib"
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(
                f"Model bundle not found at '{path}'. Please train a model first using main.py or the UI."
            )

    bundle = joblib.load(path)
    # If legacy pipeline saved directly without bundle wrapper, wrap it
    if isinstance(bundle, Pipeline):
        bundle = {
            "pipeline": bundle,
            "model_name": "Fitted Pipeline",
            "metrics": {},
            "mode": "Mode 1: Weather Scenario",
            "numerical_features": config.ENGINEERED_NUMERICAL_COLS,
            "categorical_features": config.ENGINEERED_CATEGORICAL_COLS,
            "demand_thresholds": config.DEMAND_THRESHOLDS,
            "saved_timestamp": None,
        }
    return bundle

