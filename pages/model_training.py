"""Interactive Model Training Console for Seoul Bike Sharing application."""

from __future__ import annotations

import shutil
import time
import streamlit as st
import pandas as pd
import numpy as np

import config
from src.feature_engineering import build_full_features
from src.model_training import (
    get_model_candidates,
    save_model_bundle,
    train_models_pipeline,
)
from src.preprocessing import split_data_chronologically


def render_model_training_page(cleaned_df: pd.DataFrame) -> None:
    """Render interactive training console with model selection, tuning, and safeguards."""
    st.markdown("<h1 style='margin-bottom:0;'>Model Training & Retraining Console</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94a3b8; font-size:1.05rem; margin-top:4px;'>Configure algorithms, select validation regimes, trigger training runs, and safely persist models.</p>",
        unsafe_allow_html=True,
    )

    all_available = list(get_model_candidates().keys())

    # Training Configuration Form
    with st.expander("Training Configuration & Algorithm Selection", expanded=True):
        col_m1, col_m2 = st.columns([1.2, 0.8])

        with col_m1:
            st.markdown("#### Select Algorithms to Benchmark")
            selected_models = st.multiselect(
                "Algorithms",
                options=all_available,
                default=["LightGBM", "Extra Trees", "Random Forest", "HistGradientBoosting", "Ridge Regression"],
                help="Select one or more regression algorithms to train and evaluate.",
            )

        with col_m2:
            st.markdown("#### Operational Paradigm & Split")
            train_mode = st.radio(
                "Feature & Prediction Mode",
                ["Mode 1: Weather & Time Scenario (Exogenous Planning)", "Mode 2: Historical Lag-Assisted (Auto-Regressive Dispatch)"],
                index=0,
            )
            tune_toggle = st.checkbox("Enable TimeSeriesSplit Hyperparameter Tuning (RandomizedSearchCV)", value=False)
            split_ratio = st.slider("Chronological Train Split Ratio", 0.60, 0.85, 0.80, 0.05)

    if not selected_models:
        st.warning("Please select at least one algorithm to begin training.")
        return

    # Trigger Training Button
    start_train_btn = st.button("Start Model Training & Benchmarking", type="primary", use_container_width=True)

    if start_train_btn:
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(pct: float, text: str):
            progress_bar.progress(pct)
            status_text.markdown(f"**Status:** {text}")

        include_lags = ("Mode 2" in train_mode)
        status_text.markdown("**Status:** Preparing features and temporal splits...")

        try:
            # 1. Feature Engineering
            featured_df = build_full_features(cleaned_df, include_lags=include_lags)

            # 2. Chronological Split
            train_df, test_df = split_data_chronologically(featured_df, split_ratio=split_ratio)

            num_cols = config.ENGINEERED_NUMERICAL_COLS.copy()
            if include_lags:
                num_cols.extend(config.LAG_FEATURES)
            cat_cols = config.ENGINEERED_CATEGORICAL_COLS.copy()

            num_cols = [c for c in num_cols if c in train_df.columns]
            cat_cols = [c for c in cat_cols if c in train_df.columns]

            # 3. Training Execution
            (
                new_comparison_df,
                new_fitted_pipelines,
                new_best_name,
                new_test_predictions_df,
                new_importance_df,
            ) = train_models_pipeline(
                train_df=train_df,
                test_df=test_df,
                numerical_features=num_cols,
                categorical_features=cat_cols,
                model_names=selected_models,
                tune_hyperparameters=tune_toggle,
                progress_callback=update_progress,
            )

            progress_bar.progress(1.0)
            status_text.success("Training and benchmarking completed successfully!")

            # Store in session state for review
            st.session_state.training_results = {
                "comparison": new_comparison_df,
                "best_name": new_best_name,
                "best_pipe": new_fitted_pipelines[new_best_name],
                "test_preds": new_test_predictions_df,
                "importance": new_importance_df,
                "mode": "Mode 2" if include_lags else "Mode 1",
                "num_cols": num_cols,
                "cat_cols": cat_cols,
            }

        except Exception as e:
            status_text.error(f"Training run failed: {e}")
            return

    # Review & Save Training Results
    if "training_results" in st.session_state:
        res = st.session_state.training_results
        st.markdown("<hr style='margin:20px 0; border-color:#1f293d;'>", unsafe_allow_html=True)
        st.markdown("### Benchmarking Results from Current Run")

        st.dataframe(
            res["comparison"][["Model", "MAE", "RMSE", "R2", "WAPE", "sMAPE", "Training_Time_s", "Hyperparameters"]].style.format(
                {"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.4f}", "WAPE": "{:.2f}%", "sMAPE": "{:.2f}%", "Training_Time_s": "{:.2f}s"}
            ),
            use_container_width=True,
        )

        best_model = res["best_name"]
        best_rmse = res["comparison"].iloc[0]["RMSE"]
        st.success(f"Best Model from this run: **{best_model}** (Test RMSE: {best_rmse:.2f})")

        # Persistence with Safeguard
        st.markdown("#### Model Persistence & Deployment Safeguard")
        st.info("Saving this model will update `models/best_model.joblib`. An automated backup of any existing model is preserved automatically.")

        if st.button(f"Deploy & Save {best_model} as Active Production Model", key="save_trained_btn"):
            try:
                # Backup existing model if present
                if config.BEST_MODEL_PATH.exists():
                    backup_path = config.MODELS_DIR / "best_model_backup.joblib"
                    shutil.copy2(config.BEST_MODEL_PATH, backup_path)

                best_metrics = res["comparison"].iloc[0].to_dict()
                save_model_bundle(
                    pipeline=res["best_pipe"],
                    model_name=best_model,
                    metrics=best_metrics,
                    mode=f"Mode: {res['mode'].upper()}",
                    numerical_features=res["num_cols"],
                    categorical_features=res["cat_cols"],
                )
                # Update CSV outputs
                res["comparison"].to_csv(config.MODEL_COMPARISON_PATH, index=False)
                res["test_preds"].to_csv(config.TEST_PREDICTIONS_PATH, index=False)
                res["importance"].to_csv(config.FEATURE_IMPORTANCE_PATH, index=False)

                st.success(f"Model **{best_model}** deployed successfully! All model comparison tables and predictions have been updated.")
            except Exception as save_err:
                st.error(f"Failed to persist model: {save_err}")
