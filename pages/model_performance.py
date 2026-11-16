"""Model Performance & Evaluation Dashboard for Seoul Bike Sharing application."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

import config
from src.evaluation import get_metric_explanations
from src.visualization import (
    plot_model_comparison_bar,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_residual_distribution,
)


def render_model_performance_page(
    comparison_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    test_pred_df: pd.DataFrame,
    metadata: dict,
) -> None:
    """Render comprehensive ML model performance benchmarks and residual diagnostics."""
    st.markdown("<h1 style='margin-bottom:0;'>Model Performance & Diagnostic Evaluation</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94a3b8; font-size:1.05rem; margin-top:4px;'>Honest chronological benchmarking against untouched future observations and historical baseline comparison.</p>",
        unsafe_allow_html=True,
    )

    best_model_name = metadata.get("best_model", comparison_df.iloc[0]["Model"] if not comparison_df.empty else "Extra Trees")
    best_row = comparison_df.iloc[0] if not comparison_df.empty else {}

    # Top KPI row for best model
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Best Model", best_model_name)
    with c2:
        st.metric("Test MAE", f"{best_row.get('MAE', 0.0):.2f}")
    with c3:
        st.metric("Test RMSE", f"{best_row.get('RMSE', 0.0):.2f}")
    with c4:
        st.metric("Test R²", f"{best_row.get('R2', 0.0):.4f}")
    with c5:
        st.metric("Test WAPE", f"{best_row.get('WAPE', 0.0):.2f}%")

    st.markdown("<hr style='margin:16px 0 20px 0; border-color:#1f293d;'>", unsafe_allow_html=True)

    perf_tabs = st.tabs(
        [
            "Chronological Benchmark Comparison",
            "Residual Diagnostics & Fits",
            "Chronological vs Baseline Analysis",
            "Metric Formal Definitions",
        ]
    )

    # Tab 1: Chronological Comparison
    with perf_tabs[0]:
        st.markdown("### Chronological Test Set Evaluation Leaderboard")
        st.caption(
            "Models were trained on earlier chronological data (80%) and evaluated strictly on "
            "the subsequent untouched chronological test observations (20%)."
        )

        display_cols = ["Model", "MAE", "RMSE", "R2", "WAPE", "sMAPE", "Training_Time_s"]
        avail_cols = [c for c in display_cols if c in comparison_df.columns]
        st.dataframe(
            comparison_df[avail_cols].style.format(
                {"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.4f}", "WAPE": "{:.2f}%", "sMAPE": "{:.2f}%", "Training_Time_s": "{:.2f}s"}
            ),
            use_container_width=True,
        )

        # Download Model Comparison CSV button
        if config.MODEL_COMPARISON_PATH.exists():
            st.download_button(
                label="📥 Download Model_Comparison.csv",
                data=config.MODEL_COMPARISON_PATH.read_bytes(),
                file_name="Model_Comparison.csv",
                mime="text/csv",
                key="dl_model_comp_csv",
            )

        st.markdown("#### Interactive Metric Comparison Chart")
        selected_metric = st.selectbox("Select Evaluation Metric for Visual Ranking", ["RMSE", "MAE", "R2", "WAPE", "sMAPE"], index=0)
        st.plotly_chart(plot_model_comparison_bar(comparison_df, selected_metric), use_container_width=True)

    # Tab 2: Residual Diagnostics
    with perf_tabs[1]:
        st.markdown(f"### Residual Diagnostics: {best_model_name}")
        st.caption("Detailed error analysis on test observations (Residual = Actual - Predicted).")

        if not test_pred_df.empty:
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.plotly_chart(plot_actual_vs_predicted(test_pred_df, best_model_name), use_container_width=True)
            with r_col2:
                st.plotly_chart(plot_residuals(test_pred_df, best_model_name), use_container_width=True)

            st.plotly_chart(plot_residual_distribution(test_pred_df, best_model_name), use_container_width=True)

            # Error breakdown statistics
            residuals = test_pred_df["Residual"].to_numpy()
            abs_err = np.abs(residuals)
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.metric("Mean Residual", f"{np.mean(residuals):.2f}", help="Close to 0 indicates unbiased predictions.")
            with s2:
                st.metric("Std Dev Residuals", f"{np.std(residuals):.2f}")
            with s3:
                st.metric("Median Absolute Error", f"{np.median(abs_err):.2f}")
            with s4:
                st.metric("Max Underprediction", f"{np.max(residuals):.1f}")

            # Download Test Predictions button
            if config.TEST_PREDICTIONS_PATH.exists():
                st.download_button(
                    label="📥 Download Test_Predictions.csv (Row-by-Row Actual vs Predicted)",
                    data=config.TEST_PREDICTIONS_PATH.read_bytes(),
                    file_name="Test_Predictions.csv",
                    mime="text/csv",
                    key="dl_test_preds_csv",
                )
        else:
            st.info("Test predictions DataFrame is not available. Please train models via main.py or the UI.")

    # Tab 3: Chronological vs Baseline
    with perf_tabs[2]:
        st.markdown("### Honest Methodological Comparison: Random Split vs Chronological Split")

        st.warning(
            "**Academic Integrity & Methodology Insight:**\n\n"
            "The historical baseline implementation reported in the original project achieved MAE ~82.80 and R² ~0.9392. "
            "Our rigorous investigation revealed that the original workflow used a **random 80/20 train_test_split** on hourly time series observations while feeding in exact `Day`, `Month`, and `DayOfYear` features. "
            "In a random split, hours from the same day (e.g. 13:00 and 15:00) are placed in the training set while 14:00 is placed in the test set, creating artificial temporal interpolation (data leakage)."
        )

        st.markdown("#### Side-by-Side Evaluation")
        col_bench1, col_bench2 = st.columns(2)
        with col_bench1:
            st.markdown("**Historical Baseline (Random 80/20 Split)**")
            if not baseline_df.empty:
                b_cols = [c for c in ["Model", "MAE", "RMSE", "R2", "WAPE"] if c in baseline_df.columns]
                st.dataframe(baseline_df[b_cols].style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.4f}", "WAPE": "{:.2f}%"}), use_container_width=True)
                if config.BASELINE_COMPARISON_PATH.exists():
                    st.download_button(
                        label="📥 Download Baseline_Model_Comparison.csv",
                        data=config.BASELINE_COMPARISON_PATH.read_bytes(),
                        file_name="Baseline_Model_Comparison.csv",
                        mime="text/csv",
                        key="dl_baseline_comp_csv",
                    )
            else:
                st.info("Baseline comparison not yet generated.")
        with col_bench2:
            st.markdown("**Upgraded Rigorous Benchmark (Strict Chronological 80/20 Split)**")
            c_cols = [c for c in ["Model", "MAE", "RMSE", "R2", "WAPE"] if c in comparison_df.columns]
            st.dataframe(comparison_df[c_cols].style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.4f}", "WAPE": "{:.2f}%"}), use_container_width=True)

        st.markdown(
            """
            #### Key Scientific Takeaways:
            1. **True Generalization:** Evaluating models on an untouched chronological future period faithfully measures actual operational forecasting performance.
            2. **Mode 1 vs Mode 2:** When historical preceding-hour lags (Mode 2) are available in operations, the upgraded pipeline achieves **MAE ~63.19, RMSE ~105.57, and R² ~0.9703**, genuinely surpassing the baseline without any temporal leakage.
            """
        )

    # Tab 4: Metric Definitions
    with perf_tabs[3]:
        st.markdown("### Metric Mathematical Formulations & Plain-Language Explanations")
        metric_defs = get_metric_explanations()

        for m_name, m_desc in metric_defs.items():
            st.markdown(f"**{m_name}:** {m_desc}")
            st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)
