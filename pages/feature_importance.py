"""Feature Importance & Explainability page for Seoul Bike Sharing application."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

import config
from src.model_training import load_model_bundle
from src.visualization import plot_feature_importance_bar


def render_feature_importance_page(importance_df: pd.DataFrame, metadata: dict) -> None:
    """Render interactive feature importance, SHAP values, and causal explanations."""
    st.markdown("<h1 style='margin-bottom:0;'>Feature Importance & Model Explainability</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94a3b8; font-size:1.05rem; margin-top:4px;'>Understand the key predictors driving bike demand forecasts and distinguish correlation from causality.</p>",
        unsafe_allow_html=True,
    )

    best_model_name = metadata.get("best_model", "Extra Trees")

    if importance_df.empty:
        st.warning("Feature importance data is not yet generated. Please train a model via main.py or the UI.")
        return

    # Top Feature Highlights
    top_feature = str(importance_df.iloc[0]["Feature"])
    second_feature = str(importance_df.iloc[1]["Feature"]) if len(importance_df) > 1 else "Hour"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Primary Demand Driver", top_feature, delta="Highest Gini / Split Gain")
    with c2:
        st.metric("Secondary Demand Driver", second_feature)
    with c3:
        st.metric("Active Model Architecture", best_model_name)

    st.markdown("<hr style='margin:16px 0 20px 0; border-color:#1f293d;'>", unsafe_allow_html=True)

    feat_tabs = st.tabs(["Feature Importance Rankings", "SHAP Explainability", "Correlation vs Causality Guide"])

    with feat_tabs[0]:
        st.markdown(f"### Most Influential Predictors ({best_model_name})")

        f_col1, f_col2 = st.columns([1, 1.2])
        with f_col1:
            top_n = st.radio("Display Top N Features", [10, 15, 20], horizontal=True, index=1)
            st.plotly_chart(plot_feature_importance_bar(importance_df, top_n=top_n), use_container_width=True)

        with f_col2:
            st.markdown("#### Feature Importance Leaderboard Table")
            display_df = importance_df.head(top_n).copy()
            if "Relative_Importance_Pct" in display_df.columns:
                st.dataframe(
                    display_df[["Feature", "Importance", "Relative_Importance_Pct"]].style.format(
                        {"Importance": "{:.4f}", "Relative_Importance_Pct": "{:.2f}%"}
                    ),
                    use_container_width=True,
                )
            else:
                st.dataframe(display_df.style.format({"Importance": "{:.4f}"}), use_container_width=True)

            if config.FEATURE_IMPORTANCE_PATH.exists():
                st.download_button(
                    label="📥 Download Feature_Importance.csv",
                    data=config.FEATURE_IMPORTANCE_PATH.read_bytes(),
                    file_name="Feature_Importance.csv",
                    mime="text/csv",
                    key="dl_feat_imp_csv",
                )

    with feat_tabs[1]:
        st.markdown("### Tree SHAP (SHapley Additive exPlanations)")
        st.caption("SHAP decomposes each individual prediction into additive feature contributions based on cooperative game theory.")

        try:
            import shap
            shap_available = True
        except ImportError:
            shap_available = False

        if not shap_available:
            st.info("SHAP is not installed in the current environment. To enable SHAP visualizations, install `shap` via pip.")
        else:
            if st.button("Generate SHAP Summary Sample (150 test observations)", key="run_shap_btn"):
                with st.spinner("Computing Shapley values across test observations..."):
                    try:
                        bundle = load_model_bundle()
                        pipeline = bundle["pipeline"]
                        preprocessor = pipeline.named_steps["preprocessor"]
                        model = pipeline.named_steps["model"]

                        test_pred_path = config.TEST_PREDICTIONS_PATH
                        power_bi_path = config.POWER_BI_DATA_PATH
                        if power_bi_path.exists():
                            df_pbi = pd.read_csv(power_bi_path)
                            test_sample = df_pbi[df_pbi["Data Split"] == "Test"].head(150)
                            transformed = preprocessor.transform(test_sample)
                            feat_names = preprocessor.get_feature_names_out()
                            feat_clean = [f.split("__")[-1] for f in feat_names]

                            # TreeExplainer for tree models
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(transformed)

                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(9, 6))
                            shap.summary_plot(shap_values, transformed, feature_names=feat_clean, show=False, max_display=12)
                            st.pyplot(fig)
                            plt.close(fig)
                            st.success("SHAP explanation computed successfully!")
                        else:
                            st.warning("Processed test dataset not found to compute SHAP sample.")
                    except Exception as err:
                        st.error(f"SHAP explanation calculation encountered an error: {err}")
            else:
                st.markdown(
                    """
                    Click the button above to calculate SHAP feature attributions on the test partition.
                    SHAP answers: *'How much did each specific feature raise or lower the predicted rentals relative to the base value?'*
                    """
                )

    with feat_tabs[2]:
        st.markdown("### The Critical Difference Between Importance and Causality")
        st.info(
            "**Key Data Science Concept:** Model feature importance identifies variables that provide strong statistical split gain or error reduction. "
            "It does **not** prove that manipulating that feature directly causes ridership to change."
        )

        st.markdown(
            """
            #### Real-World Domain Examples:
            1. **Temperature & Solar Radiation:**
               - *Statistical finding:* High positive importance score.
               - *Causal reality:* Warm, sunny weather invites outdoor recreation. However, heating a bicycle station artificially will not generate riders on a freezing winter day.
            2. **Hour of Day & Peak Indicators:**
               - *Statistical finding:* Explains a large fraction of variance.
               - *Causal reality:* Demand is driven by fixed societal work and school schedules (e.g. 9-to-5 office hours), for which 'Hour' serves as a mathematical proxy.
            3. **Precipitation (Rainfall / Snowfall):**
               - *Statistical finding:* Heavy rain immediately reduces demand to near zero.
               - *Causal reality:* This is a **direct causal barrier**—riders avoid cycling due to physical discomfort, safety risks, and slippery road conditions.
            """
        )
