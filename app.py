"""Seoul Bike Sharing Demand Prediction & Intelligent Business Analytics System.

Main application entry point providing sidebar navigation, dark navy styling,
and cached data/model loaders across all 8 interactive modules.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Configure page layout and metadata
st.set_page_config(
    page_title="Seoul Bike Sharing Demand Intelligence",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import config
from src.data_loader import load_raw_data
from src.model_training import load_model_bundle

# Import Page Renderers
from pages.dashboard import render_dashboard_page
from pages.dataset_explorer import render_dataset_explorer_page
from pages.eda import render_eda_page
from pages.prediction import render_prediction_page
from pages.model_performance import render_model_performance_page
from pages.feature_importance import render_feature_importance_page
from pages.model_training import render_model_training_page
from pages.project_info import render_project_info_page

# Inject Global Modern Dark Navy CSS
CUSTOM_CSS = """
<style>
    /* Dark Navy & Charcoal Theme */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    
    /* Sleek Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f1629;
        border-right: 1px solid #1f293d;
    }
    
    /* Modern KPI Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #131b2e 0%, #17223b 100%);
        border: 1px solid #23314f;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 4px 14px 0 rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(0, 210, 255, 0.15);
        border-color: #00d2ff;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] {
        color: #00d2ff !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 2px solid #1f293d;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 6px 6px 0 0;
        color: #94a3b8;
        background-color: transparent;
        font-weight: 500;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        color: #00d2ff !important;
        border-bottom: 3px solid #00d2ff !important;
        background-color: rgba(0, 210, 255, 0.05) !important;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0284c7 0%, #00d2ff 100%);
        color: #0b0f19;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 8px 18px;
        box-shadow: 0 4px 12px rgba(0, 210, 255, 0.25);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #00d2ff 0%, #38bdf8 100%);
        box-shadow: 0 6px 16px rgba(0, 210, 255, 0.4);
        transform: translateY(-1px);
        color: #0b0f19;
    }
    
    /* Dataframes and tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #1f293d;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Badges and Callouts */
    .stAlert {
        background-color: #131b2e;
        border: 1px solid #1f293d;
        border-radius: 8px;
        color: #e2e8f0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Cached Data Loaders
@st.cache_data(show_spinner=False)
def load_app_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    """Load and cache dataset and benchmark outputs."""
    raw_df = load_raw_data()

    # Load cleaned dataset
    cleaned_path = config.CLEANED_DATA_PATH
    if cleaned_path.exists():
        cleaned_df = pd.read_csv(cleaned_path)
    else:
        from src.preprocessing import clean_raw_data
        cleaned_df, _ = clean_raw_data(raw_df)

    # Ensure engineered calendar and weather features exist
    if "Weather Category" not in cleaned_df.columns:
        from src.feature_engineering import add_calendar_features, add_weather_features
        cleaned_df = add_calendar_features(cleaned_df)
        cleaned_df = add_weather_features(cleaned_df)

    # Load Model Comparison
    if config.MODEL_COMPARISON_PATH.exists():
        comparison_df = pd.read_csv(config.MODEL_COMPARISON_PATH)
    else:
        comparison_df = pd.DataFrame()

    # Load Baseline Comparison
    if config.BASELINE_COMPARISON_PATH.exists():
        baseline_df = pd.read_csv(config.BASELINE_COMPARISON_PATH)
    else:
        baseline_df = pd.DataFrame()

    # Load Test Predictions
    if config.TEST_PREDICTIONS_PATH.exists():
        test_pred_df = pd.read_csv(config.TEST_PREDICTIONS_PATH)
    else:
        test_pred_df = pd.DataFrame()

    # Load Feature Importance
    if config.FEATURE_IMPORTANCE_PATH.exists():
        importance_df = pd.read_csv(config.FEATURE_IMPORTANCE_PATH)
    else:
        importance_df = pd.DataFrame()

    # Load Analysis Summary
    if config.ANALYSIS_SUMMARY_PATH.exists():
        summary_df = pd.read_csv(config.ANALYSIS_SUMMARY_PATH)
    else:
        summary_df = pd.DataFrame()

    # Load Metadata
    if config.METADATA_PATH.exists():
        try:
            metadata = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    else:
        metadata = {}

    return raw_df, cleaned_df, comparison_df, baseline_df, test_pred_df, importance_df, summary_df, metadata


def main() -> None:
    """Main application controller and sidebar routing."""
    # Load all cached assets
    (
        raw_df,
        cleaned_df,
        comparison_df,
        baseline_df,
        test_pred_df,
        importance_df,
        summary_df,
        metadata,
    ) = load_app_data()

    # Sidebar Header & Brand
    with st.sidebar:
        st.markdown(
            """
            <div style='display:flex; align-items:center; gap:10px; margin-bottom:12px;'>
                <span style='font-size:2rem;'>🚲</span>
                <div>
                    <h2 style='color:#00d2ff; margin:0; font-size:1.25rem; font-weight:700;'>SEOUL BIKE</h2>
                    <p style='color:#94a3b8; margin:0; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px;'>Demand Intelligence</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<hr style='margin:8px 0 16px 0; border-color:#1f293d;'>", unsafe_allow_html=True)

        pages = [
            "Dashboard Overview",
            "Dataset Explorer",
            "Exploratory Data Analysis",
            "Demand Prediction",
            "Model Performance",
            "Feature Importance",
            "Model Training",
            "Project Information",
        ]

        selected_page = st.radio("Navigation", pages, index=0, label_visibility="collapsed")

        st.markdown("<hr style='margin:24px 0 16px 0; border-color:#1f293d;'>", unsafe_allow_html=True)
        st.caption("Active Model: **Extra Trees** (Mode 1)")
        st.caption("Evaluation Split: **Chronological 80/20**")
        st.caption("Version: **2.0 Production Ready**")

    # Route to Selected Page
    if selected_page == "Dashboard Overview":
        render_dashboard_page(cleaned_df, comparison_df, summary_df)
    elif selected_page == "Dataset Explorer":
        render_dataset_explorer_page(raw_df, cleaned_df, metadata)
    elif selected_page == "Exploratory Data Analysis":
        render_eda_page(cleaned_df)
    elif selected_page == "Demand Prediction":
        render_prediction_page()
    elif selected_page == "Model Performance":
        render_model_performance_page(comparison_df, baseline_df, test_pred_df, metadata)
    elif selected_page == "Feature Importance":
        render_feature_importance_page(importance_df, metadata)
    elif selected_page == "Model Training":
        render_model_training_page(cleaned_df)
    elif selected_page == "Project Information":
        render_project_info_page()


if __name__ == "__main__":
    main()
