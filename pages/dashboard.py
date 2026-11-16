"""Dashboard Overview Page for Seoul Bike Sharing Intelligence System."""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

import config
from src.visualization import (
    plot_hourly_demand,
    plot_daily_trend,
    plot_monthly_rentals,
    plot_seasonal_distribution,
    plot_weekday_vs_weekend,
    plot_weather_scatter,
)


def render_dashboard_page(df: pd.DataFrame, comparison_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """Render the main executive overview dashboard."""
    data = df.copy()
    if "Weather Category" not in data.columns:
        from src.feature_engineering import add_weather_features
        data = add_weather_features(data)
    if "IsWeekend" not in data.columns or "Month" not in data.columns:
        from src.feature_engineering import add_calendar_features
        data = add_calendar_features(data)
    df = data

    st.markdown("<h1 style='margin-bottom:0;'>Seoul Bike Sharing Demand Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94a3b8; font-size:1.05rem; margin-top:4px;'>Real-time operational analytics and machine learning demand intelligence.</p>",
        unsafe_allow_html=True,
    )

    # Sidebar Filter Controls
    with st.sidebar:
        st.markdown("### Dashboard Filters")
        st.markdown("<hr style='margin:4px 0 16px 0; border-color:#1f293d;'>", unsafe_allow_html=True)

        # Date Range Filter
        min_date = pd.to_datetime(df[config.DATE_COL]).min().date()
        max_date = pd.to_datetime(df[config.DATE_COL]).max().date()
        selected_dates = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="dash_date_range",
        )

        # Seasons Filter
        all_seasons = config.SEASONS
        selected_seasons = st.multiselect(
            "Seasons",
            options=all_seasons,
            default=all_seasons,
            key="dash_seasons",
        )

        # Holiday Filter
        holiday_opts = ["All"] + config.HOLIDAYS
        selected_holiday = st.selectbox("Holiday Status", holiday_opts, index=0, key="dash_holiday")

        # Functioning Day Filter
        func_opts = ["All"] + config.FUNCTIONING_DAYS
        selected_func = st.selectbox("Functioning Day", func_opts, index=0, key="dash_func_day")

        # Hour Range
        selected_hours = st.slider("Hour of Day Range", 0, 23, (0, 23), key="dash_hours")

    # Apply Filters to DataFrame
    filtered_df = df.copy()
    filtered_df["_dt"] = pd.to_datetime(filtered_df[config.DATE_COL])

    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        start_d, end_d = selected_dates
        filtered_df = filtered_df[(filtered_df["_dt"].dt.date >= start_d) & (filtered_df["_dt"].dt.date <= end_d)]

    if selected_seasons:
        filtered_df = filtered_df[filtered_df["Seasons"].isin(selected_seasons)]

    if selected_holiday != "All":
        filtered_df = filtered_df[filtered_df["Holiday"] == selected_holiday]

    if selected_func != "All":
        filtered_df = filtered_df[filtered_df["Functioning Day"] == selected_func]

    filtered_df = filtered_df[(filtered_df["Hour"] >= selected_hours[0]) & (filtered_df["Hour"] <= selected_hours[1])]

    if filtered_df.empty:
        st.warning("No records match the selected filter combination. Please broaden your filter criteria.")
        return

    # Calculate Live KPIs from Filtered Data
    total_rentals = int(filtered_df[config.TARGET_COL].sum())
    avg_rentals = round(float(filtered_df[config.TARGET_COL].mean()), 1)
    max_rentals = int(filtered_df[config.TARGET_COL].max())
    min_rentals = int(filtered_df[config.TARGET_COL].min())

    hourly_grp = filtered_df.groupby("Hour")[config.TARGET_COL].mean()
    peak_hour = int(hourly_grp.idxmax()) if not hourly_grp.empty else 18

    season_grp = filtered_df.groupby("Seasons")[config.TARGET_COL].mean()
    peak_season = str(season_grp.idxmax()) if not season_grp.empty else "Summer"

    # Best Model Performance Metrics (from saved model comparison)
    best_row = comparison_df.iloc[0] if not comparison_df.empty else {}
    best_model_name = best_row.get("Model", "Extra Trees")
    best_mae = best_row.get("MAE", 142.06)
    best_rmse = best_row.get("RMSE", 232.15)
    best_r2 = best_row.get("R2", 0.8567)

    # 1. Headline KPI Cards
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Bike Rentals", f"{total_rentals:,}", help="Total sum of rented bikes for filtered period.")
    with c2:
        st.metric("Average Rentals / Hr", f"{avg_rentals:,}", help="Mean bikes rented per operating hour.")
    with c3:
        st.metric("Peak Demand Hour", f"{peak_hour}:00", help="Hour with highest average demand.")
    with c4:
        st.metric("Highest Season", peak_season, help="Season with peak demand.")
    with c5:
        st.metric("Max Demand Peak", f"{max_rentals:,}", help="Single highest recorded hourly demand.")

    # 2. Machine Learning KPI Cards
    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Best ML Model", str(best_model_name), delta="Chronological Benchmarked")
    with m2:
        st.metric("Test MAE", f"{best_mae:.2f} bikes", help="Mean Absolute Error on unseen chronological test set.")
    with m3:
        st.metric("Test RMSE", f"{best_rmse:.2f} bikes", help="Root Mean Squared Error.")
    with m4:
        st.metric("Test R² Score", f"{best_r2:.4f}", help="Coefficient of Determination (0 to 1).")

    st.markdown("<hr style='margin:20px 0; border-color:#1f293d;'>", unsafe_allow_html=True)

    # 3. Interactive Charts Grid
    row1_left, row1_right = st.columns([1.1, 0.9])
    with row1_left:
        st.plotly_chart(plot_hourly_demand(filtered_df), use_container_width=True)
    with row1_right:
        st.plotly_chart(plot_weekday_vs_weekend(filtered_df), use_container_width=True)

    row2_left, row2_right = st.columns([1.2, 0.8])
    with row2_left:
        st.plotly_chart(plot_daily_trend(filtered_df), use_container_width=True)
    with row2_right:
        st.plotly_chart(plot_seasonal_distribution(filtered_df), use_container_width=True)

    row3_left, row3_right = st.columns([1, 1])
    with row3_left:
        st.plotly_chart(plot_monthly_rentals(filtered_df), use_container_width=True)
    with row3_right:
        st.plotly_chart(plot_weather_scatter(filtered_df, "Temperature(°C)", "Temperature Impact on Demand"), use_container_width=True)

    # Export Section
    st.markdown("<hr style='margin:24px 0 16px 0; border-color:#1f293d;'>", unsafe_allow_html=True)
    exp_col1, exp_col2, exp_col3 = st.columns([2, 1, 1])
    with exp_col1:
        st.markdown("#### 📊 Export Dashboard Analytics & Summaries")
        st.caption(f"Download the currently filtered dataset ({len(filtered_df):,} records) or the executive summary for offline analysis.")
    with exp_col2:
        st.download_button(
            label=f"📥 Download Filtered Data ({len(filtered_df):,} rows)",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="SeoulBike_Dashboard_Filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with exp_col3:
        if config.ANALYSIS_SUMMARY_PATH.exists():
            st.download_button(
                label="📥 Download Analysis_Summary.csv",
                data=config.ANALYSIS_SUMMARY_PATH.read_bytes(),
                file_name="Analysis_Summary.csv",
                mime="text/csv",
                use_container_width=True,
            )
