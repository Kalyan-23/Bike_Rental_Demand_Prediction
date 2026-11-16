"""Exploratory Data Analysis (EDA) page for Seoul Bike Sharing application."""

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
    plot_correlation_matrix,
    plot_rental_distribution,
)


def render_eda_page(df: pd.DataFrame) -> None:
    """Render interactive exploratory data analysis module."""
    data = df.copy()
    if "Weather Category" not in data.columns:
        from src.feature_engineering import add_weather_features
        data = add_weather_features(data)
    if "IsWeekend" not in data.columns or "Month" not in data.columns:
        from src.feature_engineering import add_calendar_features
        data = add_calendar_features(data)
    df = data

    st.markdown("<h1 style='margin-bottom:0;'>Exploratory Data Analysis (EDA)</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94a3b8; font-size:1.05rem; margin-top:4px;'>Investigate cyclical temporal dynamics, meteorological sensitivities, and feature correlations.</p>",
        unsafe_allow_html=True,
    )

    eda_tabs = st.tabs(
        [
            "Temporal & Commute Dynamics",
            "Meteorological Drivers",
            "Seasonal & Calendar Shifts",
            "Correlations & Target Distribution",
            "Custom Multi-Variable Explorer",
        ]
    )

    # 1. Temporal Dynamics
    with eda_tabs[0]:
        st.markdown("### Temporal & Commuter Rush-Hour Dynamics")
        st.info(
            "Bike rental demand exhibits dual daily peaks at **08:00 (Morning Commute)** and **18:00 (Evening Commute)** on weekdays, "
            "whereas weekend usage shifts into a broad leisure plateau spanning 14:00 to 19:00."
        )

        col1, col2 = st.columns([1.1, 0.9])
        with col1:
            st.plotly_chart(plot_hourly_demand(df), use_container_width=True)
        with col2:
            st.plotly_chart(plot_weekday_vs_weekend(df), use_container_width=True)

        st.plotly_chart(plot_daily_trend(df), use_container_width=True)

    # 2. Meteorological Drivers
    with eda_tabs[1]:
        st.markdown("### Meteorological Sensitivities & Elasticity")

        weather_vars = [
            ("Temperature(°C)", "Temperature vs Demand", "Strong positive correlation up to ~28°C; severe drops below 0°C."),
            ("Humidity(%)", "Humidity vs Demand", "Moderate negative correlation; very high humidity (>80%) discourages cycling."),
            ("Wind speed (m/s)", "Wind Speed vs Demand", "Mild negative impact at extreme wind speeds (>3.5 m/s)."),
            ("Solar Radiation (MJ/m2)", "Solar Radiation vs Demand", "Strong daylight driver; clear sunny skies accelerate daytime usage."),
            ("Rainfall(mm)", "Rainfall vs Demand", "Precipitation is an acute deterrent; >2mm rainfall collapses demand by over 80%."),
            ("Snowfall (cm)", "Snowfall vs Demand", "Snow sharply curtails riding due to icy road conditions."),
        ]

        weather_choice = st.selectbox("Select Meteorological Factor to Analyze", [w[0] for w in weather_vars], index=0)
        selected_info = next((w for w in weather_vars if w[0] == weather_choice), weather_vars[0])

        st.caption(f"**Domain Takeaway:** {selected_info[2]}")
        st.plotly_chart(plot_weather_scatter(df, weather_choice, selected_info[1]), use_container_width=True)

        st.markdown("#### Weather Condition Groupings")
        w_cat_avg = df.groupby("Weather Category")[config.TARGET_COL].agg(["mean", "count"]).reset_index()
        w_cat_avg["mean"] = np.round(w_cat_avg["mean"], 1)
        st.dataframe(w_cat_avg.rename(columns={"mean": "Average Hourly Rentals", "count": "Total Operating Hours"}), use_container_width=True)

    # 3. Seasonal & Calendar Shifts
    with eda_tabs[2]:
        st.markdown("### Seasonal Transitions & Holiday Effects")

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.plotly_chart(plot_seasonal_distribution(df), use_container_width=True)
        with s_col2:
            st.plotly_chart(plot_monthly_rentals(df), use_container_width=True)

        st.markdown("#### Holiday and System Functioning Impact")
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            holiday_df = df.groupby("Holiday")[config.TARGET_COL].agg(["mean", "std", "count"]).reset_index()
            st.markdown("**Holiday vs Regular Day Demand**")
            st.dataframe(holiday_df, use_container_width=True)
        with h_col2:
            func_df = df.groupby("Functioning Day")[config.TARGET_COL].agg(["mean", "sum", "count"]).reset_index()
            st.markdown("**System Operational Status (Functioning Day)**")
            st.dataframe(func_df, use_container_width=True)

    # 4. Correlation & Target Distribution
    with eda_tabs[3]:
        st.markdown("### Target Distribution & Feature Multicollinearity")

        d_col1, d_col2 = st.columns([1, 1.1])
        with d_col1:
            st.plotly_chart(plot_rental_distribution(df), use_container_width=True)
            st.caption(
                "**Distribution Shape:** Target rental counts are right-skewed with a significant cluster at zero "
                "(non-functioning days, nocturnal hours, heavy precipitation) and a long tail reaching ~3,550 bikes/hr."
            )
        with d_col2:
            st.plotly_chart(plot_correlation_matrix(df), use_container_width=True)
            st.caption(
                "**Multicollinearity Notice:** High correlation exists between Temperature and Dew Point Temperature (r ≈ 0.91). "
                "Tree ensembles gracefully navigate collinearity, whereas regularized Ridge relies on L2 shrinkage."
            )

    # 5. Custom Multi-Variable Explorer
    with eda_tabs[4]:
        st.markdown("### Custom Interactive Variable Comparison")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        c_col1, c_col2, c_col3 = st.columns(3)
        with c_col1:
            x_axis = st.selectbox("X-Axis Variable", num_cols, index=num_cols.index("Temperature(°C)") if "Temperature(°C)" in num_cols else 0)
        with c_col2:
            y_axis = st.selectbox("Y-Axis Variable", num_cols, index=num_cols.index(config.TARGET_COL) if config.TARGET_COL in num_cols else 1)
        with c_col3:
            color_var = st.selectbox("Color Category Variable", ["None"] + cat_cols, index=1 if cat_cols else 0)

        import plotly.express as px
        try:
            import statsmodels
            has_sm = True
        except ImportError:
            has_sm = False

        fig_custom = px.scatter(
            df.sample(min(3000, len(df)), random_state=42),
            x=x_axis,
            y=y_axis,
            color=color_var if color_var != "None" else None,
            opacity=0.5,
            trendline="ols" if has_sm else None,
            title=f"Custom Scatter: {x_axis} vs {y_axis}",
        )
        from src.visualization import apply_dark_layout
        st.plotly_chart(apply_dark_layout(fig_custom, f"Custom Relationship: {x_axis} vs {y_axis}", x_axis, y_axis), use_container_width=True)
