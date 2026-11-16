"""Interactive Demand Prediction Interface for Seoul Bike Sharing application."""

from __future__ import annotations

import datetime
import streamlit as st
import pandas as pd
import numpy as np

import config
from src.prediction import DemandPredictor, categorize_demand
from src.visualization import plot_prediction_gauge


def render_prediction_page() -> None:
    """Render interactive demand prediction interface with presets and validation."""
    st.markdown("<h1 style='margin-bottom:0;'>Real-Time Bike Demand Prediction</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94a3b8; font-size:1.05rem; margin-top:4px;'>Forecast hourly rental volumes based on atmospheric conditions and temporal schedules.</p>",
        unsafe_allow_html=True,
    )

    # Initialize predictor
    try:
        predictor = DemandPredictor()
    except Exception as exc:
        st.error(f"Could not load prediction model bundle: {exc}. Please train a model via the Model Training page first.")
        return

    # Scenario Presets Selector
    st.markdown("#### Scenario Quick Presets")
    preset_cols = st.columns(4)

    preset_values = {
        "Sunny Summer Evening Rush": {
            "Date": "2018-07-16",
            "Hour": 18,
            "Temperature(°C)": 28.5,
            "Humidity(%)": 45.0,
            "Wind speed (m/s)": 1.8,
            "Visibility (10m)": 2000.0,
            "Dew point temperature(°C)": 15.2,
            "Solar Radiation (MJ/m2)": 1.4,
            "Rainfall(mm)": 0.0,
            "Snowfall (cm)": 0.0,
            "Seasons": "Summer",
            "Holiday": "No Holiday",
            "Functioning Day": "Yes",
        },
        "Pleasant Spring Morning Commute": {
            "Date": "2018-05-10",
            "Hour": 8,
            "Temperature(°C)": 18.0,
            "Humidity(%)": 52.0,
            "Wind speed (m/s)": 1.2,
            "Visibility (10m)": 2000.0,
            "Dew point temperature(°C)": 8.0,
            "Solar Radiation (MJ/m2)": 0.8,
            "Rainfall(mm)": 0.0,
            "Snowfall (cm)": 0.0,
            "Seasons": "Spring",
            "Holiday": "No Holiday",
            "Functioning Day": "Yes",
        },
        "Freezing Winter Night": {
            "Date": "2018-01-15",
            "Hour": 23,
            "Temperature(°C)": -8.0,
            "Humidity(%)": 40.0,
            "Wind speed (m/s)": 2.5,
            "Visibility (10m)": 1800.0,
            "Dew point temperature(°C)": -19.0,
            "Solar Radiation (MJ/m2)": 0.0,
            "Rainfall(mm)": 0.0,
            "Snowfall (cm)": 1.5,
            "Seasons": "Winter",
            "Holiday": "No Holiday",
            "Functioning Day": "Yes",
        },
        "Heavy Autumn Downpour": {
            "Date": "2018-10-05",
            "Hour": 17,
            "Temperature(°C)": 16.0,
            "Humidity(%)": 92.0,
            "Wind speed (m/s)": 3.8,
            "Visibility (10m)": 450.0,
            "Dew point temperature(°C)": 14.5,
            "Solar Radiation (MJ/m2)": 0.1,
            "Rainfall(mm)": 14.0,
            "Snowfall (cm)": 0.0,
            "Seasons": "Autumn",
            "Holiday": "No Holiday",
            "Functioning Day": "Yes",
        },
    }

    if "prediction_inputs" not in st.session_state:
        st.session_state.prediction_inputs = preset_values["Sunny Summer Evening Rush"].copy()

    for idx, (preset_name, p_vals) in enumerate(preset_values.items()):
        with preset_cols[idx]:
            if st.button(preset_name, key=f"preset_btn_{idx}", use_container_width=True):
                st.session_state.prediction_inputs = p_vals.copy()
                st.rerun()

    cur_inputs = st.session_state.prediction_inputs

    # Mode Selector
    st.markdown("<hr style='margin:16px 0 20px 0; border-color:#1f293d;'>", unsafe_allow_html=True)
    mode_col1, mode_col2 = st.columns([1, 2])
    with mode_col1:
        pred_mode = st.radio(
            "Prediction Paradigm",
            ["Mode 1: Weather & Time Scenario (Instant Planning)", "Mode 2: Historical-Data-Assisted Forecasting (Lag Dispatch)"],
            index=0,
            help="Mode 1 requires only calendar and weather forecasts. Mode 2 uses preceding hours' demand if operational telematics are available.",
        )

    # Input Form
    with st.form("demand_prediction_form"):
        st.markdown("### Atmospheric & Operational Parameter Inputs")

        # Row 1: Calendar & Operational Status
        c_cal1, c_cal2, c_cal3, c_cal4 = st.columns(4)
        with c_cal1:
            default_d = datetime.date.fromisoformat(cur_inputs.get("Date", "2018-05-15"))
            in_date = st.date_input("Prediction Date", value=default_d)
        with c_cal2:
            in_hour = st.slider("Hour of Day (0 - 23)", 0, 23, int(cur_inputs.get("Hour", 18)))
        with c_cal3:
            in_season = st.selectbox("Season", config.SEASONS, index=config.SEASONS.index(cur_inputs.get("Seasons", "Summer")))
        with c_cal4:
            in_holiday = st.selectbox("Holiday Status", config.HOLIDAYS, index=config.HOLIDAYS.index(cur_inputs.get("Holiday", "No Holiday")))

        # Row 2: Operational Functioning & Temperatures
        c_tmp1, c_tmp2, c_tmp3, c_tmp4 = st.columns(4)
        with c_tmp1:
            in_func = st.selectbox("Functioning Day", config.FUNCTIONING_DAYS, index=config.FUNCTIONING_DAYS.index(cur_inputs.get("Functioning Day", "Yes")), help="Select 'No' if the bike sharing network is shut down for maintenance or emergency.")
        with c_tmp2:
            in_temp = st.number_input("Temperature (°C)", min_value=-25.0, max_value=45.0, value=float(cur_inputs.get("Temperature(°C)", 25.0)), step=0.5)
        with c_tmp3:
            in_humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=float(cur_inputs.get("Humidity(%)", 50.0)), step=1.0)
        with c_tmp4:
            in_dew = st.number_input("Dew Point Temp (°C)", min_value=-35.0, max_value=35.0, value=float(cur_inputs.get("Dew point temperature(°C)", 12.0)), step=0.5)

        # Row 3: Wind, Solar, Precipitation
        c_wth1, c_wth2, c_wth3, c_wth4 = st.columns(4)
        with c_wth1:
            in_wind = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=25.0, value=float(cur_inputs.get("Wind speed (m/s)", 1.5)), step=0.1)
        with c_wth2:
            in_solar = st.number_input("Solar Radiation (MJ/m2)", min_value=0.0, max_value=5.0, value=float(cur_inputs.get("Solar Radiation (MJ/m2)", 1.2)), step=0.1)
        with c_wth3:
            in_rain = st.number_input("Rainfall (mm)", min_value=0.0, max_value=100.0, value=float(cur_inputs.get("Rainfall(mm)", 0.0)), step=0.5)
        with c_wth4:
            in_snow = st.number_input("Snowfall (cm)", min_value=0.0, max_value=50.0, value=float(cur_inputs.get("Snowfall (cm)", 0.0)), step=0.5)

        # Row 4: Visibility
        c_vis1, c_vis2 = st.columns([1, 3])
        with c_vis1:
            in_vis = st.number_input("Visibility (10m)", min_value=0.0, max_value=2500.0, value=float(cur_inputs.get("Visibility (10m)", 2000.0)), step=50.0)

        # Mode 2 Lag Inputs (if selected)
        lag_inputs = {}
        if "Mode 2" in pred_mode:
            st.markdown("#### Mode 2 Operational Telematics Inputs")
            l_col1, l_col2, l_col3 = st.columns(3)
            with l_col1:
                lag_inputs["Lag_1h"] = st.number_input("Demand at Hour t-1 (Previous Hour)", 0, 4000, 750, step=25)
            with l_col2:
                lag_inputs["Lag_24h"] = st.number_input("Demand at Same Hour Yesterday (t-24)", 0, 4000, 700, step=25)
            with l_col3:
                lag_inputs["Rolling_Mean_3h"] = st.number_input("Rolling 3-Hour Average Demand", 0, 4000, 680, step=25)

        submit_btn = st.form_submit_button("Predict Bike Rental Demand", type="primary", use_container_width=True)

    # Handle Prediction Output
    if submit_btn:
        raw_submission = {
            "Date": in_date.strftime("%Y-%m-%d"),
            "Hour": in_hour,
            "Temperature(°C)": in_temp,
            "Humidity(%)": in_humidity,
            "Wind speed (m/s)": in_wind,
            "Visibility (10m)": in_vis,
            "Dew point temperature(°C)": in_dew,
            "Solar Radiation (MJ/m2)": in_solar,
            "Rainfall(mm)": in_rain,
            "Snowfall (cm)": in_snow,
            "Seasons": in_season,
            "Holiday": in_holiday,
            "Functioning Day": in_func,
        }
        if "Mode 2" in pred_mode:
            raw_submission.update(lag_inputs)

        # Update session state with current inputs
        st.session_state.prediction_inputs = raw_submission.copy()

        with st.spinner("Generating demand forecast and calculating uncertainty bounds..."):
            try:
                res = predictor.predict_single(raw_submission)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

        st.markdown("<hr style='margin:24px 0 20px 0; border-color:#1f293d;'>", unsafe_allow_html=True)
        st.markdown("### Prediction Results & Operational Intelligence")

        res_col_left, res_col_right = st.columns([1.1, 0.9])

        with res_col_left:
            st.plotly_chart(plot_prediction_gauge(res["predicted_count"], res["demand_category"]), use_container_width=True)

            cat_color = res["category_color"]
            cat_name = res["demand_category"]
            st.markdown(
                f"""
                <div style='background-color:#131b2e; border:1px solid {cat_color}; border-radius:8px; padding:16px; margin-top:8px;'>
                    <span style='background-color:{cat_color}; color:#0b0f19; font-weight:700; padding:4px 10px; border-radius:4px; font-size:0.9rem;'>
                        {cat_name.upper()} DEMAND TIER
                    </span>
                    <p style='color:#e2e8f0; font-size:0.98rem; margin:10px 0 4px 0;'>
                        <b>Operational Guidance:</b> {res['category_description']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col_right:
            st.markdown("#### Forecast Summary & Bounds")
            k1, k2 = st.columns(2)
            with k1:
                st.metric("Predicted Rentals", f"{res['predicted_count']:,} bikes", help="Expected hourly volume.")
            with k2:
                st.metric("90% Confidence Interval", f"{res['lower_bound_90pct']:,} - {res['upper_bound_90pct']:,}", help="Expected bound based on test residual RMSE.")

            st.markdown("#### Model & Environmental Context")
            st.markdown(
                f"""
                - **Active Estimator:** `{res['model_used']}`
                - **Test MAE Benchmark:** `{res['model_mae']:.1f} bikes`
                - **Functioning Status:** `{res['functioning_day']}`
                - **Discomfort Index:** `{res['input_summary']['Feels Like Index']}`
                - **Precipitation Threat:** `{'Yes' if (in_rain > 0 or in_snow > 0) else 'None'}`
                """
            )
            if in_func == "No":
                st.warning("Network status is non-functioning; demand is strictly enforced to 0 bikes.")
