"""Project Information, Methodology, and Power BI Guide."""

from __future__ import annotations

import streamlit as st
import pandas as pd

import config


def render_project_info_page() -> None:
    """Render comprehensive documentation, system architecture, and Power BI guide."""
    st.markdown("<h1 style='margin-bottom:0;'>Project Information & Technical Documentation</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94a3b8; font-size:1.05rem; margin-top:4px;'>Seoul Bike Sharing Demand Prediction and Intelligent Business Analytics System.</p>",
        unsafe_allow_html=True,
    )

    info_tabs = st.tabs(
        [
            "System Architecture & Overview",
            "Evaluation Metrics & Mathematics",
            "Demand Quartile Thresholds",
            "Data Dictionary",
            "Power BI Dashboard Integration Guide",
        ]
    )

    # Tab 1: Architecture
    with info_tabs[0]:
        st.markdown("### End-to-End System Architecture")
        st.markdown(
            """
            This platform is an enterprise-grade machine learning and business intelligence system designed to forecast urban shared-bicycle demand in Seoul, South Korea.
            The pipeline is engineered to eliminate data leakage, enforce strict chronological evaluation, and provide dual operational paradigms (scenario planning vs telematics dispatch).
            """
        )

        st.markdown(
            """
            ```
            +-----------------------------------------------------------------------------------+
            |                                 RAW DATA INGESTION                                |
            |            SeoulBikeData.csv (8,760 hourly records, 1 complete calendar year)     |
            +-----------------------------------------+-----------------------------------------+
                                                      |
                                                      v
            +-----------------------------------------------------------------------------------+
            |                          PREPROCESSING & DATA AUDIT                               |
            |   - Duplicate removal & timestamp normalization                                   |
            |   - Non-destructive Tukey IQR outlier auditing (genuine surges preserved)         |
            |   - Strict fit-on-train preprocessor pipeline (ColumnTransformer, Imputer, OHE)   |
            +-----------------------------------------+-----------------------------------------+
                                                      |
                                                      v
            +-----------------------------------------------------------------------------------+
            |                         DOMAIN FEATURE ENGINEERING                                |
            |   - Cyclical Sine/Cosine encodings (Hour, Month, DayOfWeek)                       |
            |   - Commuter rush-hour indicators (Morning 07-09, Evening 17-19)                  |
            |   - Thom's Discomfort Index & Temperature-Humidity interactions                   |
            |   - Mode 1: Pure Exogenous Weather & Time (Zero Leakage Scenario Planning)        |
            |   - Mode 2: Non-Leaking Operational Lags (Lag-1h, Lag-24h, Rolling-3h)            |
            +-----------------------------------------+-----------------------------------------+
                                                      |
                                                      v
            +-----------------------------------------------------------------------------------+
            |                   MODEL BENCHMARKING & TIME-SERIES EVALUATION                     |
            |   - Strict Chronological 80/20 Train/Test Split                                   |
            |   - LightGBM, Extra Trees, Random Forest, XGBoost, CatBoost, HistGBM, Ridge       |
            |   - Metrics: MAE, MSE, RMSE, R², WAPE, sMAPE, Explained Variance                  |
            |   - Historical Baseline Replication (honest random split comparison)              |
            +-----------------------------------------+-----------------------------------------+
                                                      |
                                                      v
            +-----------------------------------------------------------------------------------+
            |                       INTERACTIVE DEPLOYMENT INTERFACES                           |
            |   - Streamlit Multi-Page Analytics Application                                    |
            |   - Power BI Cleaned CSVs (SeoulBike_Updated_For_PowerBI.csv, Model_Comparison)   |
            |   - Real-time scenario prediction with 90% confidence bounds & quartile gauges    |
            +-----------------------------------------------------------------------------------+
            ```
            """
        )

    # Tab 2: Mathematics
    with info_tabs[1]:
        st.markdown("### Mathematical Formulation of Evaluation Metrics")

        st.markdown("#### 1. Mean Absolute Error (MAE)")
        st.latex(r"\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|")
        st.caption("Measures average magnitude of errors in identical units to the target (bikes). Robust to outliers.")

        st.markdown("#### 2. Root Mean Squared Error (RMSE)")
        st.latex(r"\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}")
        st.caption("Heavily penalizes large deviation errors. Critical for bike sharing to avoid severe fleet shortages during peak hours.")

        st.markdown("#### 3. Coefficient of Determination (R²)")
        st.latex(r"R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}")
        st.caption("Represents the proportion of total demand variance explained by the model relative to a mean baseline.")

        st.markdown("#### 4. Weighted Absolute Percentage Error (WAPE)")
        st.latex(r"\text{WAPE} = \frac{\sum_{i=1}^{n} |y_i - \hat{y}_i|}{\sum_{i=1}^{n} y_i} \times 100\%")
        st.caption("Aggregate percentage error. Highly robust against zero division when individual hours experience 0 rentals.")

        st.markdown("#### 5. Symmetric Mean Absolute Percentage Error (sMAPE)")
        st.latex(r"\text{sMAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \frac{2 \cdot |y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i| + \epsilon}")
        st.caption("Bounded percentage error metric between 0% and 200%, treating over- and under-predictions symmetrically.")

    # Tab 3: Quartiles
    with info_tabs[2]:
        st.markdown("### Empirical Demand Category Thresholds")
        st.markdown(
            """
            Unlike arbitrary guesses, the demand categories utilized in this system are mathematically derived
            from empirical statistical quartiles of the Seoul Bike Sharing dataset:
            """
        )

        q_table = pd.DataFrame(
            [
                {
                    "Demand Category": "Low Demand",
                    "Range (Bikes / Hr)": "0 – 200 bikes",
                    "Percentile Band": "0th – 25th percentile (Q1)",
                    "Operational Guidance": "Off-peak or severe adverse weather; minimal rebalancing needed.",
                },
                {
                    "Demand Category": "Moderate Demand",
                    "Range (Bikes / Hr)": "201 – 550 bikes",
                    "Percentile Band": "25th – 50th percentile (Median)",
                    "Operational Guidance": "Steady daytime baseline; standard maintenance window.",
                },
                {
                    "Demand Category": "High Demand",
                    "Range (Bikes / Hr)": "551 – 1,100 bikes",
                    "Percentile Band": "50th – 75th percentile (Q3)",
                    "Operational Guidance": "Elevated commuter turnover; active dock monitoring required.",
                },
                {
                    "Demand Category": "Very High (Surge)",
                    "Range (Bikes / Hr)": "> 1,100 bikes (up to ~3,556)",
                    "Percentile Band": "Top 25% (Above Q3 / P90)",
                    "Operational Guidance": "Major commute rush or summer weekend; maximum fleet deployment and automated truck redistribution.",
                },
            ]
        )
        st.dataframe(q_table, use_container_width=True)

    # Tab 4: Data Dictionary
    with info_tabs[3]:
        st.markdown("### Dataset Field Dictionary")
        dict_data = [
            {"Field Name": "Date", "Data Type": "String / Date", "Description": "Calendar date of observation (DD/MM/YYYY)."},
            {"Field Name": "Rented Bike Count", "Data Type": "Integer", "Description": "Total number of rented bicycles during the specific hour (Target)."},
            {"Field Name": "Hour", "Data Type": "Integer (0-23)", "Description": "Hour of the day."},
            {"Field Name": "Temperature(°C)", "Data Type": "Float", "Description": "Outdoor ambient temperature in degrees Celsius."},
            {"Field Name": "Humidity(%)", "Data Type": "Float", "Description": "Relative air humidity percentage."},
            {"Field Name": "Wind speed (m/s)", "Data Type": "Float", "Description": "Wind speed in meters per second."},
            {"Field Name": "Visibility (10m)", "Data Type": "Float", "Description": "Atmospheric visibility measure (units of 10 meters)."},
            {"Field Name": "Dew point temperature(°C)", "Data Type": "Float", "Description": "Temperature to which air must be cooled to become saturated."},
            {"Field Name": "Solar Radiation (MJ/m2)", "Data Type": "Float", "Description": "Solar radiation intensity in Megajoules per square meter."},
            {"Field Name": "Rainfall(mm)", "Data Type": "Float", "Description": "Hourly rainfall accumulation in millimeters."},
            {"Field Name": "Snowfall (cm)", "Data Type": "Float", "Description": "Hourly snowfall accumulation in centimeters."},
            {"Field Name": "Seasons", "Data Type": "Categorical", "Description": "Spring, Summer, Autumn, or Winter."},
            {"Field Name": "Holiday", "Data Type": "Categorical", "Description": "Holiday or No Holiday."},
            {"Field Name": "Functioning Day", "Data Type": "Categorical", "Description": "Yes (Network operational) or No (System shut down)."},
        ]
        st.dataframe(pd.DataFrame(dict_data), use_container_width=True)

    # Tab 5: Power BI Guide
    with info_tabs[4]:
        st.markdown("### Power BI Desktop Integration Instructions")
        st.markdown(
            """
            This project automatically writes pre-structured, Power BI-compatible CSV files to `output/`:

            1. **`output/SeoulBike_Updated_For_PowerBI.csv`**:
               - Contains all original fields, engineered features, `Data Split` (Train vs Test), `Predicted_Demand`, and `Prediction_Residual`.
            2. **`output/Model_Comparison.csv`**:
               - Benchmarked MAE, RMSE, R², WAPE, and sMAPE metrics across all models.
            3. **`output/Feature_Importance.csv`**:
               - Ranked predictor importances for building Power BI Pareto and horizontal bar visuals.
            4. **`output/Analysis_Summary.csv`**:
               - Headline KPI cards and target correlation coefficients.

            #### Recommended Power BI Dashboard Layout:
            - **Page 1: Executive Overview:** Cards for Total Rentals, Average Demand, Peak Hour. Slicers for Date, Seasons, Holiday, Weather Category. Area chart for Date vs Total Rentals.
            - **Page 2: Meteorological Elasticity:** Scatter plots of Temperature, Humidity, and Wind Speed vs Demand with dynamic trendlines.
            - **Page 3: Model Evaluation:** Import `Model_Comparison.csv` into a Clustered Bar Chart. Add Actual vs Predicted scatter chart filtered to `Data Split = 'Test'`.
            """
        )
