# Seoul Bike Sharing Demand Prediction & Intelligent Business Analytics System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

An enterprise-grade, end-to-end Machine Learning and Business Intelligence application designed to predict and analyze hourly urban shared-bicycle demand in Seoul, South Korea. Built upon rigorous time-series methodology, dual operational forecasting paradigms, leakage-free preprocessing, and interactive modern dashboards.

---

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. Problem Formulation & Dataset](#2-problem-formulation--dataset)
- [3. Key Architectural Upgrades](#3-key-architectural-upgrades)
- [4. Dual Operational Prediction Modes](#4-dual-operational-prediction-modes)
- [5. Machine Learning Benchmarks & Baseline Analysis](#5-machine-learning-benchmarks--baseline-analysis)
- [6. Project Structure](#6-project-structure)
- [7. Installation & Quickstart Guide (Windows)](#7-installation--quickstart-guide-windows)
- [8. Application Features & UI Walkthrough](#8-application-features--ui-walkthrough)
- [9. Power BI Dashboard Integration Guide](#9-power-bi-dashboard-integration-guide)
- [10. Automated Validation & Unit Testing](#10-automated-validation--unit-testing)
- [11. Business Insights & Causal Policy Recommendations](#11-business-insights--causal-policy-recommendations)

---

## 1. Executive Summary

Municipal bike sharing systems play a vital role in smart urban transit, reducing carbon emissions and mitigating traffic congestion. However, bike rebalancing presents an acute operational challenge: dock stations frequently suffer from **asymmetric demand surges** during commute peaks or rapid weather fluctuations.

This project transforms the initial academic baseline into a full-scale analytical platform:
1. **Methodological Rigor:** Eliminates temporal leakage by implementing strict chronological splits (first 80% train, final 20% untouched future test) and `TimeSeriesSplit` cross-validation.
2. **Dual-Paradigm Forecasting:** Supports both **Mode 1** (exogenous weather & schedule simulation for planning) and **Mode 2** (auto-regressive lag-assisted dispatch for real-time station balancing).
3. **Multi-Model Suite:** Evaluates 9 regression architectures including LightGBM, Extra Trees, Random Forest, HistGradientBoosting, XGBoost, CatBoost, Gradient Boosting, Ridge Regression, and Decision Trees.
4. **Interactive Dashboard:** Built with Streamlit and Plotly using a dark navy & cyan aesthetic, featuring KPI monitoring, scenario simulation, and explainability (SHAP).
5. **Power BI Ready:** Includes a Power BI report file in `dashboards/` and exports cleaned, split-annotated, and feature-engineered datasets with prediction residuals.

---

## 2. Problem Formulation & Dataset

The platform operates on the official **Seoul Bike Sharing Demand Dataset** (`data/SeoulBikeData.csv`), containing **8,760 continuous hourly observations** spanning an entire calendar year (December 1, 2017 to November 30, 2018; 365 days × 24 hours).

### Key Dataset Attributes:
- **Temporal:** `Date` (DD/MM/YYYY), `Hour` (0 to 23).
- **Target:** `Rented Bike Count` (Non-negative integer; Range: 0 to 3,556; Mean: 704.6; Median: 504.5).
- **Meteorological Predictors:**
  - `Temperature(°C)` (-17.8°C to 39.4°C)
  - `Humidity(%)` (0% to 98%)
  - `Wind speed (m/s)` (0 to 7.4 m/s)
  - `Visibility (10m)` (20 to 2,000)
  - `Dew point temperature(°C)` (-30.6°C to 27.2°C)
  - `Solar Radiation (MJ/m2)` (0 to 3.52 MJ/m²)
  - `Rainfall(mm)` (0 to 35 mm)
  - `Snowfall (cm)` (0 to 8.8 cm)
- **Categorical & Operational Context:**
  - `Seasons`: Spring, Summer, Autumn, Winter (2,160 to 2,208 hours each)
  - `Holiday`: Holiday (432 hours) vs No Holiday (8,328 hours)
  - `Functioning Day`: Yes (8,465 hours) vs No (295 hours — rentals strictly 0 during maintenance/shutdown)

---

## 3. Key Architectural Upgrades

### 3.1 Leakage-Free Preprocessing (`src/preprocessing.py`)
- **Fit-on-Train Guarantee:** Imputers (`SimpleImputer`), encoders (`OneHotEncoder`), and scalers (`StandardScaler`) are encapsulated in a `ColumnTransformer` pipeline fitted **strictly on training observations**. No future test data influences transformations.
- **Tukey IQR Outlier Auditing:** Non-destructive outlier analysis. High-demand observations (e.g. rush hours on warm summer evenings up to 3,556 rentals) represent genuine real-world surge events and are preserved to maintain genuine distribution variance.
- **System Closure Rule:** When `Functioning Day == 'No'`, the system strictly enforces 0 demand.

### 3.2 Advanced Domain Feature Engineering (`src/feature_engineering.py`)
- **Cyclical Trigonometric Transformations:**
  - $\text{Hour}_{\sin} = \sin\left(\frac{2\pi \cdot \text{Hour}}{24}\right)$, $\text{Hour}_{\cos} = \cos\left(\frac{2\pi \cdot \text{Hour}}{24}\right)$
  - $\text{Month}_{\sin} = \sin\left(\frac{2\pi \cdot \text{Month}}{12}\right)$, $\text{Month}_{\cos} = \cos\left(\frac{2\pi \cdot \text{Month}}{12}\right)$
  - $\text{DayOfWeek}_{\sin} = \sin\left(\frac{2\pi \cdot \text{DayOfWeek}}{7}\right)$, $\text{DayOfWeek}_{\cos} = \cos\left(\frac{2\pi \cdot \text{DayOfWeek}}{7}\right)$
- **Commuter Rush-Hour Flags:** `IsPeakHour` (07-09 & 17-19), `IsMorningRush` (07-09), `IsEveningRush` (17-19), `IsNight` (22-05), `IsBusinessHour` (weekday 09-17).
- **Bioclimatic Weather Indices:**
  - *Thom's Discomfort Index (Korean Meteorological Standard):*
    $$\text{DI} = 0.81 \cdot T + 0.01 \cdot \text{RH} \cdot (0.99 \cdot T - 14.3) + 46.3$$
  - *Interaction terms:* $\text{Temperature} \times \text{Humidity}$, $\text{Is\_Raining}$, $\text{Is\_Snowing}$, $\text{High\_Wind}$, $\text{Weather Category}$ (Clear, Rain, Snow, Foggy).

---

## 4. Dual Operational Prediction Modes

To prevent misleading deployment expectations, the system distinguishes between two operational modes:

| Dimension | Mode 1: Weather & Schedule Scenario | Mode 2: Historical Lag-Assisted Dispatch |
|---|---|---|
| **Primary Use Case** | Municipal transit planning, tomorrow's schedule simulation, capacity sizing | Real-time hourly dispatch, station dock rebalancing, dynamic fleet allocation |
| **Input Requirements** | Calendar date, hour, weather forecast (Temp, Rain, Solar, Wind) | Calendar, weather + prior telematics (`Lag_1h`, `Lag_24h`, `Rolling_Mean_3h`) |
| **Target Leakage Risk** | Zero (strictly exogenous predictors) | Zero (lags strictly shifted by $\ge 1$ hour) |
| **Test Performance** | MAE: **142.06**, RMSE: **232.15**, R²: **0.8567** | MAE: **63.19**, RMSE: **105.57**, R²: **0.9703** |

---

## 5. Machine Learning Benchmarks & Baseline Analysis

### 5.1 The Methodological Baseline Finding
In the historical project, baseline models evaluated under a **random 80/20 train_test_split** produced:
- Extra Trees: MAE: 82.80, RMSE: 159.11, R²: 0.9392
- Random Forest: MAE: 96.65, RMSE: 172.57, R²: 0.9285

**Investigation Finding:** A random row shuffle on hourly time series data with `Day` and `DayOfYear` features allows deep decision trees to interpolate between adjacent hours (e.g. using 13:00 and 15:00 to predict 14:00 on the same date). While reproducible, this does not reflect true chronological generalization.

### 5.2 Chronological Test Set Benchmark (Mode 1 — Untouched Future Period)
Evaluated on the final 20% of the year (September 19, 2018 to November 30, 2018; 1,752 untouched hours):

| Model | MAE (bikes) | RMSE (bikes) | R² Score | WAPE (%) | sMAPE (%) | Training Time |
|---|---:|---:|---:|---:|---:|---:|
| **Extra Trees** | **142.06** | **232.15** | **0.8567** | **18.77%** | **24.40%** | 0.93s |
| **Random Forest** | 181.20 | 280.84 | 0.7903 | 23.94% | 33.78% | 2.11s |
| **Gradient Boosting** | 191.97 | 281.23 | 0.7898 | 25.36% | 33.65% | 6.85s |
| **LightGBM** | 198.41 | 287.83 | 0.7798 | 26.21% | 35.16% | 3.03s |
| **HistGradientBoosting** | 201.94 | 288.38 | 0.7789 | 26.68% | 36.13% | 2.52s |
| **CatBoost** | 213.43 | 294.90 | 0.7688 | 28.19% | 38.52% | 1.39s |
| **XGBoost** | 208.42 | 296.44 | 0.7664 | 27.53% | 37.23% | 0.49s |
| **Ridge Regression** | 220.10 | 326.68 | 0.7163 | 29.08% | 39.48% | 0.04s |
| **Decision Tree** | 235.21 | 355.03 | 0.6649 | 31.07% | 39.74% | 0.10s |

### 5.3 Mode 2 Benchmark (Lag-Assisted Operational Dispatch)
When preceding hourly telematics are available, auto-regressive tree ensembles achieve:
- **Extra Trees (Mode 2):** MAE: **63.19**, RMSE: **105.57**, R²: **0.9703**, WAPE: **8.33%**
- **LightGBM (Mode 2):** MAE: **68.17**, RMSE: **114.55**, R²: **0.9651**, WAPE: **8.98%**

---

## 6. Project Structure

```
BIA FINAL PROJECT/
│
├── app.py                      # Streamlit application entry point & modern router
├── main.py                     # Standalone end-to-end CLI workflow
├── config.py                   # Central paths, features, thresholds & theme config
├── requirements.txt            # Locked production dependencies
├── pytest.ini                  # Pytest configuration
├── README.md                   # Complete architectural and operational manual
│
├── data/
│   └── SeoulBikeData.csv       # Preserved original dataset (8,760 rows)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Robust CSV loading, encoding & schema validation
│   ├── preprocessing.py        # Leakage-free transformations, cleaning & splitting
│   ├── feature_engineering.py  # Cyclical, rush-hour, discomfort index & lag engineering
│   ├── model_training.py       # TimeSeriesSplit CV, pipeline fitting & serialization
│   ├── evaluation.py           # Multi-metric evaluation (MAE, RMSE, R2, WAPE, sMAPE)
│   ├── prediction.py           # Real-time inference engine, input validation & bounds
│   └── visualization.py        # Interactive Plotly figures & static high-res PNG export
│
├── pages/
│   ├── dashboard.py            # Page 1: Executive KPI overview & dynamic filters
│   ├── dataset_explorer.py     # Page 2: Paginated data browser, stats & CSV exporter
│   ├── eda.py                  # Page 3: Exploratory Data Analysis & variable relations
│   ├── prediction.py           # Page 4: Real-time scenario prediction & quartile gauges
│   ├── model_performance.py    # Page 5: Benchmarks, residual analysis & baseline review
│   ├── feature_importance.py   # Page 6: Top features, SHAP explainability & causality
│   ├── model_training.py       # Page 7: Interactive retraining console with safeguards
│   └── project_info.py         # Page 8: Architecture, mathematics & Power BI instructions
│
├── models/
│   └── best_model.joblib       # Serialized production pipeline & metadata bundle
│
├── output/
│   ├── Model_Comparison.csv
│   ├── Baseline_Model_Comparison.csv
│   ├── Analysis_Summary.csv
│   ├── Feature_Importance.csv
│   ├── Test_Predictions.csv
│   ├── SeoulBike_Cleaned.csv
│   ├── SeoulBike_Updated_For_PowerBI.csv
│   ├── Descriptive_Statistics.csv
│   ├── run_metadata.json
│   └── visualizations/
│       ├── hourly_rentals.png
│       ├── daily_rentals.png
│       ├── seasonal_rentals.png
│       ├── temperature_rentals.png
│       ├── actual_vs_predicted.png
│       └── feature_importance.png
│
└── tests/
    ├── test_preprocessing.py   # Cleaning, duplicates & chronological split tests
    ├── test_prediction.py      # Inference, input validation & quartile mapping tests
    └── test_model_pipeline.py  # Feature engineering, clipping & metric safety tests
```

---

## 7. Installation & Quickstart Guide (Windows)

### Prerequisites:
- Windows 10 or 11
- Python 3.10, 3.11, or 3.12 (Python 3.12.6 tested and verified)
- PowerShell or Windows Command Prompt

### Step 1: Open PowerShell in Project Root
```powershell
cd "c:\Users\jayan\OneDrive\Desktop\BIA FINAL PROJECT"
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run the End-to-End Pipeline via CLI
Execute data validation, cleaning, feature engineering, baseline reproduction, model benchmarking, and artifact generation:
```powershell
python main.py
```
*(Optional flags: `--mode mode1`, `--mode mode2`, `--tune` to trigger hyperparameter tuning)*

### Step 4: Launch the Interactive Web Application
```powershell
streamlit run app.py
```
The browser will automatically open to `http://localhost:8501`.

---

## 8. Application Features & UI Walkthrough

The web application is styled in **dark navy and charcoal** (`#0b0f19`, `#131b2e`) with **cyan and sky blue accents** (`#00d2ff`, `#38bdf8`):

1. **Dashboard Overview (`pages/dashboard.py`):**
   - Live KPI cards: Total rentals (6.17M), Average demand (705/hr), Peak hour (18:00), Highest season (Summer), Best model test MAE & RMSE.
   - Interactive Plotly charts: Hourly rush-hour envelopes, Weekday vs Weekend profiles, 7-day moving average daily trends, Seasonal violin distributions, Temperature scatter.
   - Comprehensive sidebar filters: Date range picker, Season multi-select, Holiday status, Functioning status, Hour slider.

2. **Dataset Explorer (`pages/dataset_explorer.py`):**
   - Paginated record browser (25, 50, 100, 250 rows per page) preventing browser lag.
   - Live search filter, column visibility toggles, descriptive statistics table.
   - Data cleaning audit & Tukey IQR outlier statistics.
   - One-click filtered CSV export button.

3. **Exploratory Data Analysis (`pages/eda.py`):**
   - Four dedicated exploration tabs: Temporal & Commute, Meteorological Elasticity, Seasonal Shifts, Target Distribution & Correlation Heatmaps.
   - Custom multi-variable explorer allowing user-selected X/Y axes and categorical grouping.

4. **Demand Prediction Interface (`pages/prediction.py`):**
   - Mode 1 vs Mode 2 toggle.
   - Quick-preset scenario buttons: *Sunny Summer Evening Rush*, *Pleasant Spring Commute*, *Freezing Winter Night*, *Heavy Autumn Downpour*.
   - Physical range validation on all sliders/inputs.
   - Output gauge indicator with empirical quartile zones (**Low**, **Moderate**, **High**, **Very High**).
   - 90% confidence interval estimates based on residual error variance.

5. **Model Performance (`pages/model_performance.py`):**
   - Chronological benchmark comparison across all 7 metrics (MAE, MSE, RMSE, R², WAPE, sMAPE, Explained Variance).
   - Actual vs Predicted scatter with 1:1 ideal fit line.
   - Residual scatter and error distribution histogram.
   - Honest side-by-side comparison between the Historical Random Split and the Upgraded Chronological Split.

6. **Feature Importance & Explainability (`pages/feature_importance.py`):**
   - Top 10 and Top 20 ranked predictor horizontal bar charts.
   - Interactive Tree SHAP calculation on test observations.
   - In-depth guide distinguishing statistical importance from physical causal influence.

7. **Model Training Console (`pages/model_training.py`):**
   - Select and compare algorithms interactively.
   - Configure split ratio, lag features, and hyperparameter tuning.
   - Real-time progress bar.
   - Safe model deployment with automated backup protection (`best_model_backup.joblib`).

8. **Project Information (`pages/project_info.py`):**
   - Full system architecture diagram.
   - Mathematical definitions of all metrics (KaTeX rendered).
   - Empirical demand quartile thresholds table.
   - Complete data dictionary and Power BI guide.

---

## 9. Power BI Dashboard Integration Guide

The prepared Power BI report is available at [`dashboards/Bike_Rental_Analytics.pbix`](dashboards/Bike_Rental_Analytics.pbix). Open it with Power BI Desktop, then update the data-source paths if the project is moved to another machine.

The project writes clean, optimized CSV files directly into `output/` for immediate ingestion into Microsoft Power BI Desktop.

### Step-by-Step Instructions:
1. Open **Power BI Desktop** and select **Get data > Text/CSV**.
2. Select `output/SeoulBike_Updated_For_PowerBI.csv`.
3. In Power Query Editor, verify data types:
   - `Date`: Date
   - `Hour`, `IsWeekend`, `IsPeakHour`: Whole Number
   - `Rented Bike Count`, `Predicted_Demand`, `Prediction_Residual`: Decimal / Whole Number
   - `Data Split`, `Seasons`, `Holiday`, `Functioning Day`: Text
4. Select **Close & Apply**.
5. Repeat for `output/Model_Comparison.csv` and `output/Feature_Importance.csv`.

### Recommended Visual Layout:
- **Page 1: Fleet Executive Overview:**
  - Card visuals: Total Rentals (`sum(Rented Bike Count)`), Average Demand (`average(Rented Bike Count)`).
  - Slicers: `Date`, `Seasons`, `Holiday`, `Weather Category`.
  - Line Chart: `Date` vs `Rented Bike Count`.
  - Clustered Column Chart: `Hour` vs `Rented Bike Count`.
- **Page 2: Weather & Environmental Elasticity:**
  - Scatter Chart: `Temperature(°C)` (X) vs `Rented Bike Count` (Y), Legend: `Seasons`.
  - Clustered Bar Chart: `Weather Category` vs Average Rentals.
- **Page 3: Machine Learning Model Evaluation:**
  - Clustered Bar Chart from `Model_Comparison.csv`: `Model` vs `RMSE` and `MAE`.
  - Scatter Chart from `SeoulBike_Updated_For_PowerBI.csv` (filtered to `Data Split = 'Test'`): `Rented Bike Count` (X) vs `Predicted_Demand` (Y).
  - Residual Scatter: `Predicted_Demand` (X) vs `Prediction_Residual` (Y).

---

## 10. Automated Validation & Unit Testing

The project includes an automated test suite verifying data transformation, prediction logic, and pipeline consistency.

To run the test suite:
```powershell
pytest
```
*Expected output: `13 passed in ~3.4s` (100% pass rate).*

### Test Coverage Summary:
- `tests/test_preprocessing.py`:
  - `test_clean_raw_data_valid`: Normal record cleaning.
  - `test_clean_raw_data_handles_duplicates`: Duplicate detection.
  - `test_clean_raw_data_drops_invalid_target`: Target validity constraints.
  - `test_chronological_split_strict_ordering`: Enforces training date $\le$ test date.
  - `test_random_split_reproducibility`: Fixed seed verification.
- `tests/test_prediction.py`:
  - `test_categorize_demand_quartiles`: Validates Low/Moderate/High/Very High boundaries.
  - `test_validate_raw_input_valid`: Passes legitimate weather inputs.
  - `test_validate_raw_input_catches_invalid_ranges`: Rejects out-of-bound hours, temps, etc.
  - `test_predictor_single_and_functioning_day`: Verifies non-negative bounds and 0 closure enforcement.
- `tests/test_model_pipeline.py`:
  - `test_add_calendar_features`: Cyclical sin/cos encodings and rush hour flags.
  - `test_add_weather_features`: Discomfort index and weather categorization.
  - `test_post_process_predictions`: Non-negative clipping and closure rules.
  - `test_evaluation_metrics_zero_safety`: Zero-division safety in WAPE and sMAPE.

---

## 11. Business Insights & Causal Policy Recommendations

1. **Dual Rush-Hour Rebalancing Windows:**
   - Weekday demand is concentrated at **08:00** (morning work commute) and **18:00** (evening return commute).
   - *Operational recommendation:* Fleet rebalancing vans should position excess inventory at suburban residential transit stations prior to 07:00, and reverse distribution back to business districts prior to 16:30.
2. **Precipitation Elasticity (Direct Causal Deterrent):**
   - Rainfall exceeding 2.0 mm/hr precipitates an acute ridership collapse of **>80%**.
   - *Operational recommendation:* Automated telematics can trigger fleet retrieval and suspend rebalancing crews during rain warnings, reducing maintenance costs.
3. **Summer Leisure Plateau:**
   - Weekend demand transitions from dual sharp peaks to an elongated leisure curve spanning 14:00 to 19:00.
   - *Operational recommendation:* Reposition fleet toward riverfront parks (e.g. Han River trails) and recreational hubs on Saturday and Sunday afternoons.
4. **Winter Demand Mitigation:**
   - Freezing temperatures (< -5°C) compress demand to under 20% of summer peaks.
   - *Operational recommendation:* Schedule major annual overhaul, battery servicing, and fleet maintenance during December through February when demand is naturally subdued.

---

## 12. Authors & Credits

- **Project:** Seoul Bike Sharing Demand Prediction and Intelligent Business Analytics System
- **Frameworks:** Python, Streamlit, Scikit-Learn, LightGBM, XGBoost, CatBoost, Plotly, Pandas, NumPy
- **Academic Year:** 2026
