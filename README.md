
# Seoul Bike Sharing Demand Prediction & Business Analytics

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Project Overview

Bike-sharing services have become an important part of urban transportation. However, managing bike availability can be challenging because customer demand changes throughout the day depending on weather, seasons, and travel patterns.

This project focuses on predicting hourly bike rental demand in Seoul, South Korea, using Machine Learning techniques.

The system analyzes historical rental data, weather conditions, and time-based patterns to estimate future demand. It also provides an interactive Streamlit dashboard and Power BI reports to make the results easier to understand.

The main goal is to help understand rental patterns, improve bike distribution, and support better operational planning.

## Project Highlights

- Analyzed 8,760 hourly records from the Seoul Bike Sharing dataset.
- Developed Machine Learning models to predict bike rental demand.
- Compared 9 different regression algorithms.
- Used chronological train-test splitting to avoid data leakage.
- Implemented two prediction modes for planning and operational use.
- Created interactive dashboards using Streamlit and Plotly.
- Integrated analytical outputs with Power BI.
- Added automated testing to validate the prediction pipeline.

## Technologies Used

| Category | Tools |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Plotly |
| Machine Learning | Scikit-Learn, XGBoost, LightGBM, CatBoost |
| Web Application | Streamlit |
| Business Intelligence | Power BI |
| Testing | Pytest |

## Dataset Information

The project uses the Seoul Bike Sharing Demand Dataset.

It contains 8,760 hourly observations collected between December 2017 and November 2018.

Important features include:

- Date and Hour
- Temperature and Humidity
- Wind Speed and Visibility
- Rainfall and Snowfall
- Seasons
- Holiday Information
- Functioning Day
- Rented Bike Count

The target variable is **Rented Bike Count**, which represents the number of bikes rented during each hour.

## Machine Learning Approach

The project follows a complete Machine Learning workflow:

1. Data Loading – Importing and validating the original dataset.
2. Data Preprocessing – Handling missing values, encoding categorical variables, and scaling numerical features.
3. Feature Engineering – Creating time-based, weather-related, and rush-hour features.
4. Model Training – Training and comparing multiple regression algorithms.
5. Model Evaluation – Measuring performance using MAE, RMSE, R², WAPE, and sMAPE.
6. Prediction – Estimating bike rental demand using different input scenarios.

### Models Evaluated

The following 9 algorithms were compared:

- Extra Trees Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- LightGBM
- HistGradientBoosting
- CatBoost
- XGBoost
- Ridge Regression
- Decision Tree Regressor

### Best Model Results

The Extra Trees model achieved the best performance in both prediction modes.

| Metric | Mode 1 | Mode 2 |
|---|---:|---:|
| MAE | 142.06 | 63.19 |
| RMSE | 232.15 | 105.57 |
| R² Score | 0.8567 | 0.9703 |
| WAPE | 18.77% | 8.33% |

**Mode 1:** Uses weather and calendar information to simulate future rental demand.

**Mode 2:** Uses historical rental patterns along with weather and calendar information for operational predictions.

## Application Features

The project includes an interactive Streamlit application with the following sections:

1. **Dashboard:** Displays important rental statistics, demand trends, and key performance indicators.
2. **Dataset Explorer:** Allows users to browse, filter, analyze, and export dataset records.
3. **Exploratory Data Analysis:** Helps understand the relationship between rental demand, weather, and time.
4. **Demand Prediction:** Allows users to enter different weather and calendar conditions to estimate bike rentals.
5. **Model Performance:** Compares different Machine Learning algorithms and their prediction errors.
6. **Feature Importance:** Identifies the features that contribute most to the model's predictions.
7. **Model Training:** Provides options to train and compare models.
8. **Project Information:** Explains the system architecture, evaluation metrics, and Power BI integration.

## Application Screenshots

### Power BI Dashboard

<img width="1282" height="721" alt="Screenshot 2026-11-16 195409" src="https://github.com/user-attachments/assets/9293557c-dfa6-4306-9a28-2ad2e61af839" />

## Power BI Integration

The project also supports Power BI for business-level analysis.

The generated CSV files can be imported into Power BI Desktop to create interactive reports.

Recommended dashboard sections:

- Overall Bike Rental Overview
- Hourly and Seasonal Demand Analysis
- Weather Impact on Bike Rentals
- Machine Learning Model Comparison
- Actual vs Predicted Demand

## Project Structure

```text
BIA FINAL PROJECT/
│
├── app.py
├── main.py
├── config.py
├── requirements.txt
├── README.md
│
├── data/
├── src/
├── pages/
├── models/
├── output/
├── dashboards/
├── tests/
└── assets/
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the Project Folder

```bash
cd "BIA FINAL PROJECT"
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Machine Learning Pipeline

```bash
python main.py
```

### 5. Launch the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## Testing

The project includes automated tests to check preprocessing, feature engineering, prediction logic, and model pipeline consistency.

Run the tests using:

```bash
pytest
```

The expected test result is 13 passed tests.

## Key Business Insights

The analysis provides several useful observations:

- **Rush Hour Demand:** Bike rentals increase during morning and evening commuting hours.
- **Weather Impact:** Rainfall has a strong negative relationship with bike rental demand.
- **Seasonal Patterns:** Summer generally experiences higher rental activity.
- **Winter Demand:** Extremely cold weather is associated with lower rental demand.

These insights can help bike-sharing operators plan fleet distribution, improve availability, and manage resources more effectively.

## Future Improvements

- Integrate live weather information.
- Connect real-time bike availability data.
- Deploy the prediction system to a cloud platform.
- Improve forecasting using additional time-series models.
- Expand the system to support multiple cities.

## Conclusion

This project combines Machine Learning, data analytics, and business intelligence to understand and predict bike-sharing demand.

By analyzing historical rental patterns and environmental conditions, the system provides useful insights into how bike demand changes over time.

It also demonstrates how predictive analytics can support real-world transportation planning and decision-making.

---

**Developed as an academic Business Intelligence and Analytics project | 2026**
