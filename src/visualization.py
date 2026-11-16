"""Visualization module for Seoul Bike Sharing application.

Provides interactive Plotly visualizations designed with a modern dark navy
and cyan business intelligence aesthetic, as well as static Matplotlib exports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

import config

# Color theme constants
DARK_BG = "#0b0f19"
PANEL_BG = "#131b2e"
TEXT_COLOR = "#e2e8f0"
GRID_COLOR = "#1f293d"
CYAN_ACCENT = "#00d2ff"
BLUE_ACCENT = "#3a7bd5"
AMBER_ACCENT = "#fbbf24"
RED_ACCENT = "#f87171"
GREEN_ACCENT = "#34d399"


def apply_dark_layout(fig: go.Figure, title: str, x_title: str = "", y_title: str = "") -> go.Figure:
    """Apply consistent modern dark navy styling to any Plotly figure."""
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(color=TEXT_COLOR, size=16), x=0.02),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(19, 27, 46, 0.6)",
        font=dict(color=TEXT_COLOR, family="Segoe UI, Roboto, Helvetica, sans-serif"),
        xaxis=dict(
            title=x_title,
            color=TEXT_COLOR,
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            tickfont=dict(color=TEXT_COLOR),
        ),
        yaxis=dict(
            title=y_title,
            color=TEXT_COLOR,
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            tickfont=dict(color=TEXT_COLOR),
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            bgcolor="rgba(19, 27, 46, 0.8)",
            bordercolor=GRID_COLOR,
            font=dict(color=TEXT_COLOR),
        ),
        hoverlabel=dict(bgcolor=PANEL_BG, font_color=TEXT_COLOR),
    )
    return fig


def plot_hourly_demand(df: pd.DataFrame) -> go.Figure:
    """Create interactive hourly rental demand profile with commuter rush-hour callouts."""
    hourly_stats = df.groupby("Hour")[config.TARGET_COL].agg(["mean", "std"]).reset_index()
    hourly_stats["mean"] = np.round(hourly_stats["mean"], 1)

    fig = go.Figure()
    # Add shaded standard deviation envelope
    upper = hourly_stats["mean"] + hourly_stats["std"] * 0.5
    lower = np.maximum(0, hourly_stats["mean"] - hourly_stats["std"] * 0.5)

    fig.add_trace(
        go.Scatter(
            x=list(hourly_stats["Hour"]) + list(hourly_stats["Hour"][::-1]),
            y=list(upper) + list(lower[::-1]),
            fill="toself",
            fillcolor="rgba(0, 210, 255, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="Normal Variation",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=hourly_stats["Hour"],
            y=hourly_stats["mean"],
            mode="lines+markers",
            line=dict(color=CYAN_ACCENT, width=3),
            marker=dict(size=8, color=CYAN_ACCENT),
            name="Mean Demand",
            hovertemplate="Hour %{x}:00<br>Average Rentals: <b>%{y}</b> bikes<extra></extra>",
        )
    )

    # Highlight commute peaks (8 AM and 6 PM)
    fig.add_vrect(x0=7.5, x1=9.5, fillcolor="rgba(52, 211, 153, 0.15)", line_width=0, annotation_text="Morning Rush", annotation_position="top left")
    fig.add_vrect(x0=17.5, x1=19.5, fillcolor="rgba(251, 191, 36, 0.15)", line_width=0, annotation_text="Evening Rush", annotation_position="top left")

    fig.update_xaxes(tickmode="linear", tick0=0, dtick=2)
    return apply_dark_layout(fig, "Average Hourly Rental Demand Dynamics", "Hour of Day", "Average Bikes Rented")


def plot_daily_trend(df: pd.DataFrame) -> go.Figure:
    """Create interactive daily total rental demand timeline."""
    date_series = pd.to_datetime(df[config.DATE_COL])
    daily = df.assign(DateOnly=date_series.dt.date).groupby("DateOnly")[config.TARGET_COL].sum().reset_index()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["DateOnly"],
            y=daily[config.TARGET_COL],
            mode="lines",
            line=dict(color=CYAN_ACCENT, width=2),
            name="Daily Total",
            hovertemplate="Date: %{x}<br>Total Rentals: <b>%{y:,}</b><extra></extra>",
        )
    )
    # Add 7-day rolling average
    daily["Rolling7"] = daily[config.TARGET_COL].rolling(7, min_periods=1).mean()
    fig.add_trace(
        go.Scatter(
            x=daily["DateOnly"],
            y=daily["Rolling7"],
            mode="lines",
            line=dict(color=AMBER_ACCENT, width=2.5, dash="dot"),
            name="7-Day Moving Avg",
            hovertemplate="7-Day Avg: <b>%{y:,.0f}</b><extra></extra>",
        )
    )
    return apply_dark_layout(fig, "Daily Total Bike Rental Timeline (Full Year)", "Date", "Total Daily Bikes")


def plot_monthly_rentals(df: pd.DataFrame) -> go.Figure:
    """Create interactive monthly average rental chart."""
    data = df.copy()
    if "Month" not in data.columns:
        data["Month"] = pd.to_datetime(data[config.DATE_COL]).dt.month
    monthly = data.groupby("Month")[config.TARGET_COL].mean().reset_index()
    monthly["Month_Name"] = pd.to_datetime(monthly["Month"], format="%m").dt.strftime("%b")

    fig = px.bar(
        monthly,
        x="Month_Name",
        y=config.TARGET_COL,
        color=config.TARGET_COL,
        color_continuous_scale=["#1e3a8a", "#0284c7", "#00d2ff"],
        labels={config.TARGET_COL: "Avg Rentals", "Month_Name": "Month"},
    )
    fig.update_coloraxes(showscale=False)
    return apply_dark_layout(fig, "Average Rental Demand by Month", "Month", "Average Hourly Bikes")


def plot_seasonal_distribution(df: pd.DataFrame) -> go.Figure:
    """Create interactive seasonal box plot showing demand distribution and medians."""
    season_order = ["Spring", "Summer", "Autumn", "Winter"]
    colors = {"Spring": "#34d399", "Summer": "#00d2ff", "Autumn": "#fbbf24", "Winter": "#60a5fa"}

    fig = go.Figure()
    for season in season_order:
        subset = df[df["Seasons"] == season]
        if not subset.empty:
            fig.add_trace(
                go.Box(
                    y=subset[config.TARGET_COL],
                    name=season,
                    marker_color=colors.get(season, CYAN_ACCENT),
                    boxpoints="outliers",
                )
            )
    return apply_dark_layout(fig, "Seasonal Demand Distribution & Variance", "Season", "Rented Bike Count")


def plot_weekday_vs_weekend(df: pd.DataFrame) -> go.Figure:
    """Compare hourly profile on weekdays vs weekends."""
    data = df.copy()
    if "IsWeekend" not in data.columns:
        data["DayOfWeek"] = pd.to_datetime(data[config.DATE_COL]).dt.dayofweek
        data["IsWeekend"] = (data["DayOfWeek"] >= 5).astype(int)

    weekday = data[data["IsWeekend"] == 0].groupby("Hour")[config.TARGET_COL].mean().reset_index()
    weekend = data[data["IsWeekend"] == 1].groupby("Hour")[config.TARGET_COL].mean().reset_index()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=weekday["Hour"],
            y=weekday[config.TARGET_COL],
            mode="lines+markers",
            name="Weekday (Mon-Fri)",
            line=dict(color=CYAN_ACCENT, width=3),
            hovertemplate="Weekday Hour %{x}:00<br>Demand: <b>%{y:.0f}</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=weekend["Hour"],
            y=weekend[config.TARGET_COL],
            mode="lines+markers",
            name="Weekend (Sat-Sun)",
            line=dict(color=AMBER_ACCENT, width=3, dash="dash"),
            hovertemplate="Weekend Hour %{x}:00<br>Demand: <b>%{y:.0f}</b><extra></extra>",
        )
    )
    fig.update_xaxes(tickmode="linear", tick0=0, dtick=2)
    return apply_dark_layout(fig, "Commuter vs Leisure Patterns: Weekday vs Weekend Hourly Demand", "Hour", "Average Bikes Rented")


def plot_weather_scatter(df: pd.DataFrame, x_col: str, title: Optional[str] = None) -> go.Figure:
    """Scatter plot of meteorological variable vs rental demand with trendline."""
    sample = df.sample(min(2500, len(df)), random_state=config.RANDOM_STATE) if len(df) > 2500 else df
    t_name = title or f"{x_col} versus Rental Demand"

    try:
        import statsmodels
        has_sm = True
    except ImportError:
        has_sm = False

    fig = px.scatter(
        sample,
        x=x_col,
        y=config.TARGET_COL,
        color="Seasons" if "Seasons" in sample.columns else None,
        color_discrete_map={"Spring": "#34d399", "Summer": "#00d2ff", "Autumn": "#fbbf24", "Winter": "#60a5fa"},
        opacity=0.45,
        trendline="ols" if has_sm else None,
        trendline_color_override="#f87171" if has_sm else None,
    )
    return apply_dark_layout(fig, t_name, x_col, "Rented Bike Count")


def plot_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    """Interactive correlation heatmap for numerical variables."""
    num_df = df.select_dtypes(include=[np.number])
    # Select most relevant variables
    cols_to_show = [c for c in num_df.columns if not c.endswith("_Sin") and not c.endswith("_Cos")][:14]
    corr = num_df[cols_to_show].corr().round(2)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale=[[0, "#1e3a8a"], [0.5, "#0f172a"], [1, "#00d2ff"]],
            zmin=-1,
            zmax=1,
            text=corr.values,
            texttemplate="%{text}",
            textfont=dict(size=10, color=TEXT_COLOR),
            hoverongaps=False,
        )
    )
    return apply_dark_layout(fig, "Feature Correlation Heatmap Matrix", "", "")


def plot_rental_distribution(df: pd.DataFrame) -> go.Figure:
    """Distribution of target variable with quartile markers."""
    fig = px.histogram(
        df,
        x=config.TARGET_COL,
        nbins=45,
        marginal="box",
        color_discrete_sequence=[CYAN_ACCENT],
        opacity=0.8,
    )
    return apply_dark_layout(fig, "Distribution of Hourly Rented Bike Count", "Rented Bike Count", "Frequency")


def plot_actual_vs_predicted(test_pred_df: pd.DataFrame, model_name: str = "Model") -> go.Figure:
    """Scatter plot of actual vs predicted rental counts with ideal reference line."""
    sample = test_pred_df.sample(min(1500, len(test_pred_df)), random_state=42) if len(test_pred_df) > 1500 else test_pred_df

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sample["Actual"],
            y=sample["Predicted"],
            mode="markers",
            marker=dict(size=6, color=CYAN_ACCENT, opacity=0.5),
            name="Predictions",
            hovertemplate="Actual: %{x}<br>Predicted: %{y}<extra></extra>",
        )
    )
    # Perfect fit line
    max_val = max(float(test_pred_df["Actual"].max()), float(test_pred_df["Predicted"].max()))
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color=AMBER_ACCENT, width=2, dash="dash"),
            name="Ideal 1:1 Fit",
        )
    )
    return apply_dark_layout(fig, f"Actual vs Predicted Rentals: {model_name}", "Actual Demand", "Predicted Demand")


def plot_residuals(test_pred_df: pd.DataFrame, model_name: str = "Model") -> go.Figure:
    """Residual scatter plot (Residuals vs Predicted Demand)."""
    sample = test_pred_df.sample(min(1500, len(test_pred_df)), random_state=42) if len(test_pred_df) > 1500 else test_pred_df

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sample["Predicted"],
            y=sample["Residual"],
            mode="markers",
            marker=dict(size=6, color=CYAN_ACCENT, opacity=0.5),
            name="Residuals",
            hovertemplate="Predicted: %{x}<br>Residual: %{y}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color=RED_ACCENT, line_width=2)
    return apply_dark_layout(fig, f"Residual Analysis: {model_name}", "Predicted Demand", "Residual (Actual - Predicted)")


def plot_residual_distribution(test_pred_df: pd.DataFrame, model_name: str = "Model") -> go.Figure:
    """Residual histogram distribution."""
    fig = px.histogram(
        test_pred_df,
        x="Residual",
        nbins=40,
        marginal="violin",
        color_discrete_sequence=[AMBER_ACCENT],
        opacity=0.8,
    )
    fig.add_vline(x=0, line_dash="dash", line_color=RED_ACCENT, line_width=2)
    return apply_dark_layout(fig, f"Residual Error Distribution: {model_name}", "Residual Error", "Frequency")


def plot_model_comparison_bar(comparison_df: pd.DataFrame, metric: str = "RMSE") -> go.Figure:
    """Bar chart comparing all models across selected evaluation metric."""
    sorted_df = comparison_df.sort_values(metric, ascending=(metric in ["MAE", "MSE", "RMSE", "WAPE", "sMAPE"]))
    fig = px.bar(
        sorted_df,
        x="Model",
        y=metric,
        color=metric,
        color_continuous_scale=["#00d2ff", "#1e3a8a"] if metric in ["R2", "Explained_Variance"] else ["#1e3a8a", "#00d2ff"],
        text_auto=".2f",
    )
    fig.update_coloraxes(showscale=False)
    return apply_dark_layout(fig, f"Model Comparison by {metric}", "Model", metric)


def plot_feature_importance_bar(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart showing ranked top feature importances."""
    top_df = importance_df.head(top_n).sort_values("Importance", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=top_df["Importance"],
            y=top_df["Feature"],
            orientation="h",
            marker=dict(
                color=top_df["Importance"],
                colorscale=[[0, "#1e3a8a"], [1, "#00d2ff"]],
            ),
            text=np.round(top_df["Importance"], 3),
            textposition="auto",
        )
    )
    return apply_dark_layout(fig, f"Top {top_n} Most Influential Predictors", "Importance Score", "Feature")


def plot_prediction_gauge(pred_value: float, category: str) -> go.Figure:
    """Sleek semi-circular gauge indicator showing predicted demand within category bands."""
    color_map = config.DEMAND_CATEGORY_COLORS
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pred_value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"<b>{category.upper()} DEMAND</b>", "font": {"color": color_map.get(category, CYAN_ACCENT), "size": 18}},
            number={"font": {"color": TEXT_COLOR, "size": 42}, "suffix": " bikes"},
            gauge={
                "axis": {"range": [0, 2500], "tickcolor": TEXT_COLOR, "tickwidth": 1},
                "bar": {"color": color_map.get(category, CYAN_ACCENT), "thickness": 0.3},
                "bgcolor": PANEL_BG,
                "borderwidth": 1,
                "bordercolor": GRID_COLOR,
                "steps": [
                    {"range": [0, 200], "color": "rgba(56, 189, 248, 0.2)"},
                    {"range": [200, 550], "color": "rgba(52, 211, 153, 0.2)"},
                    {"range": [550, 1100], "color": "rgba(251, 191, 36, 0.2)"},
                    {"range": [1100, 2500], "color": "rgba(248, 113, 113, 0.2)"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 3},
                    "thickness": 0.8,
                    "value": pred_value,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR),
        height=260,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def export_static_visualizations(
    df: pd.DataFrame,
    test_pred_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> None:
    """Save high-resolution static PNG figures to output/visualizations/ for reports & Power BI."""
    out = output_dir or config.VIS_DIR
    out.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # 1. Hourly demand profile
    fig, ax = plt.subplots(figsize=(10, 5))
    hourly = df.groupby("Hour")[config.TARGET_COL].mean().reset_index()
    sns.lineplot(data=hourly, x="Hour", y=config.TARGET_COL, marker="o", color="#0284c7", ax=ax, linewidth=2.5)
    ax.set(title="Average Hourly Bike Rental Demand", xlabel="Hour of Day", ylabel="Average Bikes")
    ax.set_xticks(range(24))
    fig.tight_layout()
    fig.savefig(out / "hourly_rentals.png", dpi=150)
    plt.close(fig)

    # 2. Daily demand timeline
    fig, ax = plt.subplots(figsize=(11, 5))
    dt_series = pd.to_datetime(df[config.DATE_COL])
    daily = df.assign(DateOnly=dt_series.dt.date).groupby("DateOnly")[config.TARGET_COL].sum().reset_index()
    sns.lineplot(data=daily, x="DateOnly", y=config.TARGET_COL, color="#0284c7", ax=ax)
    ax.set(title="Daily Total Bike Rentals (Seoul)", xlabel="Date", ylabel="Total Rented Bikes")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "daily_rentals.png", dpi=150)
    plt.close(fig)

    # 3. Seasonal demand
    fig, ax = plt.subplots(figsize=(8, 5))
    season_avg = df.groupby("Seasons")[config.TARGET_COL].mean().reset_index()
    sns.barplot(data=season_avg, x="Seasons", y=config.TARGET_COL, hue="Seasons", palette="Blues_r", legend=False, ax=ax)
    ax.set(title="Average Bike Rental Demand by Season", xlabel="Season", ylabel="Average Rented Bikes")
    fig.tight_layout()
    fig.savefig(out / "seasonal_rentals.png", dpi=150)
    plt.close(fig)

    # 4. Temperature vs demand
    fig, ax = plt.subplots(figsize=(9, 5))
    sample = df.sample(min(2000, len(df)), random_state=42)
    sns.scatterplot(data=sample, x="Temperature(°C)", y=config.TARGET_COL, alpha=0.35, s=25, color="#0284c7", ax=ax)
    sns.regplot(data=sample, x="Temperature(°C)", y=config.TARGET_COL, scatter=False, color="#dc2626", ax=ax)
    ax.set(title="Temperature versus Rental Demand", xlabel="Temperature (°C)", ylabel="Rented Bikes")
    fig.tight_layout()
    fig.savefig(out / "temperature_rentals.png", dpi=150)
    plt.close(fig)

    # 5. Actual vs Predicted
    if not test_pred_df.empty:
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.scatterplot(data=test_pred_df.sample(min(1200, len(test_pred_df)), random_state=42), x="Actual", y="Predicted", alpha=0.4, color="#0284c7", ax=ax)
        mx = max(test_pred_df["Actual"].max(), test_pred_df["Predicted"].max())
        ax.plot([0, mx], [0, mx], "--", color="black", linewidth=1.5)
        ax.set(title="Actual versus Predicted Demand", xlabel="Actual Rentals", ylabel="Predicted Rentals")
        fig.tight_layout()
        fig.savefig(out / "actual_vs_predicted.png", dpi=150)
        plt.close(fig)

    # 6. Feature Importance
    if not importance_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        top_imp = importance_df.head(15).sort_values("Importance", ascending=True)
        sns.barplot(data=top_imp, x="Importance", y="Feature", color="#0284c7", ax=ax)
        ax.set(title="Top Feature Importances", xlabel="Importance Score", ylabel="Predictor")
        fig.tight_layout()
        fig.savefig(out / "feature_importance.png", dpi=150)
        plt.close(fig)
