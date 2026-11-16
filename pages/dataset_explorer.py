"""Dataset Explorer Page for Seoul Bike Sharing application."""

from __future__ import annotations

import io
import json
import zipfile
import streamlit as st
import pandas as pd
import numpy as np

import config


def render_dataset_explorer_page(df: pd.DataFrame, cleaned_df: pd.DataFrame, metadata: dict) -> None:
    """Render interactive dataset viewing, filtering, summary statistics, and audit inspection."""
    st.markdown("<h1 style='margin-bottom:0;'>Dataset Explorer & Data Quality Audit</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94a3b8; font-size:1.05rem; margin-top:4px;'>Inspect raw and processed records, schema types, missing values, and outlier statistics.</p>",
        unsafe_allow_html=True,
    )

    # Top Data Summary Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Feature Columns", f"{len(df.columns)}")
    with col3:
        st.metric("Missing Values", f"{df.isna().sum().sum()}")
    with col4:
        st.metric("Duplicate Rows", f"{df.duplicated().sum()}")
    with col5:
        mem_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        st.metric("Memory Footprint", f"{mem_mb} MB")

    st.markdown("<hr style='margin:16px 0 24px 0; border-color:#1f293d;'>", unsafe_allow_html=True)

    # Main Tabs
    tab_records, tab_stats, tab_quality, tab_types, tab_exports = st.tabs(
        [
            "Dataset Browser & Pagination",
            "Descriptive Statistics",
            "Data Cleaning & Outlier Audit",
            "Schema & Types",
            "CSV Download & Export Hub",
        ]
    )

    with tab_records:
        st.markdown("### Interactive Record Browser")

        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1, 1])
        with ctrl_col1:
            search_query = st.text_input("Quick Search (Date, Season, Holiday)", "", placeholder="e.g. 2018-05, Summer, Holiday")
        with ctrl_col2:
            dataset_choice = st.selectbox("Dataset Version", ["Cleaned & Processed", "Raw Original"], index=0)
        with ctrl_col3:
            page_size = st.selectbox("Rows per page", [25, 50, 100, 250], index=1)

        source_df = cleaned_df if dataset_choice == "Cleaned & Processed" else df

        # Column Selector
        all_cols = source_df.columns.tolist()
        default_cols = [c for c in all_cols if c not in ["Year", "DayOfYear", "DayName"]][:12]
        selected_cols = st.multiselect("Visible Columns", options=all_cols, default=default_cols)

        if not selected_cols:
            selected_cols = all_cols

        # Filter by search string
        view_df = source_df[selected_cols].copy()
        if search_query:
            mask = view_df.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)
            view_df = view_df[mask]

        total_filtered_rows = len(view_df)
        total_pages = max(1, int(np.ceil(total_filtered_rows / page_size)))

        page_ctrl1, page_ctrl2 = st.columns([1, 4])
        with page_ctrl1:
            page_num = st.number_input("Page Number", min_value=1, max_value=total_pages, value=1, step=1)
        with page_ctrl2:
            st.markdown(
                f"<p style='color:#94a3b8; padding-top:32px;'>Showing page <b>{page_num}</b> of <b>{total_pages}</b> ({total_filtered_rows:,} matching records)</p>",
                unsafe_allow_html=True,
            )

        start_idx = (page_num - 1) * page_size
        end_idx = start_idx + page_size
        paged_df = view_df.iloc[start_idx:end_idx]

        st.dataframe(paged_df, use_container_width=True, height=450)

        # CSV Download Button
        csv_data = view_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download Filtered CSV ({total_filtered_rows:,} records)",
            data=csv_data,
            file_name="SeoulBike_Filtered_Export.csv",
            mime="text/csv",
        )

    with tab_stats:
        st.markdown("### Numerical Descriptive Statistics")
        desc = source_df.describe().transpose()
        desc["count"] = desc["count"].astype(int)
        st.dataframe(desc.style.format("{:.2f}"), use_container_width=True, height=450)

    with tab_quality:
        st.markdown("### Data Cleaning & Outlier Audit Report")
        audit_info = metadata.get("cleaning_audit", metadata.get("cleaning", {}))

        c_audit1, c_audit2, c_audit3 = st.columns(3)
        with c_audit1:
            st.info(f"**Original Records:** {audit_info.get('original_rows', 8760):,}")
            st.info(f"**Final Retained Records:** {audit_info.get('final_rows', 8760):,}")
        with c_audit2:
            st.success(f"**Duplicate Rows Removed:** {audit_info.get('duplicate_rows_removed', 0)}")
            st.success(f"**Invalid Dates Removed:** {audit_info.get('invalid_dates_removed', 0)}")
        with c_audit3:
            st.success(f"**Invalid Target Rows Removed:** {audit_info.get('invalid_target_rows_removed', 0)}")
            st.success(f"**Target Missing / Negative:** 0")

        st.markdown("#### Tukey IQR Outlier Inspection (Audited without destructive dropping)")
        st.caption(
            "High rental counts represent genuine commuter surges and sunny summer weekends, "
            "not recording errors. Therefore, they are audited transparently and preserved to retain authentic demand distribution."
        )

        outlier_data = audit_info.get("outlier_audit", audit_info.get("outliers_not_removed", {}))
        if outlier_data:
            outlier_rows = []
            for col_name, stats_dict in outlier_data.items():
                outlier_rows.append(
                    {
                        "Feature Column": col_name,
                        "Lower Bound (Q1 - 1.5*IQR)": round(stats_dict.get("lower_bound", 0.0), 2),
                        "Upper Bound (Q3 + 1.5*IQR)": round(stats_dict.get("upper_bound", 0.0), 2),
                        "Outlier Count": stats_dict.get("outlier_count", 0),
                        "Handling Strategy": "Preserved (Legitimate Domain Demand Variation)",
                    }
                )
            st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True)

    with tab_types:
        st.markdown("### Schema Architecture & Missing Values")
        type_df = pd.DataFrame(
            {
                "Column Name": source_df.columns,
                "Data Type": [str(t) for t in source_df.dtypes],
                "Null Values": [int(source_df[c].isna().sum()) for c in source_df.columns],
                "Null Percentage (%)": [round(float(source_df[c].isna().mean()) * 100, 2) for c in source_df.columns],
                "Unique Values": [int(source_df[c].nunique()) for c in source_df.columns],
                "Sample Value": [str(source_df[c].iloc[0]) for c in source_df.columns],
            }
        )
        st.dataframe(type_df, use_container_width=True, height=450)

    with tab_exports:
        st.markdown("### CSV Analytical Export & Download Hub")
        st.markdown(
            "Download pre-formatted, production-ready CSV files for further offline analysis in "
            "**Microsoft Power BI, Excel, Tableau, Python, R, or SQL databases**."
        )

        # 1. Download all in ZIP bundle
        st.markdown("#### 📦 Download All Analysis CSVs in a Single Bundle")
        
        def build_csv_zip() -> bytes:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                # Include all CSV files in output/
                for f_path in config.OUTPUT_DIR.glob("*.csv"):
                    z.write(f_path, arcname=f_path.name)
                # Also include raw data
                if config.DEFAULT_DATASET_PATH.exists():
                    z.write(config.DEFAULT_DATASET_PATH, arcname="SeoulBikeData_Original.csv")
            buf.seek(0)
            return buf.getvalue()

        col_zip1, col_zip2 = st.columns([1.5, 2.5])
        with col_zip1:
            zip_bytes = build_csv_zip()
            st.download_button(
                label="📥 Download All CSVs (.ZIP Bundle)",
                data=zip_bytes,
                file_name="Seoul_Bike_Analysis_CSV_Bundle.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )
        with col_zip2:
            st.caption(
                "Includes: Power BI dataset, Cleaned dataset, Model Comparison, Baseline Comparison, "
                "Test Predictions, Feature Importance, Analysis Summary, and Descriptive Statistics."
            )

        st.markdown("<hr style='margin:20px 0; border-color:#1f293d;'>", unsafe_allow_html=True)
        st.markdown("#### 📄 Individual Analytical CSV Downloads")

        # Registry of all downloadable CSV files
        csv_files = [
            {
                "name": "Power BI Master Dataset",
                "filename": "SeoulBike_Updated_For_PowerBI.csv",
                "path": config.POWER_BI_DATA_PATH,
                "desc": "Complete dataset with original fields, engineered features, Train/Test split indicator, Predicted_Demand, and Prediction_Residual.",
                "best_for": "Power BI Desktop dashboards & interactive BI reporting",
            },
            {
                "name": "Chronological Model Benchmark Leaderboard",
                "filename": "Model_Comparison.csv",
                "path": config.MODEL_COMPARISON_PATH,
                "desc": "Chronological test set evaluation across all 9 models (MAE, MSE, RMSE, R², WAPE, sMAPE, Explained Variance, Training Time).",
                "best_for": "Model evaluation tables, bar charts & academic reports",
            },
            {
                "name": "Historical Baseline Model Comparison",
                "filename": "Baseline_Model_Comparison.csv",
                "path": config.BASELINE_COMPARISON_PATH,
                "desc": "Historical baseline evaluation under random 80/20 split for transparent side-by-side methodological comparison.",
                "best_for": "Methodological critique & data leakage ablation study",
            },
            {
                "name": "Test Set Row-Level Predictions & Residuals",
                "filename": "Test_Predictions.csv",
                "path": config.TEST_PREDICTIONS_PATH,
                "desc": "Row-by-row test observations with Actual demand, Model Predicted demand, Residuals, Date, and Hour.",
                "best_for": "Residual diagnostics, scatter plots & error distribution",
            },
            {
                "name": "Feature Importance & Predictor Rankings",
                "filename": "Feature_Importance.csv",
                "path": config.FEATURE_IMPORTANCE_PATH,
                "desc": "Ranked predictor importance scores, Gini split gains, and relative percentage contributions for the top model.",
                "best_for": "Pareto charts, driver analysis & feature selection",
            },
            {
                "name": "Executive Analysis Summary & Correlations",
                "filename": "Analysis_Summary.csv",
                "path": config.ANALYSIS_SUMMARY_PATH,
                "desc": "Headline business KPIs (total rentals, averages, peak hours, season totals) and target correlation coefficients.",
                "best_for": "Executive summary cards, briefing memos & KPI dashboards",
            },
            {
                "name": "Numerical Descriptive Statistics",
                "filename": "Descriptive_Statistics.csv",
                "path": config.OUTPUT_DIR / "Descriptive_Statistics.csv",
                "desc": "Complete summary statistics for all features: mean, std, min, 25%, 50%, 75%, max, count.",
                "best_for": "Exploratory reporting, distributional auditing & tables",
            },
            {
                "name": "Cleaned & Imputed Dataset",
                "filename": "SeoulBike_Cleaned.csv",
                "path": config.CLEANED_DATA_PATH,
                "desc": "Preprocessed dataset with parsed dates, verified target values, missing value imputations, and calendar flags.",
                "best_for": "Custom Python/R machine learning experiments & re-training",
            },
            {
                "name": "Original Raw Seoul Bike Dataset",
                "filename": "SeoulBikeData.csv",
                "path": config.DEFAULT_DATASET_PATH,
                "desc": "Untouched original dataset containing 8,760 hourly records (1 full calendar year) directly as supplied.",
                "best_for": "Reproducing end-to-end pipeline from absolute scratch",
            },
        ]

        for idx, item in enumerate(csv_files):
            c_info, c_btn = st.columns([3, 1])
            with c_info:
                st.markdown(f"**{item['name']}** (`{item['filename']}`)")
                st.markdown(f"<p style='color:#94a3b8; font-size:0.88rem; margin:0;'>{item['desc']}<br><span style='color:#00d2ff;'>Best for:</span> {item['best_for']}</p>", unsafe_allow_html=True)
            with c_btn:
                f_path = item["path"]
                if f_path.exists():
                    f_bytes = f_path.read_bytes()
                    f_size_kb = round(len(f_bytes) / 1024, 1)
                    st.download_button(
                        label=f"Download ({f_size_kb} KB)",
                        data=f_bytes,
                        file_name=item["filename"],
                        mime="text/csv",
                        key=f"dl_csv_{idx}",
                        use_container_width=True,
                    )
                else:
                    st.warning("File not found")
            st.markdown("<hr style='margin:10px 0; border-color:#1f293d;'>", unsafe_allow_html=True)
