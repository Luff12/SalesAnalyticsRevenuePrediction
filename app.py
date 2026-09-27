
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)

# -------------------------------
# Project paths
# -------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"

st.title("📊 Sales Analytics & Revenue Prediction")
st.caption(
    "Interactive business intelligence dashboard "
    "with historical analysis and revenue forecasting."
)

# -------------------------------
# Load data
# -------------------------------
@st.cache_data
def load_sales():
    return pd.read_csv(
        DATA_DIR / "sales_cleaned.csv",
        parse_dates=["OrderDate"],
    )

@st.cache_data
def load_monthly():
    return pd.read_csv(
        DATA_DIR / "monthly_revenue.csv",
        parse_dates=["MonthStart"],
    )

@st.cache_data
def load_evaluation():
    return pd.read_csv(
        DATA_DIR / "monthly_forecast_evaluation.csv",
        parse_dates=["MonthStart"],
    )

@st.cache_data
def load_future_forecast():
    return pd.read_csv(
        DATA_DIR / "future_revenue_forecast.csv",
        parse_dates=["MonthStart"],
    )

sales = load_sales()
monthly = load_monthly()
evaluation = load_evaluation()
future_forecast = load_future_forecast()

# -------------------------------
# Sidebar filters
# -------------------------------
st.sidebar.header("Dashboard Filters")

regions = sorted(sales["Region"].dropna().unique())
products = sorted(sales["Product"].dropna().unique())

selected_regions = st.sidebar.multiselect(
    "Select Region",
    regions,
    default=regions,
)

selected_products = st.sidebar.multiselect(
    "Select Product",
    products,
    default=products,
)

filtered_sales = sales[
    sales["Region"].isin(selected_regions)
    & sales["Product"].isin(selected_products)
].copy()

if filtered_sales.empty:
    st.warning("No transactions match the selected filters.")
    st.stop()

# -------------------------------
# KPI cards
# -------------------------------
total_revenue = filtered_sales["Revenue"].sum()
total_profit = filtered_sales["Profit"].sum()
total_orders = filtered_sales["OrderID"].nunique()
total_customers = filtered_sales["CustomerID"].nunique()
profit_margin = (
    total_profit / total_revenue * 100
    if total_revenue
    else 0
)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Revenue", f"₹{total_revenue:,.0f}")
col2.metric("Total Profit", f"₹{total_profit:,.0f}")
col3.metric("Orders", f"{total_orders:,}")
col4.metric("Customers", f"{total_customers:,}")
col5.metric("Profit Margin", f"{profit_margin:.1f}%")

st.divider()

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs([
    "📈 Sales Overview",
    "🔎 Product & Region Analysis",
    "🔮 Revenue Forecast",
])

# -------------------------------
# Tab 1: Sales overview
# -------------------------------
with tab1:
    st.subheader("Monthly Revenue Trend")

    filtered_monthly = (
        filtered_sales.assign(
            MonthStart=filtered_sales["OrderDate"]
            .dt.to_period("M")
            .dt.to_timestamp()
        )
        .groupby("MonthStart", as_index=False)["Revenue"]
        .sum()
    )

    fig = px.line(
        filtered_monthly,
        x="MonthStart",
        y="Revenue",
        markers=True,
        title="Monthly Revenue",
        labels={
            "MonthStart": "Month",
            "Revenue": "Revenue (₹)",
        },
    )
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        category_sales = (
            filtered_sales.groupby("Category", as_index=False)["Revenue"]
            .sum()
            .sort_values("Revenue", ascending=False)
        )

        fig = px.bar(
            category_sales,
            x="Category",
            y="Revenue",
            title="Revenue by Category",
            labels={"Revenue": "Revenue (₹)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        channel_sales = (
            filtered_sales.groupby("SalesChannel", as_index=False)["Revenue"]
            .sum()
        )

        fig = px.pie(
            channel_sales,
            names="SalesChannel",
            values="Revenue",
            title="Revenue by Sales Channel",
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Tab 2: Product and region
# -------------------------------
with tab2:
    left, right = st.columns(2)

    with left:
        product_data = (
            filtered_sales.groupby("Product", as_index=False)["Revenue"]
            .sum()
            .sort_values("Revenue", ascending=False)
        )

        fig = px.bar(
            product_data,
            x="Revenue",
            y="Product",
            orientation="h",
            title="Product Revenue Comparison",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        region_data = (
            filtered_sales.groupby("Region", as_index=False)["Profit"]
            .sum()
            .sort_values("Profit", ascending=False)
        )

        fig = px.bar(
            region_data,
            x="Region",
            y="Profit",
            title="Profit by Region",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Transaction Details")

    st.dataframe(
        filtered_sales[
            [
                "OrderID",
                "OrderDate",
                "Product",
                "Category",
                "Region",
                "Quantity",
                "Revenue",
                "Profit",
            ]
        ].sort_values("OrderDate", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    csv_data = filtered_sales.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Filtered Sales CSV",
        data=csv_data,
        file_name="filtered_sales.csv",
        mime="text/csv",
    )

# -------------------------------
# Tab 3: Revenue forecast
# -------------------------------
with tab3:
    st.subheader("Historical Revenue Forecast Evaluation")

    st.info(
        "The model forecasts monthly revenue using historical patterns. "
        "Forecasts are based on synthetic demonstration data."
    )

    if (MODEL_DIR / "forecast_metrics.json").exists():
        with open(MODEL_DIR / "forecast_metrics.json") as file:
            metrics = json.load(file)

        m1, m2, m3 = st.columns(3)
        m1.metric("Model MAE", f"₹{metrics['mae']:,.0f}")
        m2.metric("Model RMSE", f"₹{metrics['rmse']:,.0f}")
        m3.metric("Model R²", f"{metrics['r2']:.3f}")

        st.caption(
            f"Seasonal baseline MAE: ₹{metrics['baseline_mae']:,.0f} "
            f"| Baseline RMSE: ₹{metrics['baseline_rmse']:,.0f}"
        )

    comparison = evaluation.melt(
        id_vars="MonthStart",
        value_vars=[
            "ActualRevenue",
            "PredictedRevenue",
            "BaselineRevenue",
        ],
        var_name="Series",
        value_name="Revenue",
    )

    fig = px.line(
        comparison,
        x="MonthStart",
        y="Revenue",
        color="Series",
        markers=True,
        title="Actual vs Predicted vs Baseline",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Next 12 Months Revenue Forecast")

    future_display = future_forecast.copy()
    future_display["Month"] = (
        future_display["MonthStart"].dt.strftime("%b %Y")
    )

    fig = px.line(
        future_display,
        x="Month",
        y="PredictedRevenue",
        markers=True,
        title="Forecasted Monthly Revenue",
        labels={"PredictedRevenue": "Predicted Revenue (₹)"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        future_display[["Month", "PredictedRevenue"]].rename(
            columns={"PredictedRevenue": "Predicted Revenue (₹)"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    forecast_csv = future_forecast.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Revenue Forecast",
        data=forecast_csv,
        file_name="future_revenue_forecast.csv",
        mime="text/csv",
    )

st.divider()
st.caption("Portfolio project | Python • Pandas • DuckDB • Scikit-learn • Streamlit")