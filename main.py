# === Import Libraries ===
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import statsmodels.api as sm
import json
import os

# === Streamlit Config ===
st.set_page_config(layout="wide")
st.title("📊 Supply Chain Forecasting Dashboard")

# === Sidebar for Navigation ===
st.markdown("""
    <style>
        .css-1d391kg { 
            color: blue !important; 
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")



# === Load JSON Data ===
# Load Data from JSON
file_path = os.path.join(os.getcwd(), "data", "cust_stock.json")
if not os.path.exists(file_path):
    st.error("File not found: cust_stock.json")
    st.stop()

with open(file_path, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data["items"])




# === Preprocessing ===
df["txndate"] = pd.to_datetime(df["txndate"])
df.sort_values("txndate", inplace=True)

# Calculate demand and lead time estimates
df["daily_demand"] = df["qty"] / 30
df["leadtime"] = 5
df["lead_time_stock"] = df["qty"] - (df["daily_demand"] * df["leadtime"])
df["year_month"] = df["txndate"].dt.to_period("M").astype(str)





df["txndate"] = pd.to_datetime(df["txndate"])
df.sort_values("txndate", inplace=True)

# Calculate demand and lead time estimates
df["daily_demand"] = df["qty"] / 30
df["leadtime"] = 5
df["lead_time_stock"] = df["qty"] - (df["daily_demand"] * df["leadtime"])
df["year_month"] = df["txndate"].dt.to_period("M").astype(str)



# === Key Points (Sidebar) ===
st.sidebar.subheader("Key Points")

# Collapsible section for low stock alert
with st.sidebar.expander("⚠️ Low Stock Alerts"):
    low_stock_items = df[df["lead_time_stock"] < 100]  # Items with stock less than 100 units
    st.markdown(f"{len(low_stock_items)} items are low in stock.")
    for index, item in low_stock_items.iterrows():
        st.markdown(f"- **{item['description']}** (Qty: {item['qty']})")

# Collapsible section for customer predictions
with st.sidebar.expander("🔮 Customer Demand Prediction"):
    # Example of using ARIMA or other models for forecasting customer demand
    forecast_data = []
    for desc in df["description"].unique():
        item_df = df[df["description"] == desc]
        monthly_series = item_df.resample("M", on="txndate")["qty"].sum()
        monthly_series = monthly_series.fillna(0)
        if len(monthly_series) >= 6:  # Minimum months of data
            try:
                model = ARIMA(monthly_series, order=(1, 1, 1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=12)  # Predict 12 months of future demand
                total_forecast = forecast.sum()
                forecast_data.append({"description": desc, "forecast_qty": total_forecast})
            except:
                continue

    forecast_df = pd.DataFrame(forecast_data)
    forecast_df = forecast_df.sort_values("forecast_qty", ascending=False).head(10)
    st.write("Top 10 predicted items based on customer demand for the next 12 months:")
    st.dataframe(forecast_df)

# Collapsible section for inventory analysis
with st.sidebar.expander("📊 Inventory Analysis"):
    # Example: Show the top 5 items by total quantity in stock
    top_items = df.groupby("description")["qty"].sum().reset_index().sort_values("qty", ascending=False).head(5)
    st.write("Top 5 Items by Stock Quantity:")
    st.dataframe(top_items)




# === KPIs ===
total_stock = df["qty"].sum()
total_value = df["stockvalue"].sum()
avg_stock_age = df[["aging_60", "aging_90", "aging_180", "aging_180plus"]].sum(axis=1).mean()

# === KPIs (Styled Box Metrics) ===
col1, col2, col3 = st.columns(3)

# Metric 1: Total Stock Quantity
col1.markdown(
    f"""
    <div style="background-color:#f4f4f4; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); text-align:center;">
        <h3>📦 Total Stock Quantity</h3>
        <p style="font-size: 24px; font-weight: bold;">{total_stock:,}</p>
    </div>
    """, unsafe_allow_html=True
)

# Metric 2: Total Inventory Value
col2.markdown(
    f"""
    <div style="background-color:#f4f4f4; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); text-align:center;">
        <h3>💰 Total Inventory Value</h3>
        <p style="font-size: 24px; font-weight: bold;">${total_value:,.2f}</p>
    </div>
    """, unsafe_allow_html=True
)

# Metric 3: Avg Stock Age
col3.markdown(
    f"""
    <div style="background-color:#f4f4f4; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); text-align:center;">
        <h3>⏳ Avg Stock Age</h3>
        <p style="font-size: 24px; font-weight: bold;">{avg_stock_age:.1f} days</p>
    </div>
    """, unsafe_allow_html=True
)




# === Preprocessing ===
df["txndate"] = pd.to_datetime(df["txndate"])
df["year"] = df["txndate"].dt.year
df["month"] = df["txndate"].dt.month
df["description"] = df["description"].astype(str)

# === Top Trending Items ===
st.subheader("🔥 Top Trending Items (Bar Chart)")

count_option = st.selectbox("Select number of top items to show:", [10, 50, 100], index=0)

top_items_df = df.groupby("description")["qty"].sum().reset_index()
top_items_df = top_items_df.sort_values("qty", ascending=False).head(count_option)

# Color scale and orientation based on the count_option
if count_option == 10:
    fig1 = px.bar(top_items_df, x="description", y="qty", color="qty",
                  title=f"Top {count_option} Trending Items (by Quantity)",
                  labels={"qty": "Total Qty", "description": "Item"})
    fig1.update_layout(xaxis={'categoryorder': 'total descending'})
else:
    fig1 = px.bar(top_items_df, x="qty", y="description", color="qty", orientation="h",
                  title=f"Top {count_option} Trending Items (by Quantity)",
                  labels={"qty": "Total Qty", "description": "Item"})
    fig1.update_layout(yaxis={'categoryorder': 'total ascending'})

# Plotting the first chart
st.plotly_chart(fig1, use_container_width=True)

# === Prediction for 2025 Inventory Demand ===
st.subheader("🔮 2025 Forecast: Items Likely to Be Bought Again")

# Get top 50 items to apply prediction
top_items = df.groupby("description")["qty"].sum().sort_values(ascending=False).head(50).index

forecast_data = []
for desc in top_items:
    item_df = df[df["description"] == desc]
    monthly_series = item_df.resample("M", on="txndate")["qty"].sum()
    monthly_series = monthly_series.fillna(0)
    if len(monthly_series) >= 6:  # Minimum months of data
        try:
            model = ARIMA(monthly_series, order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=12)  # Predict 12 months of 2025
            total_forecast_2025 = forecast.sum()
            forecast_data.append({"description": desc, "forecast_qty": total_forecast_2025})
        except:
            continue

forecast_df = pd.DataFrame(forecast_data)
forecast_df = forecast_df.sort_values("forecast_qty", ascending=False).head(20)

# Plotting the forecasted items chart
fig2 = px.bar(forecast_df, x="forecast_qty", y="description", color="forecast_qty", orientation="h",
              title="📦 Predicted Top Items for 2025", labels={"forecast_qty": "Forecasted Qty", "description": "Item"})
fig2.update_layout(yaxis={'categoryorder': 'total ascending'})

st.plotly_chart(fig2, use_container_width=True)



# === Preprocess Data ===
df["txndate"] = pd.to_datetime(df["txndate"])
df["description"] = df["description"].astype(str)
df = df.sort_values("txndate")

# === Select Description ===
desc_list = sorted(df["description"].dropna().unique())
selected_desc = st.selectbox("🔍 Search Item by Description", desc_list)

if selected_desc:
    desc_df = df[df["description"] == selected_desc]

    # Group by Date for Aging Values
    aging_grouped = desc_df.groupby("txndate")[["aging_60", "aging_90", "aging_180", "aging_180plus"]].sum().reset_index()

    # Plot Aging Data
    fig = px.line(
        aging_grouped,
        x="txndate",
        y=["aging_60", "aging_90", "aging_180", "aging_180plus"],
        labels={"value": "Stock Quantity", "txndate": "Date", "variable": "Aging Bucket"},
        title=f"📈 Aging Trend for '{selected_desc}' Over Time",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

# === Inventory Item Forecast ===
st.subheader("📦 Forecast Inventory Item Quantity")

desc_list = sorted(df["description"].dropna().unique())
selected_desc = st.selectbox("Select Item Description", desc_list)

if selected_desc:
    item_df = df[df["description"] == selected_desc].copy()
    item_df_grouped = item_df.groupby("txndate")["qty"].sum().reset_index()

    if len(item_df_grouped) > 3:
        item_df_grouped.set_index("txndate", inplace=True)
        model_item = ARIMA(item_df_grouped["qty"], order=(1, 1, 1))
        results_item = model_item.fit()

        forecast_item = results_item.get_forecast(steps=6)
        future_dates = pd.date_range(item_df_grouped.index[-1], periods=7, freq="M")[1:]
        forecast_item_df = pd.DataFrame({
            "txndate": future_dates,
            "forecast_qty": forecast_item.predicted_mean
        })

        # Merge actual and forecast data
        actual_forecast_df = item_df_grouped.reset_index()
        actual_forecast_df["forecast_qty"] = np.nan
        forecast_item_df["qty"] = np.nan
        combined_df = pd.concat([actual_forecast_df, forecast_item_df], ignore_index=True)

        # Plot
        fig_item = px.line(
            combined_df,
            x="txndate",
            y=["qty", "forecast_qty"],
            title=f"📈 Forecast for '{selected_desc}'",
            labels={"value": "Quantity", "txndate": "Date", "variable": "Type"}
        )
        st.plotly_chart(fig_item, use_container_width=True)
    else:
        st.warning("Not enough data to forecast this item.")


   # 1 Inventory Position Donut Chart

position_counts = {
    "Excess": (df["qty"] > 100).sum(),
    "Out of Stock": (df["qty"] == 0).sum(),
    "Below Panic Point": ((df["qty"] < 10) & (df["qty"] > 0)).sum(),
}
fig = px.pie(
    names=list(position_counts.keys()),
    values=list(position_counts.values()),
    title="Stock Status Distribution",
    color_discrete_sequence=["pink", "green", "purple"]
)
fig.update_layout(
    height=400,  # Resize the height of the chart
    width=400,   # Resize the width of the chart
    margin=dict(t=30, b=30, l=30, r=30)  # Reduce margins to fit content
)
col1.plotly_chart(fig, use_container_width=False)

# 2 Inventory at Lead Time Donut Chart

leadtime_counts = {
    "Safe": (df["lead_time_stock"] > 50).sum(),
    "At Risk": ((df["lead_time_stock"] > 10) & (df["lead_time_stock"] <= 50)).sum(),
    "Critical": (df["lead_time_stock"] <= 10).sum(),
}
fig = px.pie(
    names=list(leadtime_counts.keys()),
    values=list(leadtime_counts.values()),
    title="Expected Stock at Lead Time",
    color_discrete_sequence=["green", "yellow", "red"]
)
fig.update_layout(
    height=400,  # Resize the height of the chart
    width=400,   # Resize the width of the chart
    margin=dict(t=30, b=30, l=30, r=30)  # Reduce margins to fit content
)
col2.plotly_chart(fig, use_container_width=False)

# 3 Usage Pattern Types Donut Chart

df["usage_type"] = np.select(
    [
        df["qty"] == 0,
        df["qty"] < 10,
        (df["qty"] >= 10) & (df["qty"] < 50),
        df["qty"] >= 50,
    ],
    ["Dead", "Slow", "Sporadic", "Recurring"],
    default="New",
)
usage_counts = df["usage_type"].value_counts()
fig = px.pie(
    names=usage_counts.index,
    values=usage_counts.values,
    title="Consumption Patterns",
    color_discrete_sequence=["sea green", "orange", "purple", "blue"]
)
fig.update_layout(
    height=400,  # Resize the height of the chart
    width=400,   # Resize the width of the chart
    margin=dict(t=30, b=30, l=30, r=30)  # Reduce margins to fit content
)
col3.plotly_chart(fig, use_container_width=False)





    
# === Preprocess ===
df["txndate"] = pd.to_datetime(df["txndate"])
df["description"] = df["description"].astype(str)
df["month"] = df["txndate"].dt.strftime("%b")  # Month abbreviation (Jan, Feb)
df.sort_values("txndate", inplace=True)

# === Search by Description ===
st.subheader("🔍 Search by Item Description")
desc_list = sorted(df["description"].dropna().unique())
selected_desc = st.selectbox("Select Item", desc_list)

if selected_desc:
    item_df = df[df["description"] == selected_desc].copy()

    # === Monthly Aggregation ===
    item_df["month_only"] = item_df["txndate"].dt.to_period("M").dt.strftime("%b")
    monthly_df = item_df.groupby("month_only")["qty"].sum().reset_index()
    monthly_df = monthly_df.sort_values("month_only")  # Ensure months in order

    st.subheader(f"📅 Monthly Stock Trends for: {selected_desc}")
    fig = px.line(monthly_df, x="month_only", y="qty", markers=True,
                  title=f"📦 Stock Quantity Trend - {selected_desc}",
                  labels={"qty": "Quantity", "month_only": "Month"})
    st.plotly_chart(fig, use_container_width=True)

    # === Forecast Next Month Stock ===
    qty_series = item_df.groupby(item_df["txndate"].dt.to_period("M"))["qty"].sum()
    qty_series.index = qty_series.index.to_timestamp()
    if len(qty_series) >= 4:
        model = ARIMA(qty_series, order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)  # 1 month ahead

        next_month_qty = forecast.values[0]
        current_stock = item_df["qty"].sum()

        st.subheader("🔮 Prediction")
        st.markdown(f"**Estimated Required Qty for Next Month:** `{int(next_month_qty)}`")
        st.markdown(f"**Current Available Stock:** `{int(current_stock)}`")

        # Probability Logic
        if current_stock <= next_month_qty:
            st.error("⚠️ High probability this item will run out next month!")
        elif current_stock - next_month_qty < 10:
            st.warning("🟠 Low stock buffer, may run out soon.")
        else:
            st.success("✅ Stock level is sufficient for next month.")
    else:
        st.info("📉 Not enough data for forecasting. Need at least 4 months of data.")






  
