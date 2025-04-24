# === Import Libraries ===
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import json
import os

# === Streamlit Config ===
st.set_page_config(layout="wide")
st.title("📊 Supply Chain Forecasting Dashboard")



# === Sidebar Navigation Styling ===
st.markdown("""
    <style>
        .css-1d391kg { color: blue !important; }
    </style>
""", unsafe_allow_html=True)
st.sidebar.title("")

# === Load JSON Data ===
file_path = os.path.join(os.getcwd(), "data", "cust_stock.json")
if not os.path.exists(file_path):
    st.error("File not found: cust_stock.json")
    st.stop()

with open(file_path, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data["items"])

import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Sample data for illustration
# df = pd.DataFrame(data["items"])

import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Sample data for illustration
# df = pd.DataFrame(data["items"])

# === Preprocessing ===
df["txndate"] = pd.to_datetime(df["txndate"])
df.sort_values("txndate", inplace=True)
df["daily_demand"] = df["qty"] / 30  # Assumption: 30 days in a month
df["leadtime"] = 5  # Lead time is fixed at 5 days
df["lead_time_stock"] = df["qty"] - (df["daily_demand"] * df["leadtime"])
df["year_month"] = df["txndate"].dt.to_period("M").astype(str)
df["year"] = df["txndate"].dt.year
df["month"] = df["txndate"].dt.month
df["description"] = df["description"].astype(str)

# === Filter Data for 2024 and 2025 ===
df_2024 = df[(df["year"] == 2024)]
df_2025 = df[(df["year"] == 2025)]

# === Sidebar: Key Points ===
st.sidebar.subheader("Key Points")

# Low Stock Alert
with st.sidebar.expander("⚠️ Low Stock Alerts"):
    low_stock_items = df[df["lead_time_stock"] < 100]
    st.markdown(f"{len(low_stock_items)} items are low in stock.")
    for _, item in low_stock_items.iterrows():
        st.markdown(f"- **{item['description']}** (Qty: {item['qty']})")

# Customer Demand Prediction (for 2024)
with st.sidebar.expander("🔮 Customer Demand Prediction (2024)"):
    forecast_data = []
    for desc in df_2024["description"].unique():
        item_df = df_2024[df_2024["description"] == desc]
        monthly_series = item_df.resample("M", on="txndate")["qty"].sum().fillna(0)
        
        if len(monthly_series) >= 6:  # Need at least 6 months of data for ARIMA
            try:
                model = ARIMA(monthly_series, order=(1, 1, 1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=12)
                total_forecast = forecast.sum()  # Total demand forecast for the next 12 months (for 2025)
                forecast_data.append({"description": desc, "forecast_qty": total_forecast})
            except Exception as e:
                st.write(f"Error forecasting {desc}: {str(e)}")
                continue

    # Prepare forecast data for display
    if forecast_data:
        forecast_df = pd.DataFrame(forecast_data).sort_values("forecast_qty", ascending=False).head(10)
        st.write("Top 10 predicted items based on customer demand for the next 12 months (2025):")
        st.dataframe(forecast_df)

# Inventory Analysis (for both 2024 and 2025 data)
with st.sidebar.expander("📊 Inventory Analysis"):
    # Combine both 2024 and 2025 data for inventory analysis
    combined_df = pd.concat([df_2024, df_2025])
    top_items = combined_df.groupby("description")["qty"].sum().reset_index().sort_values("qty", ascending=False).head(5)
    st.write("Top 5 Items by Stock Quantity:")
    st.dataframe(top_items)

import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Assuming 'combined_df' is already available and contains the necessary data

# Filter data for 2024
data_2024 = combined_df[combined_df["year"] == 2024]

# === KPI Metrics for 2024 ===
total_stock = data_2024["qty"].sum()
total_value = data_2024["stockvalue"].sum() if "stockvalue" in data_2024.columns else 0
avg_stock_age = data_2024[["aging_60", "aging_90", "aging_180", "aging_180plus"]].sum(axis=1).mean() if all(col in data_2024.columns for col in ["aging_60", "aging_90", "aging_180", "aging_180plus"]) else 0

# Prepare data for training the model (we'll use the features for 2024 to forecast for 2025)
X_2024 = data_2024[["aging_60", "aging_90", "aging_180", "aging_180plus"]]
y_2024 = data_2024["qty"]

# Handle missing values (if any) in the features
X_2024 = X_2024.fillna(0)  # Impute missing values if necessary

# Train the model (Linear Regression as an example)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_2024, y_2024)

# Use the trained model to predict 2025 quantities based on 2024 data
predictions_2025 = model.predict(X_2024)

# Add predictions to the 2024 data (forecast for 2025)
data_2024["predicted_qty_2025"] = predictions_2025

# === Display Predictions ===
import streamlit as st

# Display KPIs for 2024
col1, col2, col3 = st.columns(3)

# === KPI Metrics for 2024 ===
total_stock = data_2024["qty"].sum()
total_value = data_2024["stockvalue"].sum() if "stockvalue" in data_2024.columns else 0
avg_stock_age = data_2024[["aging_60", "aging_90", "aging_180", "aging_180plus"]].sum(axis=1).mean() if all(col in data_2024.columns for col in ["aging_60", "aging_90", "aging_180", "aging_180plus"]) else 0

# === Forecast for 2025 ===
# Use the model to predict quantities for 2025
predictions_2025 = model.predict(X_2024)

# Add predictions to the 2024 data (forecast for 2025)
data_2024["predicted_qty_2025"] = predictions_2025

# === Display KPIs and Predictions ===
import streamlit as st

col1, col2, col3, col4 = st.columns(4)

# Total Stock Quantity (2024)
col1.markdown(f"""
<div style="background-color:#f4f4f4; padding:20px; border-radius:10px; box-shadow:0 4px 8px rgba(0, 0, 0, 0.1); text-align:center;">
<h3>📦 Total Stock Quantity (2024)</h3>
<p style="font-size: 24px; font-weight: bold;">{total_stock:,}</p>
</div>""", unsafe_allow_html=True)

# Total Inventory Value (2024)
col2.markdown(f"""
<div style="background-color:#f4f4f4; padding:20px; border-radius:10px; box-shadow:0 4px 8px rgba(0, 0, 0, 0.1); text-align:center;">
<h3>💰 Total Inventory Value (2024)</h3>
<p style="font-size: 24px; font-weight: bold;">{total_value:,.2f}</p>
</div>""", unsafe_allow_html=True)

# Average Stock Age (2024)
col3.markdown(f"""
<div style="background-color:#f4f4f4; padding:20px; border-radius:10px; box-shadow:0 4px 8px rgba(0, 0, 0, 0.1); text-align:center;">
<h3>⏳ Avg Stock Age (2024)</h3>
<p style="font-size: 24px; font-weight: bold;">{avg_stock_age:.2f} days</p>
</div>""", unsafe_allow_html=True)

# Predicted Stock Quantity for 2025
predicted_total_2025 = data_2024["predicted_qty_2025"].sum()  # Sum of the predicted quantities for 2025
col4.markdown(f"""
<div style="background-color:#f4f4f4; padding:20px; border-radius:10px; box-shadow:0 4px 8px rgba(0, 0, 0, 0.1); text-align:center;">
<h3>🔮 Predicted Stock Quantity (2025)</h3>
<p style="font-size: 24px; font-weight: bold;">{predicted_total_2025:,}</p>
</div>""", unsafe_allow_html=True)









# Shared pie chart configuration
def make_donut_chart(names, values, colors):
    fig = px.pie(
        names=names,
        values=values,
        hole=0.4,
        color_discrete_sequence=colors
    )
    fig.update_layout(
        height=400,
        width=400,
        margin=dict(t=30, b=30, l=30, r=30),
        showlegend=True
    )
    return fig

col1, col2, col3 = st.columns(3)

# === 1. Inventory Position Donut Chart ===
with col1:
    st.markdown(
        """
        <div style="margin-top:20px; padding:8px; background-color:#f4f4f4; border-radius:6px; font-weight:bold; text-align:center;">
           <h3 style="color: black;">📊 Stock Status Distribution (2024)</h3> 
        </div>
        """,
        unsafe_allow_html=True
    )

    df_2024 = df[df["year"] == 2024]

    position_counts = {
        "Excess": (df_2024["qty"] > 100).sum(),
        "Out of Stock": (df_2024["qty"] == 0).sum(),
        "Below Panic Point": ((df_2024["qty"] < 10) & (df_2024["qty"] > 0)).sum(),
    }

    fig1 = make_donut_chart(
        names=list(position_counts.keys()),
        values=list(position_counts.values()),
        colors=["pink", "green", "purple"]
    )
    st.plotly_chart(fig1, use_container_width=False)

# === 2. Usage Pattern Types Donut Chart ===
with col2:
    st.markdown(
        """
        <div style="margin-top:20px; padding:8px; background-color:#f4f4f4; border-radius:6px; font-weight:bold; text-align:center;">
           <h3 style="color: black;">📦 Consumption Patterns (2024)</h3> 
        </div>
        """,
        unsafe_allow_html=True
    )

    df_2024["usage_type"] = np.select(
        [
            df_2024["qty"] == 0,
            df_2024["qty"] < 10,
            (df_2024["qty"] >= 10) & (df_2024["qty"] < 50),
            df_2024["qty"] >= 50,
        ],
        ["Dead", "Slow", "Sporadic", "Recurring"],
        default="New",
    )
    usage_counts_2024 = df_2024["usage_type"].value_counts()

    fig2 = make_donut_chart(
        names=usage_counts_2024.index,
        values=usage_counts_2024.values,
        colors=["sea green", "orange", "purple", "blue"]
    )
    st.plotly_chart(fig2, use_container_width=False)

with col3:
    st.markdown(
        """
        <div style="margin-top:20px; padding:10px; background-color:#f4f4f4; border-radius:8px; font-weight:bold; text-align:center;">
           <h3 style="color: black;">📦 Top 10 Consumption Patterns (2025 Prediction)</h3> 
        </div>
        """,
        unsafe_allow_html=True
    )

# Add prediction for 2025 using ARIMA model
forecast_data_2025 = []
for desc in df["description"].unique():  # Use entire dataset, not just 2024
    item_df = df[df["description"] == desc]
    monthly_series = item_df.resample("M", on="txndate")["qty"].sum().fillna(0)
    if len(monthly_series) >= 6:
        try:
            model = ARIMA(monthly_series, order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=12)
            total_forecast_2025 = forecast.sum()
            forecast_data_2025.append({"description": desc, "forecast_qty_2025": total_forecast_2025})
        except:
            continue

# Forecast data for 2025 consumption patterns
forecast_df_2025 = pd.DataFrame(forecast_data_2025).sort_values("forecast_qty_2025", ascending=False).head(10)

# Plotting the donut chart for predicted 2025 consumption patterns
fig = px.pie(

    values=forecast_df_2025['forecast_qty_2025'],
    color_discrete_sequence=["sea green", "orange", "purple", "blue", "red", "pink"]
)

# Show the pie chart for predicted 2025 usage patterns
col3.plotly_chart(fig, use_container_width=False)






# === Top Trending Items ===
st.subheader("🔥 Top Trending Items (Bar Chart)")

count_option = st.selectbox("Select number of top items to show:", [10, 50, 100], index=0)

top_items_df = df.groupby("description")["qty"].sum().reset_index().sort_values("qty", ascending=False).head(count_option)

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

st.plotly_chart(fig1, use_container_width=True)



# === Prediction for 2025 Inventory Demand ===
st.subheader("🔮 2025 Forecast: Items Likely to Be Bought Again")

# Get top 50 items from 2024 and 2025 to apply prediction
top_items = df.groupby("description")["qty"].sum().sort_values(ascending=False).head(50).index

forecast_data = []
for desc in top_items:
    item_df = df[df["description"] == desc]
    monthly_series = item_df.resample("M", on="txndate")["qty"].sum()
    monthly_series = monthly_series.fillna(0)
    
    # Ensure we have at least 6 months of data for prediction
    if len(monthly_series) >= 6:  # Minimum months of data
        try:
            model = ARIMA(monthly_series, order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=12)  # Predict 12 months of 2025
            
            # Sum the forecast to get the total quantity for 2025
            total_forecast_2025 = forecast.sum()
            
            # Append the prediction to the forecast data list
            forecast_data.append({"description": desc, "forecast_qty": total_forecast_2025})
        except Exception as e:
            # Continue if the model fails for a particular item
            print(f"Error forecasting item {desc}: {e}")
            continue

# Create DataFrame for forecasted data
forecast_df = pd.DataFrame(forecast_data)
forecast_df = forecast_df.sort_values("forecast_qty", ascending=False).head(20)

# Plotting the forecasted items chart
fig2 = px.bar(forecast_df, x="forecast_qty", y="description", color="forecast_qty", orientation="h",
              title="📦 Predicted Top Items for 2025", labels={"forecast_qty": "Forecasted Qty", "description": "Item"})
fig2.update_layout(yaxis={'categoryorder': 'total ascending'})

# Plotting the second chart (forecast for 2025)
st.plotly_chart(fig2, use_container_width=True)









    # Rank majors within each month to only show top N in animation for 2024 and 2025
top_n = 10

# Filter the data for 2024 and 2025
df_filtered = df[df["year"].isin([2024, 2025])]

# Aggregate stockvalue per month and major category
df_filtered["year_month"] = df_filtered["txndate"].dt.to_period("M")
monthly_major_stock = (
    df_filtered.groupby(["year_month", "major"])["stockvalue"]
    .sum()
    .reset_index()
)

# Rank the major categories within each month
monthly_major_stock["rank"] = monthly_major_stock.groupby("year_month")["stockvalue"].rank("dense", ascending=False)

# Filter to include only the top N major categories within each month
monthly_major_stock = monthly_major_stock[monthly_major_stock["rank"] <= top_n]

# Plotly Animated Bar Chart for Top N Major Categories by Stock Value Over Time
fig = px.bar(
    monthly_major_stock,
    x="stockvalue",
    y="major",
    color="major",
    orientation="h",
    animation_frame="year_month",
    range_x=[0, monthly_major_stock["stockvalue"].max() * 1.1],
    title=f"Top {top_n} Major Categories by Stock Value Over Time",
    labels={"stockvalue": "Stock Value", "major": "Category"},
    height=600
)

# Customize the layout of the chart
fig.update_layout(
    yaxis={'categoryorder': 'total ascending'},
    showlegend=False,
    xaxis_title="Stock Value",
    yaxis_title="Major Category"
)

# Display the chart
st.plotly_chart(fig, use_container_width=True)






# === Preprocess Data for 2024 and 2025 ===
df["txndate"] = pd.to_datetime(df["txndate"])
df["description"] = df["description"].astype(str)
df["year"] = df["txndate"].dt.year
df = df[df["year"].isin([2024, 2025])]  # Filter data to include only 2024 and 2025
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

@st.cache_data(show_spinner=False)
def fit_arima_model(item_df_grouped):
    model = ARIMA(item_df_grouped["qty"], order=(1, 1, 1))
    return model.fit()



# === Preprocess for 2024 and 2025 ===
df["txndate"] = pd.to_datetime(df["txndate"])
df["description"] = df["description"].astype(str)
df["year"] = df["txndate"].dt.year  # Extract year from transaction date
df["month"] = df["txndate"].dt.strftime("%b")  # Month abbreviation (Jan, Feb)
df = df[df["year"].isin([2024, 2025])]  # Filter data to include only 2024 and 2025
df.sort_values("txndate", inplace=True)

# === Search by Description ===
st.subheader("🔍 Search by Item Description")
desc_list = sorted(df["description"].dropna().unique())
selected_desc = st.selectbox("Select Item", desc_list)

if selected_desc:
    item_df = df[df["description"] == selected_desc].copy()

    # === Monthly Aggregation for 2024 and 2025 ===
    item_df["month_only"] = item_df["txndate"].dt.to_period("M").dt.strftime("%b-%Y")  # Include year for proper sorting
    monthly_df = item_df.groupby("month_only")["qty"].sum().reset_index()
    monthly_df = monthly_df.sort_values("month_only")  # Ensure months in order

    st.subheader(f"📅 Monthly Stock Trends for: {selected_desc}")
    fig = px.line(monthly_df, x="month_only", y="qty", markers=True,
                  title=f"📦 Stock Quantity Trend - {selected_desc}",
                  labels={"qty": "Quantity", "month_only": "Month-Year"})
    st.plotly_chart(fig, use_container_width=True)

# === Forecast Next Month Stock for 2024 and 2025 ===
df["txndate"] = pd.to_datetime(df["txndate"])
df["description"] = df["description"].astype(str)
df["year"] = df["txndate"].dt.year  # Extract year from transaction date
df = df[df["year"].isin([2024, 2025])]  # Filter data to include only 2024 and 2025

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
# === 1. Usage Pattern Types Donut Chart with 2025 Prediction ===


