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
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor



# === Use Only Data from 2024 ===
df_2024 = df[df["year"] == 2024]

# === Sidebar: Key Points ===
st.sidebar.subheader("Key Points")

# === Helper Function: Create Lag Features ===
def create_lag_features(df, lags=[1, 2, 3]):
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df["lead_time_stock"].shift(lag)
    return df.dropna()

# === Forecasting 2025 (12 months) ===
forecasted_items = []

# Loop through each unique item in 2024 data to forecast stock levels for 2025
for item in df_2024["description"].unique():
    item_df = df_2024[df_2024["description"] == item].sort_values("txndate")
    item_df = create_lag_features(item_df)

    if len(item_df) < 3:
        continue  # Skip items with insufficient data

    X = item_df[[f"lag_{i}" for i in [1, 2, 3]]]
    y = item_df["lead_time_stock"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Use last known lags from 2024 to forecast future stock in 2025
    last_known_lags = list(X.iloc[-1])

    # Forecast the next 12 months (simulating monthly predictions for 2025)
    forecast_horizon = 12
    for month in range(forecast_horizon):
        prediction = model.predict([last_known_lags])[0]

        if prediction < 100:  # If stock is forecasted to be low
            forecasted_items.append({
                "description": item,
                "predicted_stock": prediction,
                "forecast_month": 2025 + month // 12  # Forecast Year (2025)
            })

        # Update lags for the next month prediction
        last_known_lags = [prediction] + last_known_lags[:2]  # Shift lags




# Customer Demand Prediction (for 2025)
with st.sidebar.expander("🔮 Customer Demand Prediction (2025)"):
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

col1, col2, col3 = st.columns([1, 1, 1])

card_style = """
<div style="background-color:#f4f4f4; width: 100%; padding:10px; 
            border-radius:12px; box-shadow:0 4px 8px rgba(0, 0, 0, 0.1); 
            text-align:center;">
    <h3 style="margin-bottom: 5px;">{icon} {title}</h3>
    <p style="font-size: 26px; font-weight: bold; margin: 0;">{value}</p>
</div>
"""

col1.markdown(card_style.format(
    icon="📦", title="Total Stock Quantity (2024)", value=f"{total_stock:,.2f}"
), unsafe_allow_html=True)

col2.markdown(card_style.format(
    icon="💰", title="Total Inventory Value (2024)", value=f"{total_value:,.2f}"
), unsafe_allow_html=True)

col3.markdown(card_style.format(
    icon="⏳", title="Avg Stock Age (2024)", value=f"{avg_stock_age:.2f} days"
), unsafe_allow_html=True)







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

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Define the donut chart creation function (assuming this is already defined)
def make_donut_chart(names, values, colors):
    fig = go.Figure(data=[go.Pie(
        labels=names, 
        values=values, 
        hole=0.4, 
        marker=dict(colors=colors)
    )])
    return fig

# Assuming df is your DataFrame
# Create three columns
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
        "Excess - (More than 100) ": (df_2024["qty"] > 100).sum(),
        "Out of Stock - (Qty = 0)": (df_2024["qty"] == 0).sum(),
        "Below Panic Point - (Qty < 10 and > 0)": ((df_2024["qty"] < 10) & (df_2024["qty"] > 0)).sum(),
    }

    fig1 = make_donut_chart(
        names=list(position_counts.keys()),
        values=list(position_counts.values()),
        colors=["pink", "green", "purple"]
    )
    st.plotly_chart(fig1, use_container_width=True)

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
        ["Dead - (Qty = 0)", "Slow - (Qty < 10)", "Sporadic - (Qty >=10 )", "Recurring - (Qty >= 50)"],
        default="New",
    )
    usage_counts_2024 = df_2024["usage_type"].value_counts()

    fig2 = make_donut_chart(
        names=usage_counts_2024.index,
        values=usage_counts_2024.values,
        colors=["sea green", "orange", "purple", "blue"]
    )
    st.plotly_chart(fig2, use_container_width=True)

# === 3. Top 5 Consumption Patterns Prediction (2025) ===
with col3:
    st.markdown(
        """
        <div style="margin-top:20px; padding:10px; background-color:#f4f4f4; border-radius:8px; font-weight:bold; text-align:center;">
           <h3 style="color: black;">📦 Top 5 Consumption Patterns - 2025 Prediction</h3> 
        </div>
        """,
        unsafe_allow_html=True
    )

    # Forecast data for 2025 using ARIMA model
    forecast_data_2025 = []
    for desc in df["description"].unique():
        item_df = df[df["description"] == desc]
        monthly_series = item_df.resample("M", on="txndate")["qty"].sum().fillna(0)
        if len(monthly_series) >= 6:
            try:
                model = ARIMA(monthly_series, order=(1, 1, 1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=12)
                total_forecast_2025 = forecast.sum()
                forecast_data_2025.append({
                    "description": desc,
                    "forecast_qty_2025": total_forecast_2025
                })
            except:
                continue

    # Create DataFrame and select Top 5
    forecast_df_2025 = pd.DataFrame(forecast_data_2025).sort_values("forecast_qty_2025", ascending=False).head(5)

    # Create donut chart with labels inside
    fig3 = px.pie(
        names=forecast_df_2025['description'],
        values=forecast_df_2025['forecast_qty_2025'],
        hole=0.4,
        color_discrete_sequence=["sea green", "orange", "purple", "blue", "red"]
    )

    # Customize label display
    fig3.update_traces(
        textinfo="label+percent",
        insidetextorientation="radial"
    )

    # Layout tweaks
    fig3.update_layout(
        height=400,
        width=400,
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=False  # Set True if you want legend outside
    )

    # Show chart
    st.plotly_chart(fig3, use_container_width=True)










# Your existing code
st.subheader("🔥 Top Trending Items ")

count_option = st.selectbox("Select number of top items to show:", [10, 50, 100], index=0)

top_items_df = df.groupby("description")["qty"].sum().reset_index().sort_values("qty", ascending=False).head(count_option)

# Create a Violin plot. Instead of `color_continuous_scale`, use `color` for a categorical variable.
fig8 = px.violin(top_items_df, y="qty", box=True, title=f"Top {count_option} Trending Items (Violin Plot)",
                 labels={"qty": "Total Qty", "description": "Item"}, color="description")

st.plotly_chart(fig8, use_container_width=True)




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






import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
import streamlit as st

# === Preprocess Data for 2024 (No 2025 data) ===
df["txndate"] = pd.to_datetime(df["txndate"])
df["description"] = df["description"].astype(str)
df["year"] = df["txndate"].dt.year
df = df[df["year"] == 2024]  # Filter data to only include 2024
df = df.sort_values("txndate")

# === Select Description ===
desc_list = sorted(df["description"].dropna().unique())
selected_desc = st.selectbox("🔍 Search Item by Description", desc_list)

if selected_desc:
    desc_df = df[df["description"] == selected_desc]

    # Group by Date for Aging Values
    aging_grouped = desc_df.groupby("txndate")[["aging_60", "aging_90", "aging_180", "aging_180plus"]].sum().reset_index()

    # Forecasting using ARIMA (Example: Forecasting aging_60)
    aging_60_df = aging_grouped[["txndate", "aging_60"]]
    aging_60_df.set_index("txndate", inplace=True)

    # Fit ARIMA model on the 2024 data
    model = ARIMA(aging_60_df, order=(5,1,0))  # ARIMA(p,d,q)
    model_fit = model.fit()

    # Forecast for the next 12 months (for 2025)
    forecast_steps = 12  # Forecast for 12 months
    forecast = model_fit.forecast(steps=forecast_steps)

    # Create date range for 2025 (next year)
    forecast_dates = pd.date_range(start=aging_60_df.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='M')

    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        "txndate": forecast_dates,
        "forecast_aging_60": forecast
    })

    # Combine the historical and forecasted data
    combined_df = pd.concat([aging_60_df, forecast_df], axis=0)

    # Plot the Aging Data with Forecast for 2025
    fig = go.Figure()

    # Plot the actual data from 2024
    fig.add_trace(go.Scatter(x=aging_60_df.index, y=aging_60_df["aging_60"], mode='lines', name="Actual Aging 60"))

    # Plot the forecasted data for 2025
    fig.add_trace(go.Scatter(x=forecast_df["txndate"], y=forecast_df["forecast_aging_60"], mode='lines', name="Forecast Aging 60", line=dict(dash='dash')))

    fig.update_layout(
        title=f"📈 Aging Trend for '{selected_desc}' (Aging 60) with Forecast for 2025",
        xaxis_title="Date",
        yaxis_title="Stock Quantity",
        template="plotly_dark",
        showlegend=True
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

import pandas as pd
import plotly.express as px
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# === Search by Description ===
st.subheader("🔍 Search by Item Description")
desc_list = sorted(df["description"].dropna().unique())
selected_desc = st.selectbox("Select Item", desc_list)

if selected_desc:
    item_df = df[df["description"] == selected_desc].copy()
    item_df["txndate"] = pd.to_datetime(item_df["txndate"])

    # === Monthly Aggregation (including past years for model training) ===
    item_df["month"] = item_df["txndate"].dt.to_period("M").dt.to_timestamp()
    monthly_df = item_df.groupby("month")["qty"].sum().reset_index()
    monthly_df = monthly_df.sort_values("month")

    # === Forecasting using ARIMA ===
    monthly_df.set_index("month", inplace=True)
    model = ARIMA(monthly_df["qty"], order=(1, 1, 1))
    model_fit = model.fit()

    forecast_steps = 12  # Forecast for 12 months (2025)
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_df = forecast.summary_frame()
    forecast_df["month"] = pd.date_range(start=monthly_df.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
    forecast_df.rename(columns={"mean": "Forecast"}, inplace=True)

    # === Combine historical + forecast ===
    monthly_df_reset = monthly_df.reset_index()
    all_data = pd.concat([
        monthly_df_reset.rename(columns={"qty": "Quantity"}),
        forecast_df[["month", "Forecast"]].rename(columns={"month": "month"})
    ], ignore_index=True)

    st.subheader(f"📅 Forecasted Stock Trends for: {selected_desc}")
    fig = px.line(all_data, x="month", y=["Quantity", "Forecast"], markers=True,
                  title=f"📦 Stock Forecast (ARIMA) - {selected_desc}",
                  labels={"value": "Quantity", "month": "Month-Year", "variable": "Type"})
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


