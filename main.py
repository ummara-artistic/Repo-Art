# === Import Libraries ===
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import warnings
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import json

# Train the model (Linear Regression as an example)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# === Streamlit Config ===
st.set_page_config(layout="wide")
st.title("📊 Supply Chain Forecasting Dashboard")
# Suppress warnings related to ARIMA model fitting
warnings.filterwarnings("ignore")

st.markdown("""
    <style>
        /* Target the sidebar expander's button and its text */
        .css-1d391kg .st-expander {
            background-color: #0000FF !important;  /* Blue background for expander */
            color: white !important;  /* White text color for the expander */
        }

        /* Change the color when the expander is active (clicked/expanded) */
        .css-1d391kg .st-expander[aria-expanded="true"] {
            background-color: #0000FF !important;  /* Blue background when expanded */
            color: white !important;  /* White text when expanded */
        }

        /* Optional: Change the text color inside the expander to ensure it's visible */
        .css-1d391kg .st-expander .css-14xtw13 {
            color: white !important; /* Ensure text remains white in the expander */
        }
        
        /* Optional: Change the expander icon color */
        .css-1d391kg .st-expander .st-expanderIcon {
            color: white !important;  /* White color for the icon */
        }
    </style>
""", unsafe_allow_html=True)











# === Load JSON Data ===
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



# Assuming 'df' is already preprocessed
with st.sidebar.expander("📊 Inventory Analysis"):
    # Combine both 2024 and 2025 data
    combined_df = pd.concat([df_2024, df_2025])

    # Convert txndate to datetime if not already
    combined_df["txndate"] = pd.to_datetime(combined_df["txndate"])

    # Group by 'major', 'description', and 'fabtype' to calculate sum of stockvalue and qty, aging columns
    inventory_summary = (
        combined_df.groupby(["major", "description", "fabtype"])[["stockvalue", "qty", "aging_60", "aging_90", "aging_180", "aging_180plus"]]
        .sum()
        .reset_index()
    )

    # Forecast stock values for 2025 based on 'fabtype'
    forecast_data = []

    for fabtype in inventory_summary["fabtype"].unique():
        fabtype_df = inventory_summary[inventory_summary["fabtype"] == fabtype]

        for _, row in fabtype_df.iterrows():
            # For each item, create a time series forecast
            item_df = combined_df[combined_df["description"] == row["description"]]
            item_monthly_series = item_df.resample("M", on="txndate")["stockvalue"].sum().fillna(0)

            if len(item_monthly_series) >= 6:  # At least 6 months of data for ARIMA
                try:
                    model = ARIMA(item_monthly_series, order=(1, 1, 1))
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=12)  # Forecast next 12 months
                    total_forecast = forecast.sum()

                    forecast_data.append({
                        "fabtype": fabtype,
                        "description": row["description"],
                        "forecast_stockvalue_2025": total_forecast
                    })
                except Exception as e:
                    st.write(f"Error forecasting {row['description']} with fabtype {fabtype}: {str(e)}")
                    continue

    # Prepare and display forecast data
    if forecast_data:
        forecast_df = pd.DataFrame(forecast_data)
        

        # === Animated Plotly Bar Chart ===
        forecast_df["rank"] = forecast_df.groupby("fabtype")["forecast_stockvalue_2025"].rank(method="first", ascending=False)
        animated_df = forecast_df[forecast_df["rank"] <= 5]  # Top 5 per fabtype

        fig_forecast_anim = px.bar(
            animated_df,
            x="forecast_stockvalue_2025",
            y="description",
            color="description",
            animation_frame="fabtype",
            orientation="h",
            range_x=[0, animated_df["forecast_stockvalue_2025"].max() * 1.4],
            title=" Top 50 Items by Fab Type for 2025",
            labels={"forecast_stockvalue_2025": "Forecasted Stock Value", "description": "Item"},
            height=600
        )
        st.plotly_chart(fig_forecast_anim, use_container_width=True)

    # Sort by stockvalue and select the top 5 items
    top_inventory = inventory_summary.sort_values("stockvalue", ascending=False).head(50)

    # Remove the 'inventory_item_id' column if it exists in the summary
    if 'inventory_item_id' in top_inventory.columns:
        top_inventory = top_inventory.drop(columns=['inventory_item_id'])



# Filter data for 2024
data_2024 = combined_df[combined_df["year"] == 2024]

# === KPI Metrics for 2024 ===
#.sum(axis=1) means value added by row wise
total_stock = data_2024["qty"].sum()
total_value = data_2024["stockvalue"].sum() if "stockvalue" in data_2024.columns else 0 
avg_stock_age = data_2024[["aging_60", "aging_90", "aging_180", "aging_180plus"]].sum(axis=1).mean() if all(col in data_2024.columns for col in ["aging_60", "aging_90", "aging_180", "aging_180plus"]) else 0

# Prepare data for training the model (we'll use the features for 2024 to forecast for 2025)
X_2024 = data_2024[["aging_60", "aging_90", "aging_180", "aging_180plus"]]
y_2024 = data_2024["qty"]

# Handle missing values (if any) in the features
X_2024 = X_2024.fillna(0)  # Impute missing values if necessary


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
# === Forecast for 2025 ===
# Calculate total stock, value, and average stock age for the 2024 data
total_stock = data_2024["qty"].sum()
total_value = data_2024["stockvalue"].sum() if "stockvalue" in data_2024.columns else 0
avg_stock_age = data_2024[["aging_60", "aging_90", "aging_180", "aging_180plus"]].sum(axis=1).mean() if all(col in data_2024.columns for col in ["aging_60", "aging_90", "aging_180", "aging_180plus"]) else 0

# Use the model to predict quantities for 2025
predictions_2025 = model.predict(X_2024)

# Add predictions to the 2024 data (forecast for 2025)
data_2024["predicted_qty_2025"] = predictions_2025

# === Display KPIs and Predictions ===


col1, col2, col3 = st.columns([1, 1, 1])

card_template = """
<div style="
    background-color:#f9f9f9; 
    padding: 20px; 
    border-radius: 10px; 
    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    text-align:center;
    font-family:sans-serif;
">
    <h4 style="margin-bottom: 10px;">{icon} {title}</h4>
    <p style="font-size: 28px; font-weight: bold; color: #333;">{value}</p>
</div>
"""

# Display each KPI card
col1.markdown(card_template.format(
    icon="📦",
    title="Predicted Stock Qty (2025)",
    value=f"{data_2024['predicted_qty_2025'].sum():,.0f}"
), unsafe_allow_html=True)

col2.markdown(card_template.format(
    icon="💰",
    title="Predicted Inventory Value (2025)",
    value=f"${total_value:,.2f}"  # Assuming it remains the same
), unsafe_allow_html=True)

col3.markdown(card_template.format(
    icon="⏳",
    title="Avg Stock Age",
    value=f"{avg_stock_age:.1f} days"
), unsafe_allow_html=True)



# Calculate the total monthly demand for 2024
df_2024_monthly = df_2024.groupby("year_month").agg(
    total_demand=("daily_demand", "sum")
).reset_index()

# Convert 'year_month' to datetime for feature creation
df_2024_monthly['year_month'] = pd.to_datetime(df_2024_monthly['year_month'])

# Extract additional time-based features (Month and Day of Year)
df_2024_monthly['month'] = df_2024_monthly['year_month'].dt.month
df_2024_monthly['day_of_year'] = df_2024_monthly['year_month'].dt.dayofyear

# Features and target variable
X = df_2024_monthly[['month', 'day_of_year']]
y = df_2024_monthly['total_demand']

# Split the data into train and test sets (optional, for validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Random Forest Regressor ===
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# === Forecast for 2025 ===
# Generate features for the 12 months of 2025 (we'll use the same features as 2024)
forecast_2025 = pd.DataFrame({
    'month': list(range(1, 13)),
    # Ensure day_of_year is calculated correctly for 2025, mapping each month to a unique day of year value
    'day_of_year': [pd.to_datetime(f'2025-{month:02d}-01').dayofyear for month in range(1, 13)]
})

# Predict demand for 2025
forecast_2025['predicted_demand'] = rf_model.predict(forecast_2025[['month', 'day_of_year']])

# === Streamlit Visualization ===
# Create bar traces for 2024 demand
trace_2024 = go.Bar(
    x=df_2024_monthly['year_month'].dt.strftime('%b %Y'),
    y=df_2024_monthly["total_demand"],
    name='2024 Demand',
    opacity=0.7
)

# Create bar traces for forecasted 2025 demand
trace_2025 = go.Bar(
    x=pd.date_range(start='2025-01-01', periods=12, freq='M').strftime('%b %Y'),
    y=forecast_2025['predicted_demand'],
    name='Forecasted 2025 Demand',
    opacity=0.7,
    marker=dict(color='orange')
)
layout1 = go.Layout(
    title="Demand Forecast: 2024 vs 2025",
    
    xaxis=dict(title="Month", tickangle=45),
    yaxis=dict(title="Demand"),
    barmode='group'
)

# Set up the layout for the graph
layout = go.Layout(
    title="Price Sensitivity Forecasting: Actual vs Forecasted Demand (2024 vs 2025)",
    xaxis=dict(title="Month", tickangle=45),
    yaxis=dict(title="Demand"),
    barmode='group',
    showlegend=True
)


# Create the figure and plot it
fig = go.Figure(data=[trace_2024, trace_2025], layout=layout)





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



# Define the donut chart creation function (assuming this is already defined)
def make_donut_chart(names, values, colors):
    fig = go.Figure(data=[go.Pie(
        labels=names, 
        values=values, 
        hole=0.4, 
        marker=dict(colors=colors)
    )])
    return fig

fig_demand = go.Figure(data=[trace_2024, trace_2025], layout=layout1)

# Assuming df is your DataFrame
# Create three columns
col1, col2, col3 = st.columns(3)

# === 1. Inventory Position Donut Chart ===
# Layout Row 1: Demand Forecast & Inventory Position
row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    st.plotly_chart(fig_demand, use_container_width=True)

with row1_col2:
    st.markdown("<h4 style='text-align:center;'>📊 Stock Status Distribution (2024)</h4>", unsafe_allow_html=True)
    position_counts = {
        "Excess - (More than 100) ": (df_2024["qty"] > 100).sum(),
        "Out of Stock - (Qty = 0)": (df_2024["qty"] == 0).sum(),
        "Below Panic Point - (Qty < 10 and > 0)": ((df_2024["qty"] < 10) & (df_2024["qty"] > 0)).sum(),
    }
    fig_stock = make_donut_chart(
        names=list(position_counts.keys()),
        values=list(position_counts.values()),
        colors=["pink", "green", "purple"]
    )
    st.plotly_chart(fig_stock, use_container_width=True)




    
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import streamlit as st

# Layout Row 2: Usage Patterns & Top 5 Forecast (2025)
row2_col1, row2_col2 = st.columns(2)









import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA

st.markdown("<h4 style='text-align:center;'>📦 Top 100 Monthly Consumption Predictions (2025)</h4>", unsafe_allow_html=True)

# Step 1: Filter for 2024 data
df_2024 = df[(df["txndate"] >= "2024-01-01") & (df["txndate"] <= "2024-12-31")]

# Step 2: Initialize forecast results
forecast_data_2025 = []

# Step 3: Loop through each product and forecast 12 months
for desc in df_2024["description"].unique():
    item_df = df_2024[df_2024["description"] == desc]
    monthly_series = item_df.resample("M", on="txndate")["qty"].sum().fillna(0)

    if len(monthly_series) >= 6:
        try:
            model = ARIMA(monthly_series, order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=12)
            forecast_months = pd.date_range(start="2025-01-01", periods=12, freq="M")

            for date, qty in zip(forecast_months, forecast):
                forecast_data_2025.append({
                    "description": desc,
                    "month": date.strftime("%Y-%m"),
                    "forecast_qty": qty
                })
        except:
            continue

# Step 4: Create DataFrame of all forecasts
forecast_df_2025 = pd.DataFrame(forecast_data_2025)

# Step 5: Select top 100 items by total forecasted quantity for 2025
top_100 = (
    forecast_df_2025.groupby("description")["forecast_qty"]
    .sum()
    .nlargest(100)
    .index
)

top_100_df = forecast_df_2025[forecast_df_2025["description"].isin(top_100)]

# Step 6: Plot animated bar chart
fig_animated = px.bar(
    top_100_df,
    x="description",
    y="forecast_qty",
    animation_frame="month",
    color="forecast_qty",
    color_continuous_scale="Cividis",
    title="Top 100 Monthly Forecasted Consumptions for 2025"
)

fig_animated.update_layout(
    xaxis={'categoryorder': 'total descending'},
    xaxis_title="Item Description",
    yaxis_title="Forecasted Quantity",
    template="plotly_white",
    height=600
)

fig_animated.update_traces(marker_line_width=0.5, marker_line_color="gray")

# Step 7: Show chart
st.plotly_chart(fig_animated, use_container_width=True)








# === 1. Usage Pattern Types Donut Chart with 2025 Prediction ===
top_n = 10

# Step 1: Filter only 2024 data
df_2024 = df[df["year"] == 2024].copy()
df_2024["year_month"] = df_2024["txndate"].dt.to_period("M")

# Step 2: Aggregate monthly average stock value per major
monthly_avg = (
    df_2024.groupby(["major", df_2024["txndate"].dt.month])["stockvalue"]
    .mean()
    .reset_index()
    .rename(columns={"txndate": "month", "stockvalue": "avg_stockvalue"})
)

# Step 3: Build predicted 2025 data using 2024 monthly average
predicted_2025 = []
for month in range(1, 13):
    month_str = f"2025-{month:02d}"
    temp_df = monthly_avg[monthly_avg["month"] == month].copy()
    temp_df["year_month"] = pd.Period(month_str)
    temp_df["stockvalue"] = temp_df["avg_stockvalue"]
    predicted_2025.append(temp_df[["major", "year_month", "stockvalue"]])

predicted_2025_df = pd.concat(predicted_2025, ignore_index=True)

# Step 4: Rank majors within each 2025 month
predicted_2025_df["rank"] = predicted_2025_df.groupby("year_month")["stockvalue"].rank("dense", ascending=False)

# Step 5: Filter top N per month
top_predicted = predicted_2025_df[predicted_2025_df["rank"] <= top_n]

# Step 6: Plot animated bar chart
fig_bar = px.bar(
    top_predicted,
    x="stockvalue",
    y="major",
    color="major",
    orientation="h",
    animation_frame=top_predicted["year_month"].astype(str),
    range_x=[0, top_predicted["stockvalue"].max() * 1.1],
    title=f"Predicted Top {top_n} Major Categories by Stock Value (2025)",
    labels={"stockvalue": "Predicted Stock Value", "major": "Category"},
    height=600
)

fig_bar.update_layout(
    yaxis={'categoryorder': 'total ascending'},
    showlegend=False,
    xaxis_title="Predicted Stock Value",
    yaxis_title="Major Category"
)

# === 2. Top Trending Items (Violin Plot) ===
count_option = st.selectbox("Select number of top items to show:", [10, 50, 100], index=0)
top_items_df = df.groupby("description")["stockvalue"].sum().reset_index().sort_values("stockvalue", ascending=False).head(100)

fig_violin = px.violin(
    top_items_df,
    y="stockvalue",
    box=True,
    title=f"Trending Items 2025",
    labels={"value": "Stock Value"},
    color="description"
)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

# === Displaying the graphs side by side ===
row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    st.plotly_chart(fig_violin, use_container_width=True)  # Violin plot
with row1_col2:
    st.plotly_chart(fig_bar, use_container_width=True)  # Bar chart

# === 3. Search by Description ===
import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA

# === Subheader and Item Selection ===
st.subheader("🔍 Search by Item Description")
desc_list = sorted(df["description"].dropna().unique())
selected_desc = st.selectbox("Select Item", desc_list)

if selected_desc:
    item_df = df[df["description"] == selected_desc].copy()
    item_df["txndate"] = pd.to_datetime(item_df["txndate"])

    # === Monthly Aggregation for Forecasting ===
    item_df["month"] = item_df["txndate"].dt.to_period("M").dt.to_timestamp()
    monthly_df = item_df.groupby("month")["qty"].sum().reset_index().sort_values("month")

    # === ARIMA Forecasting ===
    monthly_df.set_index("month", inplace=True)
    model = ARIMA(monthly_df["qty"], order=(1, 1, 1))
    model_fit = model.fit()

    forecast_steps = 12
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_df = forecast.summary_frame()
    forecast_df["month"] = pd.date_range(start=monthly_df.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
    forecast_df.rename(columns={"mean": "Forecast"}, inplace=True)

    # Combine historical + forecast
    monthly_df_reset = monthly_df.reset_index()
    all_data = pd.concat([
        monthly_df_reset.rename(columns={"qty": "Quantity"}),
        forecast_df[["month", "Forecast"]]
    ], ignore_index=True)

    # Plot ARIMA forecast
    fig_trends = px.line(
        all_data,
        x="month",
        y=["Quantity", "Forecast"],
        markers=True,
        title=f"📦 Stock Forecast (ARIMA) - {selected_desc}",
        labels={"value": "Quantity", "month": "Month-Year", "variable": "Type"}
    )

    # === Forecast Next Month Requirement Logic ===
    qty_series = item_df.groupby(item_df["txndate"].dt.to_period("M"))["qty"].sum()
    qty_series.index = qty_series.index.to_timestamp()

    if len(qty_series) >= 4:
        model = ARIMA(qty_series, order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)

        next_month_qty = forecast.values[0]
        current_stock = item_df["qty"].sum()

        st.subheader("🔮 Prediction")
        st.markdown(f"**Estimated Required Qty for Next Month:** `{int(next_month_qty)}`")
        st.markdown(f"**Current Available Stock:** `{int(current_stock)}`")

        # Probability Warnings
        if current_stock <= next_month_qty:
            st.error("⚠️ High probability this item will run out next month!")
        elif current_stock - next_month_qty < 10:
            st.warning("🟠 Low stock buffer, may run out soon.")
        else:
            st.success("✅ Stock level is sufficient for next month.")
    else:
        st.info("📉 Not enough data for forecasting. Need at least 4 months of data.")

# === Calculate Repurchase Likelihood for 2025 ===
df["txndate"] = pd.to_datetime(df["txndate"])
df["year"] = df["txndate"].dt.year
df_2024 = df[df["year"] == 2024].copy()

# Total sales per item
item_sales = df_2024.groupby("description")["qty"].sum().reset_index().sort_values("qty", ascending=False)

# Purchase frequency
item_frequency = df_2024.groupby("description")["txndate"].nunique().reset_index().rename(columns={"txndate": "purchase_frequency"})

# Merge and compute repurchase likelihood
item_data = pd.merge(item_sales, item_frequency, on="description")
item_data["repurchase_likelihood"] = item_data["qty"] * item_data["purchase_frequency"]
item_data["predicted_2025_likelihood"] = item_data["repurchase_likelihood"] * 1.1  # +10%

top_items_2025 = item_data.sort_values("predicted_2025_likelihood", ascending=False).head(20)

# Create repurchase chart
fig = px.bar(
    top_items_2025,
    x="predicted_2025_likelihood",
    y="description",
    color="description",
    orientation="h",
    title="Top 20 Items Predicted to be Bought Again by Customers in 2025",
    labels={"predicted_2025_likelihood": "Predicted Repurchase Likelihood", "description": "Item Description"},
    height=600
)

# === Display Both Charts Side-by-Side ===
row_col1, row_col2 = st.columns(2)

with row_col1:
    st.subheader(f"📅 Forecasted Stock Trends for: {selected_desc}")
    st.plotly_chart(fig_trends, use_container_width=True)

with row_col2:
    st.subheader("📦 Top Repurchase Predictions for 2025")
    st.plotly_chart(fig, use_container_width=True)
