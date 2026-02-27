import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import re

st.set_page_config(page_title="Ice Cream Sales Forecast", layout="wide")

st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Sales Forecast"])

template_style = "plotly_dark"

# Load trained model
model = joblib.load("sales_model.pkl")

# =========================================================
# SALES PAGE
# =========================================================
if page == "Sales Forecast":

    st.title("🍦 Hyderabad Ice Cream Sales Forecast")

    df = pd.read_csv("Hyderabad_IceCream_2024_2025_With_Sales.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    future_days = st.slider("Select Forecast Days", 10, 120, 50)

    # ---------------- FUTURE DATA GENERATION ----------------
    future_dates = pd.date_range(
        df["Date"].iloc[-1] + pd.Timedelta(days=1),
        periods=future_days,
        freq="D"
    )

    future_df = pd.DataFrame()
    future_df["Date"] = future_dates
    future_df["Month"] = future_df["Date"].dt.month
    future_df["Weekend"] = future_df["Date"].dt.dayofweek.isin([5,6]).astype(int)
    future_df["Holiday"] = 0

    # Simple seasonal temperature simulation
    future_df["Temperature"] = 25 + 10*np.sin(2*np.pi*future_df.index/365)

    # Predict
    X_future = future_df[["Temperature","Month","Weekend","Holiday"]]
    predictions = model.predict(X_future)

    # ================= KPIs =================
    col1, col2, col3 = st.columns(3)
    col1.metric("Last Actual Sales", round(df["Sales"].iloc[-1],2))
    col2.metric("Next Day Prediction", round(predictions[0],2))
    col3.metric("Total Next 7 Days", round(np.sum(predictions[:7]),2))

    # ================= GRAPH =================
    chart_type = st.selectbox(
        "Visualization Type",
        ["Line Chart", "Scatter Plot", "Area Chart"]
    )

    fig = go.Figure()

    # Actual
    if chart_type == "Line Chart":
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["Sales"],
            mode='lines',
            name="Actual",
            line=dict(color="blue", width=3)
        ))

    elif chart_type == "Scatter Plot":
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["Sales"],
            mode='markers',
            name="Actual",
            marker=dict(color="blue")
        ))

    elif chart_type == "Area Chart":
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["Sales"],
            fill='tozeroy',
            mode='lines',
            name="Actual",
            line=dict(color="blue")
        ))

    # Predicted
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        mode='lines',
        name="Predicted",
        line=dict(color="yellow", width=3, dash='dash')
    ))

    fig.update_layout(
        title="Hyderabad Ice Cream Sales Forecast",
        template=template_style
    )

    st.plotly_chart(fig, use_container_width=True)

    # ================= DOWNLOAD =================
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Sales": predictions
    })

    st.download_button(
        "Download Forecast CSV",
        forecast_df.to_csv(index=False),
        "forecast.csv",
        "text/csv"
    )

    # ================= SMART CHATBOT =================
    st.subheader("🤖 Smart Assistant")
    user_input = st.text_input("Ask: total next 20 days / average next 10 days")

    if user_input:
        user_input = user_input.lower()
        numbers = re.findall(r'\d+', user_input)

        if numbers:
            days = int(numbers[0])

            if days > future_days:
                st.warning("Selected days exceed forecast range.")
            else:
                selected = predictions[:days]

                if "average" in user_input:
                    st.success(f"Average: {round(np.mean(selected),2)}")
                elif "max" in user_input:
                    st.success(f"Maximum: {round(np.max(selected),2)}")
                elif "min" in user_input:
                    st.success(f"Minimum: {round(np.min(selected),2)}")
                else:
                    st.success(f"Total: {round(np.sum(selected),2)}")
        else:
            st.warning("Please include number of days.")