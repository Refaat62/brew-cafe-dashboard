import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

# ======================================================
# Load data and model
# ======================================================

@st.cache_data
def load_data():
    df = pd.read_excel("Brew_Café_Cleaned.xlsx")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

df = load_data()
model = load_model()

# ======================================================
# Page setup
# ======================================================
st.set_page_config(page_title="☕ Brew Café Dashboard", layout="wide")
st.title("☕ Brew Café Analytics & AI Revenue Predictor")
st.markdown("An interactive dashboard for understanding café performance and predicting daily revenue 💡")

# ======================================================
# Data Analysis Dashboard (Plotly)
# ======================================================
st.header("📊 Data Analysis Dashboard")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    " Daily Revenue", 
    " Time of Day", 
    " Top Services",
    " Weekly Trend", 
    " Monthly Trend", 
    " Customer Comparison",
    " Ads vs Revenue"
])

# =============== Daily Revenue ==================
with tab1:
    st.subheader("Daily Revenue Over Time")
    daily_rev = df.groupby("Date")["Revenue ($)"].sum().reset_index()
    fig = px.line(daily_rev, x="Date", y="Revenue ($)", title="Daily Revenue Trend", markers=True)
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Total Revenue", f"${df['Revenue ($)'].sum():,.2f}")
    st.metric("Number of Days", f"{df['Date'].nunique()} days")

# =============== Time of Day ==================
with tab2:
    st.subheader("Average Revenue by Time of Day")
    time_rev = df.groupby("Time of Day")["Revenue ($)"].mean().reset_index().sort_values(by="Revenue ($)", ascending=False)
    fig = px.bar(time_rev, x="Time of Day", y="Revenue ($)", color="Time of Day", title="Average Revenue by Time of Day")
    st.plotly_chart(fig, use_container_width=True)

# =============== Top Services ==================
with tab3:
    st.subheader("Top 5 Services by Total Revenue")
    top_services = df.groupby("Service")["Revenue ($)"].sum().sort_values(ascending=False).head(5).reset_index()
    fig = px.bar(top_services, x="Service", y="Revenue ($)", text="Revenue ($)", color="Service", title="Top 5 Revenue-Generating Services")
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

# =============== Weekly Trend ==================
with tab4:
    st.subheader("Average Revenue by Day of Week")
    df["Day of Week"] = df["Date"].dt.day_name()
    week_rev = df.groupby("Day of Week")["Revenue ($)"].mean().reindex(
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    ).reset_index()
    fig = px.bar(week_rev, x="Day of Week", y="Revenue ($)", color="Day of Week", title="Revenue by Weekday")
    st.plotly_chart(fig, use_container_width=True)

# =============== Monthly Trend ==================
with tab5:
    st.subheader("Monthly Revenue Trend")
    df["Month"] = df["Date"].dt.strftime("%B")
    month_rev = df.groupby("Month")["Revenue ($)"].sum().reset_index()
    fig = px.line(month_rev, x="Month", y="Revenue ($)", markers=True, title="Monthly Revenue Overview")
    st.plotly_chart(fig, use_container_width=True)

# =============== Customer Comparison ==================
with tab6:
    st.subheader("Average Revenue by Customer Type")
    cust_rev = df.groupby("Customer Type")["Revenue ($)"].mean().reset_index()
    fig = px.bar(cust_rev, x="Customer Type", y="Revenue ($)", color="Customer Type", title="Customer Type Revenue Comparison")
    st.plotly_chart(fig, use_container_width=True)

# =============== Ad Spend vs Revenue ==================
with tab7:
    st.subheader("Ad Spend vs Revenue Correlation")
    fig = px.scatter(df, x="Ad Spend ($)", y="Revenue ($)", color="Time of Day",
                     trendline="ols", title="Advertising Spend vs Revenue")
    st.plotly_chart(fig, use_container_width=True)
    corr = df["Ad Spend ($)"].corr(df["Revenue ($)"])
    st.write(f"📊 Correlation between Ad Spend and Revenue: **{corr:.2f}**")

# ======================================================
# AI Revenue Prediction
# ======================================================
st.header("🤖 Predict Café Revenue")

st.markdown("Enter details below to estimate the expected revenue:")

col1, col2, col3 = st.columns(3)
with col1:
    time_of_day = st.selectbox("Time of Day", df["Time of Day"].unique())
with col2:
    service = st.selectbox("Service Type", df["Service"].unique())
with col3:
    customer_type = st.selectbox("Customer Type", df["Customer Type"].unique())

col4, col5 = st.columns(2)
with col4:
    ad_spend = st.number_input("Advertising Spend ($)", min_value=0.0, max_value=500.0, value=50.0)
with col5:
    conversions = st.number_input("Conversions", min_value=0, max_value=20, value=5)

# Encoding function
def encode_input(df, time_of_day, service, customer_type, ad_spend, conversions):
    label_cols = ['Time of Day', 'Service', 'Customer Type']
    encoders = {}
    df_enc = df.copy()
    for col in label_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])
        encoders[col] = le

    input_data = pd.DataFrame({
        'Ad Spend ($)': [ad_spend],
        'Time of Day': [encoders['Time of Day'].transform([time_of_day])[0]],
        'Service': [encoders['Service'].transform([service])[0]],
        'Customer Type': [encoders['Customer Type'].transform([customer_type])[0]],
        'Conversions': [conversions]
    })
    return input_data

# Prediction button
if st.button("🔮 Predict Revenue"):
    try:
        input_df = encode_input(df, time_of_day, service, customer_type, ad_spend, conversions)
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Expected Revenue: **${prediction:.2f}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# ======================================================
# Footer
# ======================================================
st.markdown("---")
st.markdown("Developed by **AI Assistant for Brew Café** | Smart Analytics and Revenue Forecasting ☕")
