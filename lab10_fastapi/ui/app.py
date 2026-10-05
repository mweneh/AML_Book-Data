import os
import requests
import streamlit as st


st.set_page_config(page_title="Fraud Detection", page_icon="💰",)

# Where the FastAPI server lives (override with the API_URL environment variable)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8001/predict")

st.title("Fraud Detection")
st.write("Enter a transaction and the model will predict whether it is fraud.")

# Inputs, grouped into two columns. Defaults are a real fraud example from the data.
col1, col2 = st.columns(2)

with col1:
    st.subheader("Amount & account")
    abs_amount = st.number_input("Amount", min_value=0.0, value=1571.0)
    log_amount = st.number_input("Log amount", value=7.36)
    balance_after = st.number_input("Balance after", value=15237.0)
    log_balance = st.number_input("Log balance", value=9.63)
    is_debit = st.selectbox("Is debit?", [0, 1], index=0)
    risk_score = st.slider("Risk score", 0.0, 1.0, 1.0)
    txn_type_enc = st.number_input("Transaction type (encoded)", min_value=0, value=5)
    region_enc = st.number_input("Region (encoded)", min_value=0, value=0)
    segment_enc = st.number_input("Segment (encoded)", min_value=0, value=2)

with col2:
    st.subheader("Time & location")
    hour = st.slider("Hour", 0, 23, 0)
    day_of_week = st.slider("Day of week (0 = Mon)", 0, 6, 1)
    day_of_month = st.slider("Day of month", 1, 31, 16)
    month = st.slider("Month", 1, 12, 9)
    is_weekend = st.selectbox("Is weekend?", [0, 1], index=0)
    is_night = st.selectbox("Is night?", [0, 1], index=1)
    has_gps = st.selectbox("Has GPS?", [0, 1], index=1)
    gps_lat = st.number_input("GPS latitude", value=-3.74336, format="%.5f")
    gps_lon = st.number_input("GPS longitude", value=33.67018, format="%.5f")

if st.button("Predict", type="primary"):
    # Build the JSON body exactly like the API's Transaction model expects
    payload = {
        "abs_amount": abs_amount, "log_amount": log_amount, "is_debit": is_debit,
        "balance_after": balance_after, "log_balance": log_balance, "risk_score": risk_score,
        "hour": hour, "day_of_week": day_of_week, "day_of_month": day_of_month,
        "month": month, "is_weekend": is_weekend, "is_night": is_night,
        "has_gps": has_gps, "gps_lat": gps_lat, "gps_lon": gps_lon,
        "txn_type_enc": int(txn_type_enc), "region_enc": int(region_enc),
        "segment_enc": int(segment_enc),
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the API. Is it running?\n\n{e}")
    else:
        prob = result["fraud_probability"]
        if result["is_fraud"]:
            st.error(f"FRAUD — probability {prob:.1%}")
        else:
            st.success(f"Not fraud — fraud probability {prob:.1%}")
        st.progress(prob)
