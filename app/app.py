import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys 

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.predict import (
    load_trained_model,
    predict_customer
)

st.set_page_config(
    page_title = "Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Customer Churn Prediction")
st.markdown(
    """
Predict whether a telecom customer is likely to churn using an optimized Artificial Neural Network.
"""
)

model = load_trained_model()

st.sidebar.title("Project Information")
st.sidebar.markdown(
    """
### Model

Artificial Neural Network

### Framework

TensorFlow / Keras

### Target

Customer Churn
"""
)

st.info(
    "Customer input form"
)

left_col, right_col = st.columns(2)

with left_col:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )
    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No",  "Yes"]
    )
    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )
    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )
    tenure = st.slider(
        "Tenure (Months)",
        0, 72, 12
    )
    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )
    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]      
    )

with right_col:
    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )
    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value = 0.0,
        value = 70.0,
        step = 0.5
    )
    total_charges = st.number_input(
        "Toatl Charges",
        min_value = 0.0,
        value = 1000.0,
        step = 10.0
    )

st.divider()
predict_button = st.button(
    "Predict Customer Churn"
)
st.divider()

if predict_button:
    customer = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    result = predict_customer(
        model,
        customer
    )
    probability = result["probability"]
    if result["prediction"] == 1:
        st.error("⚠️ Customer is likely to churn.")

    else:
        st.success("✅ Customer is likely to stay.")

    st.metric(
        label = "Churn Probability",
        value = f"{probability:.2%}"
    )
    if probability >= 0.80:
        st.warning("High Confidence Prediction")
    elif probability >= 0.60:
        st.info("Moderate Confidence Prediction")
    else:
        st.success("Low Risk Customer")
    
    with st.expander("Customer Details"):
        st.dataframe(customer)