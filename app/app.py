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
        ["yes", "No", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )
    device_protection = st.selectbox(
        "Device Protection",
        ["yes", "No", "No internet service"]
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

predict_button = st.button(
    "Predict Customer Churn"
)

if predict_button:
    st.write("Inputs captured successfully")