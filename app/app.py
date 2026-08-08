import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys 
import time 
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

st.caption(
    "AI-powered customer churn prediction using an optimized Artificial Neural Network."
)

st.divider()

@st.cache_resource
def get_model():
    return load_trained_model()

try:
    model = get_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

st.sidebar.header("📋 Model Information")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.markdown("""
### Model Details

- Architecture: 32 → 16
- Optimizer: Adam
- Learning Rate: 0.1
- Threshold: 0.30
- Framework: TensorFlow / Keras
""")
st.sidebar.divider()

st.sidebar.markdown("""
### 👨‍💻 Developer

Vivek Kumar

AWS Data Engineer

AI Engineer Aspirant
""")
st.sidebar.divider()
st.sidebar.caption("Version 1.0.0 | Last Updated: Aug 2026")


with st.container(border=True):
    st.subheader("📊 Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", "ANN")

    with col2:
        st.metric("F1 Score", "0.6134")

    with col3:
        st.metric("ROC-AUC", "0.8308")
st.divider()
st.subheader("📝 Customer Information")

with st.form("prediction_form"):
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
            "Total Charges",
            min_value = 0.0,
            value = 1000.0,
            step = 10.0
        )

    st.divider()
    predict_button = st.form_submit_button(
        "🚀 Predict Customer Churn",
        use_container_width=True
    )
st.divider()

if predict_button:
    st.toast("Customer profile received.")

    if(phone_service == "No" and multiple_lines == "Yes"):
        st.error("A customer without phone service cannot have multiple lines.")
        st.stop()
    if internet_service == "No":
        invalid_services = [
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies
        ]
        if "Yes" in invalid_services:
            st.error(
                "Customers without internet service cannot subscribe to internet-related services."
            )
            st.stop()
    if monthly_charges == 0:
        st.warning(
            "Monthly charges are unusually low. Please verify the entered value."
        )
    expected_total = tenure * monthly_charges
    if tenure > 12 and total_charges < (0.25 * expected_total):
        st.warning(
            "Total charges appear unusually low compared to tenure and monthly charges."
        )

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
    report = customer.copy()
    with st.spinner("Analyzing customer profile..."):
        time.sleep(1)
        try:
            result = predict_customer(
                model,
                customer
            )
        except Exception as e:
            st.error(
                "Prediction failed"
            )
            st.exception(e)
            st.stop()
    if result is None:
        st.error("Prediction could not be generated.")
        st.stop()
    st.toast("✅ Prediction completed successfully!")
    prediction = result["prediction"]
    probability = result["probability"]
    report["Prediction"] = (
        "Churn" if prediction else "No Churn"
    )
    report["Probability"] = probability * 100
    risk_factors = []
    if contract == "Month-to-month":
        risk_factors.append(
            "Month-to-month contracts generally have higher churn risk."
        )
    if tenure < 12:
        risk_factors.append(
            "Short customer tenure may indicate weaker customer loyalty."
        )
    if internet_service == "Fiber optic":
        risk_factors.append(
            "Fiber optic customers have historically shown higher churn rates."
        )
    if monthly_charges > 80:
        risk_factors.append(
            "Higher monthly charges may increase churn risk."
        )
    if tech_support == "No":
        risk_factors.append(
            "Customers without technical support tend to churn more frequently."
        )
    if online_security == "No":
        risk_factors.append(
            "Customers without online security services are often at higher risk."
        )
    result_container = st.container()
    with result_container:
        st.subheader("🎯 Prediction Results")
        if prediction == 1:
            st.error("⚠ Customer is likely to churn.")
            st.progress(probability)
            st.caption(
                f"Model estimates a {probability:.1%} chance of churn."
            )
        else:
            st.success("✅ Customer Likely to Stay")
            st.progress(probability)
            st.caption(
                f"Model estimates a {probability:.1%} chance of churn."
            )

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label = "Churn Probability",
                value = f"{probability:.2%}"
            )
        with col2:
            if probability >= 0.70:
                risk = "🔴 Very High"
            elif probability >= 0.50:
                risk = "🟠 High"
            elif probability >= 0.30:
                risk = "🟡 Moderate"
            else:
                risk = "🟢 Low"
            st.metric(
                "Risk Level",
                risk
            )
            report["Risk"] = risk
        st.divider()
        if prediction == 1:
            st.warning("""
            ### Recommendation
            • Contact the customer.
            • Offer retention discounts.
            • Review current service plan.
            • Provide loyalty benefits.
            """)
        else:
            st.info("""
            ### Recommendation
            • Customer appears satisfied.
            • Continue current engagement.
            • Monitor periodically.
            """)
    st.divider()

    with st.expander("📖 Why did the model predict this?", expanded=False):

        st.markdown("### 📊 Prediction Summary")
        st.markdown("### ⚠ Risk Factors")

        if len(risk_factors) > 0:
            for factor in risk_factors:
                st.write(f"• {factor}")
        else:
            st.success(
                "No major churn risk indicators were identified based on the provided customer profile."
            )
        st.divider()
        st.markdown("### 👤 Customer Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Gender:** {gender}")
            st.write(f"**Senior Citizen:** {senior_citizen}")
            st.write(f"**Partner:** {partner}")
            st.write(f"**Dependents:** {dependents}")
            st.write(f"**Tenure:** {tenure} months")
        with col2:
            st.write(f"**Contract:** {contract}")
            st.write(f"**Internet Service:** {internet_service}")
            st.write(f"**Payment Method:** {payment_method}")
            st.write(f"**Monthly Charges:** ${monthly_charges:.2f}")
            st.write(f"**Total Charges:** ${total_charges:.2f}")

        st.divider()
        with st.expander("Customer information used for prediction", expanded=False):
            st.dataframe(
                customer,
                use_container_width=True
            )
    st.download_button(
        label = "📄 Download Prediction Report",
        data = report.to_csv(index=False),
        file_name="customer_prediction.csv",
        mime="text/csv"
    )
st.info(
    """
ℹ️ **Disclaimer**

This prediction is generated using a trained Artificial Neural Network and is intended to support business decision-making. It should not be used as the sole basis for customer retention decisions.
"""
)    
st.divider()
st.caption(
    "Developed using TensorFlow • Streamlit • Scikit-Learn • Python  \n"
    "© 2026 Vivek Kumar"
)