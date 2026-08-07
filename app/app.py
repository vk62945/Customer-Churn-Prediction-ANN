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