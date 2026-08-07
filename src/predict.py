import numpy as np 
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

from src.config import (
    MODEL_FILE,
    THRESHOLD,
    FEATURE_COLUMNS_FILE,
    SCALER_FILE
)

def load_trained_model():
    model = load_model(MODEL_FILE)
    return model 

def load_scaler():
    return joblib.load(SCALER_FILE)

def load_feature_columns():
    return joblib.load(FEATURE_COLUMNS_FILE)

def preprocess_input(input_df):
    scaler = load_scaler()
    feature_columns = load_feature_columns()
    input_df = pd.get_dummies(
        input_df,
        drop_first = True
    )
    #Align Columns
    input_df = input_df.reindex(
        columns = feature_columns,
        fill_value=0
    )

    input_scaled = scaler.transform(
        input_df
    )

    return input_scaled

def predict_customer(model, input_df):
    processed_input = preprocess_input(input_df)
    processed_input = tf.convert_to_tensor(
        processed_input,
        dtype=tf.float32
    )

    probability = model(
        processed_input,
        training=False
    ).numpy()[0][0]
    prediction = int(probability >= THRESHOLD)
    return {
        "prediction": prediction,
        "probability": round(float(probability),4)
    }

if __name__ == "__main__":
    model = load_trained_model()
    print("=" * 50)
    print("ANN Model Loaded Successfully")
    print("=" * 50)

    model.summary()