import numpy as np 
import pandas as pd

from tensorflow.keras.models import load_model

from src.config import (
    MODEL_FILE,
    THRESHOLD
)

def load_trained_model():
    model = load_model(MODEL_FILE)
    return model 

def predict_customer(model, input_data):
    if isinstance(input_data, list):
        input_data = np.array(input_data)

    #Ensure numpy array
    input_data = np.asarray(input_data)

    #Reshape for single prediction
    input_data = input_data.reshape(1,-1)

    #Predict probability
    probability = model.predict(
        input_data,
        verbose = 0
    )[0][0]

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