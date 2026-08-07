import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from src.config import(
    DATASET_FILE,
    DROP_COLUMNS,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE
)

# Data Loading Function
def load_data():
    df = pd.read_csv(DATASET_FILE)
    return df

# Data Cleaning Function
def clean_data(df):
    df = df.copy()
    df.drop(DROP_COLUMNS, axis = 1, inplace = True)
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors = "coerce"
    )
    df.dropna(inplace = True)
    return df

# Feature/Target Separation
def split_features_target(df):
    x = df.drop(TARGET_COLUMN, axis = 1)
    y = df[TARGET_COLUMN]
    return x, y

# Encode Target
def encode_target(y):
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    return y, encoder 

# One-Hot Encode Features
def encode_features(x):
    x = pd.get_dummies(
        x,
        drop_first = True
    )
    return x

# Train-Test Split
def split_data(x,y):
    return train_test_split(
        x,y,
        test_size = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify= y
    )

# Feature Scaling
def scale_features(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, scaler

# Main Preprocessing Function
def preprocess_data():
    df = load_data()
    df = clean_data(df)
    x, y = split_features_target(df)
    y, label_encoder = encode_target(y)
    x = encode_features(x)
    feature_columns = x.columns.tolist()
    x_train, x_test, y_train, y_test = split_data(x, y)
    x_train, x_test, scaler = scale_features(
        x_train, x_test
    )

    return (
        x_train, x_test, y_train, y_test,
        scaler,
        label_encoder, feature_columns
    )

if __name__ == "__main__":
    x_train, x_test, y_train, y_test ,scaler, encoder = preprocess_data()
    print("Train Shape:", x_train.shape)
    print("Test Shape:", x_test.shape)