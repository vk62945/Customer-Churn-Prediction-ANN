from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"
MODEL_PATH = PROJECT_ROOT / "models"

# Dataset File and Model File
DATASET_FILE = RAW_DATA_PATH / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_FILE = MODEL_PATH / "best_ann_model.keras"
SCALER_FILE = MODEL_PATH / "scaler.pkl"
FEATURE_COLUMNS_FILE = MODEL_PATH / "feature_columns.pkl"

# Training Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.2
EPOCHS = 100
BATCH_SIZE = 32

# ANN Architecture 
HIDDEN_LAYERS = [32, 16]
LEARNING_RATE = 0.1
DROPOUT_1 = 0.3
DROPOUT_2 = 0.2
THRESHOLD = 0.30

# TARGET COLUMN
TARGET_COLUMN = "Churn"

#Columns to Drop
DROP_COLUMNS = [
    "customerID"
]

#Create Model Directory
MODEL_PATH.mkdir(
    exist_ok = True
)
