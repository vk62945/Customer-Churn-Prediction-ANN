import tensorflow as tf
import joblib 
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

from preprocess import preprocess_data

from src.config import (
    HIDDEN_LAYERS,
    LEARNING_RATE,
    DROPOUT_1,
    DROPOUT_2,
    MODEL_FILE,
    SCALER_FILE,
    FEATURE_COLUMNS_FILE,
    EPOCHS,
    BATCH_SIZE,
    VALIDATION_SPLIT
)

# Build Model Function

def build_model(input_dim):
    model = Sequential()

    # First Hidden Layer
    model.add(
        Dense(
            HIDDEN_LAYERS[0],
            activation = "relu",
            input_shape = (input_dim,)
        )
    )
    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT_1))

    # Remaining Hidden Layers
    for units in HIDDEN_LAYERS[1:]:
        model.add(
            Dense(
                units,
                activation = "relu"
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(DROPOUT_2))

    #Output Layer
    model.add(
        Dense(
            1,
            activation = "sigmoid"
        )
    )

    #Complie Model
    model.compile(
        optimizer = Adam(learning_rate = LEARNING_RATE),
        loss = "binary_crossentropy",
        metrics = ["accuracy"]
    )

    return model

def get_callbacks():
    early_stopping =EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights = True
    )

    checkpoint = ModelCheckpoint(
        MODEL_FILE,
        monitor = "val_loss",
        save_best_only = True,
        verbose=1
    )
    return [early_stopping, checkpoint]

# Create Training Function
def train_model():
    (
        x_train,
        x_test,
        y_train,
        y_test,
        scaler,
        label_encoder,
        feature_columns
    ) = preprocess_data()
    joblib.dump(
        scaler,
        SCALER_FILE
    )
    joblib.dump(
        feature_columns,
        FEATURE_COLUMNS_FILE
    )
    model = build_model(x_train.shape[1])

    history = model.fit(
        x_train,
        y_train,
        validation_split = VALIDATION_SPLIT,
        epochs = EPOCHS,
        batch_size = BATCH_SIZE,
        callbacks = get_callbacks(),
        verbose = 1
    )
    return (
        model,
        history,
        x_test,
        y_test,
        scaler,
        label_encoder,
    )

if __name__ == "__main__":
    model, history, *_ = train_model()
    print("\n Training completed successfully")
    print(f"Model saved to : {MODEL_FILE}")

