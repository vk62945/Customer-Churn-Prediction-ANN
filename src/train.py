import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.optimizers import Adam

from config import (
    HIDDEN_LAYERS,
    LEARNING_RATE,
    DROPOUT_1,
    DROPOUT_2
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

if __name__ == "__main__":
    model = build_model(input_dim = 30)
    model.summary()
