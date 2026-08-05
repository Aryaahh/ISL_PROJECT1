import numpy as np
import tensorflow as tf
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# Load using memory mapping
X = np.load(
    "data/sequences/X.npy",
    mmap_mode="r"
)

y = np.load(
    "data/sequences/y.npy"
)


print("Input shape:", X.shape)
print("Labels shape:", y.shape)


num_classes = len(np.unique(y))

print("Classes:", num_classes)


# Split only indexes (not data)
indices = np.arange(len(y))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42
)


# Convert labels
y_cat = to_categorical(
    y,
    num_classes
)


# Data generator
def data_generator(indexes, batch_size=16):

    while True:

        np.random.shuffle(indexes)

        for start in range(0, len(indexes), batch_size):

            batch_idx = indexes[start:start+batch_size]

            X_batch = np.array(X[batch_idx])

            y_batch = y_cat[batch_idx]

            yield X_batch, y_batch



# Model
model = Sequential([

    Bidirectional(
        LSTM(
            128,
            return_sequences=True
        ),
        input_shape=(30,1659)
    ),

    Dropout(0.3),

    Bidirectional(
        LSTM(64)
    ),

    Dropout(0.3),

    Dense(
        128,
        activation="relu"
    ),

    Dense(
        num_classes,
        activation="softmax"
    )
])


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# Train
model.fit(
    data_generator(train_idx),
    steps_per_epoch=len(train_idx)//16,
    epochs=30,
    validation_data=data_generator(test_idx),
    validation_steps=len(test_idx)//16
)


model.save(
    "isl_bilstm_model.h5"
)


print("Training completed!")
