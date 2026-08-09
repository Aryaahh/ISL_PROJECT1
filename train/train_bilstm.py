import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_FOLDER = os.path.join(
    PROJECT_ROOT,
    "data",
    "training"
)

X_PATH = os.path.join(
    DATA_FOLDER,
    "X.npy"
)

Y_PATH = os.path.join(
    DATA_FOLDER,
    "y.npy"
)

LABELS_PATH = os.path.join(
    DATA_FOLDER,
    "labels.npy"
)

MODEL_FOLDER = os.path.join(
    PROJECT_ROOT,
    "models"
)

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)

MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "isl_bilstm_model.keras"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("ISL BiLSTM MODEL TRAINING")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nDataset:")
print(DATA_FOLDER)

print("\nX:")
print(X_PATH)

print("\ny:")
print(Y_PATH)

print("\nLabels:")
print(LABELS_PATH)


# ============================================================
# CHECK FILES
# ============================================================

for file_path in [
    X_PATH,
    Y_PATH,
    LABELS_PATH
]:

    if not os.path.exists(file_path):

        print("\nERROR: File not found:")
        print(file_path)

        raise SystemExit


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)

X = np.load(
    X_PATH,
    mmap_mode="r"
)

y = np.load(
    Y_PATH
)

labels = np.load(
    LABELS_PATH,
    allow_pickle=True
)


print("\nX shape:", X.shape)
print("y shape:", y.shape)
print("Number of labels:", len(labels))


# ============================================================
# DISPLAY LABELS
# ============================================================

print("\nLabels:")

for i, label in enumerate(labels):

    print(f"{i}: {label}")


# ============================================================
# DATA VALIDATION
# ============================================================

if X.ndim != 3:

    raise ValueError(
        f"X must have 3 dimensions. Found: {X.shape}"
    )


if X.shape[1] != 30:

    raise ValueError(
        f"Expected 30 frames. Found: {X.shape[1]}"
    )


if X.shape[2] != 1659:

    raise ValueError(
        f"Expected 1659 features. Found: {X.shape[2]}"
    )


if len(X) != len(y):

    raise ValueError(
        "X and y contain different numbers of samples."
    )


NUM_CLASSES = len(labels)

unique_classes = np.unique(y)

if len(unique_classes) != NUM_CLASSES:

    raise ValueError(
        f"Dataset contains {len(unique_classes)} classes "
        f"but labels.npy contains {NUM_CLASSES} labels."
    )


print("\nDataset validation successful.")

print("\nTotal samples :", len(X))
print("Sequence length:", X.shape[1])
print("Feature size   :", X.shape[2])
print("Number classes :", NUM_CLASSES)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

for class_id, label in enumerate(labels):

    count = np.sum(y == class_id)

    print(
        f"{class_id:2d}: {str(label):10s} -> {count} samples"
    )


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 70)
print("CREATING TRAIN / VALIDATION SPLIT")
print("=" * 70)

indices = np.arange(
    len(y)
)

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples  :", len(train_idx))
print("Validation samples:", len(val_idx))


# ============================================================
# CONVERT DATA
# ============================================================

X_train = np.array(
    X[train_idx],
    dtype=np.float32
)

X_val = np.array(
    X[val_idx],
    dtype=np.float32
)

y_train = y[train_idx]

y_val = y[val_idx]


print("\nTraining X shape:")
print(X_train.shape)

print("\nValidation X shape:")
print(X_val.shape)


# ============================================================
# ONE-HOT ENCODING
# ============================================================

y_train_cat = to_categorical(
    y_train,
    num_classes=NUM_CLASSES
)

y_val_cat = to_categorical(
    y_val,
    num_classes=NUM_CLASSES
)

print("\nOne-hot encoding completed.")


# ============================================================
# BUILD MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING BiLSTM MODEL")
print("=" * 70)


model = Sequential([

    Bidirectional(
        LSTM(
            128,
            return_sequences=True
        ),
        input_shape=(
            30,
            1659
        )
    ),

    Dropout(0.30),

    Bidirectional(
        LSTM(
            64
        )
    ),

    Dropout(0.30),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.30),

    Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n")

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1
)


early_stopping = EarlyStopping(

    monitor="val_accuracy",

    patience=10,

    mode="max",

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=4,

    min_lr=0.00001,

    verbose=1
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

EPOCHS = 60

BATCH_SIZE = 16


print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print("\nEpochs    :", EPOCHS)
print("Batch size:", BATCH_SIZE)

print("\nBest model will be saved to:")
print(MODEL_PATH)


# ============================================================
# TRAIN
# ============================================================

history = model.fit(

    X_train,

    y_train_cat,

    validation_data=(
        X_val,
        y_val_cat
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    shuffle=True,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)


val_loss, val_accuracy = model.evaluate(

    X_val,

    y_val_cat,

    verbose=1
)


print("\nValidation loss:")
print(f"{val_loss:.4f}")

print("\nValidation accuracy:")
print(f"{val_accuracy * 100:.2f}%")


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    MODEL_PATH
)


print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print("\nModel saved to:")
print(MODEL_PATH)

print("\nClasses trained:")

for i, label in enumerate(labels):

    print(f"{i}: {label}")


print("\nDataset:")
print(f"{len(X)} samples")

print("\nClasses:")
print(f"{NUM_CLASSES}")

print("\nInput:")
print("(30 frames, 1659 features)")