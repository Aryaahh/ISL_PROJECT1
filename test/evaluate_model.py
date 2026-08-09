import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ============================================================
# ISL BiLSTM MODEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("ISL BiLSTM MODEL EVALUATION")
print("=" * 70)

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "training"
)

X_PATH = os.path.join(
    DATA_DIR,
    "X.npy"
)

Y_PATH = os.path.join(
    DATA_DIR,
    "y.npy"
)

# IMPORTANT:
# Use the labels.npy generated together with X.npy and y.npy.
# This contains the 21 labels used to train the current model.
LABEL_PATH = os.path.join(
    DATA_DIR,
    "labels.npy"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "isl_bilstm_model.keras"
)

# ============================================================
# PRINT PATHS
# ============================================================

print("\nProject folder:")
print(BASE_DIR)

print("\nDataset:")
print(X_PATH)

print("\nLabels:")
print(LABEL_PATH)

print("\nModel:")
print(MODEL_PATH)

# ============================================================
# CHECK FILES
# ============================================================

print("\n" + "-" * 70)
print("CHECKING FILES")
print("-" * 70)

required_files = {
    "X": X_PATH,
    "y": Y_PATH,
    "labels": LABEL_PATH,
    "model": MODEL_PATH
}

for name, path in required_files.items():

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n{name} file not found:\n{path}"
        )

    print(f"{name}: OK")

# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "-" * 70)
print("LOADING DATASET")
print("-" * 70)

X = np.load(
    X_PATH,
    mmap_mode="r"
)

y = np.load(
    Y_PATH
)

labels = np.load(
    LABEL_PATH,
    allow_pickle=True
)

print("\nX shape:", X.shape)
print("y shape:", y.shape)
print("Number of labels:", len(labels))

# ============================================================
# PRINT LABELS
# ============================================================

print("\nLabels:")

for i, label in enumerate(labels):
    print(f"{i}: {label}")

# ============================================================
# DATASET VALIDATION
# ============================================================

print("\n" + "-" * 70)
print("DATASET VALIDATION")
print("-" * 70)

if len(X) != len(y):
    raise ValueError(
        "\nNumber of X samples and y labels are different.\n"
        f"X samples: {len(X)}\n"
        f"y samples: {len(y)}"
    )

if X.ndim != 3:
    raise ValueError(
        f"\nExpected X to have 3 dimensions.\n"
        f"Actual shape: {X.shape}"
    )

if X.shape[1] != 30:
    raise ValueError(
        f"\nExpected sequence length = 30.\n"
        f"Actual sequence length: {X.shape[1]}"
    )

if X.shape[2] != 1659:
    raise ValueError(
        f"\nExpected feature size = 1659.\n"
        f"Actual feature size: {X.shape[2]}"
    )

if y.ndim != 1:
    y = np.asarray(y).reshape(-1)

num_classes = len(labels)

if num_classes != 21:
    raise ValueError(
        "\nExpected 21 labels, but found "
        f"{num_classes}.\n\n"
        "Make sure this file is being used:\n"
        f"{LABEL_PATH}"
    )

if np.min(y) < 0 or np.max(y) >= num_classes:
    raise ValueError(
        "\nInvalid class IDs found in y.\n"
        f"Minimum class ID: {np.min(y)}\n"
        f"Maximum class ID: {np.max(y)}\n"
        f"Number of classes: {num_classes}"
    )

print("\nDataset validation successful.")

print("\nTotal samples:", len(X))
print("Sequence length:", X.shape[1])
print("Feature size:", X.shape[2])
print("Number of classes:", num_classes)

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("CLASS DISTRIBUTION")
print("-" * 70)

for class_id in range(num_classes):

    count = int(np.sum(y == class_id))

    print(
        f"{class_id:2d}: "
        f"{str(labels[class_id]):10s} -> "
        f"{count} samples"
    )

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n" + "-" * 70)
print("CREATING EVALUATION SET")
print("-" * 70)

# Use the same type of 80/20 stratified split as training.
# If train_bilstm.py uses a specific random_state, change
# RANDOM_STATE below to the same value.

RANDOM_STATE = 42

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTest X shape:", X_test.shape)
print("Test y shape:", y_test.shape)

# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "-" * 70)
print("LOADING MODEL")
print("-" * 70)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("\nModel loaded successfully.")

print("Model input shape :", model.input_shape)
print("Model output shape:", model.output_shape)

# ============================================================
# MODEL / DATASET COMPATIBILITY CHECK
# ============================================================

model_input_shape = model.input_shape
model_output_classes = model.output_shape[-1]

expected_sequence_length = 30
expected_feature_size = 1659

if model_input_shape[-2] != expected_sequence_length:
    raise ValueError(
        "\nModel sequence length mismatch.\n"
        f"Model: {model_input_shape[-2]}\n"
        f"Expected: {expected_sequence_length}"
    )

if model_input_shape[-1] != expected_feature_size:
    raise ValueError(
        "\nModel feature size mismatch.\n"
        f"Model: {model_input_shape[-1]}\n"
        f"Expected: {expected_feature_size}"
    )

if model_output_classes != len(labels):
    raise ValueError(
        "\nModel output/classes mismatch.\n"
        f"Model classes : {model_output_classes}\n"
        f"Labels        : {len(labels)}"
    )

print("\nModel/dataset compatibility check successful.")

print(f"\nModel classes : {model_output_classes}")
print(f"Labels        : {len(labels)}")

# ============================================================
# PREDICTION
# ============================================================

print("\n" + "-" * 70)
print("RUNNING PREDICTIONS")
print("-" * 70)

predictions = model.predict(
    X_test,
    verbose=1
)

print("\nPrediction shape:", predictions.shape)

# Convert probabilities to class IDs
y_pred = np.argmax(
    predictions,
    axis=1
)

print("Predicted class shape:", y_pred.shape)

# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 70)
print("EVALUATION RESULT")
print("=" * 70)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

target_names = [
    str(label)
    for label in labels
]

print(
    classification_report(
        y_test,
        y_pred,
        labels=np.arange(num_classes),
        target_names=target_names,
        zero_division=0
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(num_classes)
)

print("\nRows    = Actual")
print("Columns = Predicted\n")

print("     ", end="")

for i in range(num_classes):
    print(f"{i:4d}", end="")

print()

for i in range(num_classes):

    print(f"{i:3d}: ", end="")

    for j in range(num_classes):
        print(f"{cm[i, j]:4d}", end="")

    print()

# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print("\n" + "=" * 70)
print("PER-CLASS ACCURACY")
print("=" * 70)

for class_id in range(num_classes):

    total = np.sum(y_test == class_id)

    if total == 0:
        print(
            f"{class_id:2d}: "
            f"{str(labels[class_id]):10s} -> "
            "No test samples"
        )
        continue

    correct = cm[class_id, class_id]

    class_accuracy = (
        correct / total
    ) * 100

    print(
        f"{class_id:2d}: "
        f"{str(labels[class_id]):10s} -> "
        f"{class_accuracy:6.2f}% "
        f"({correct}/{total})"
    )

# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

num_samples_to_show = min(
    20,
    len(X_test)
)

for i in range(num_samples_to_show):

    actual_id = int(y_test[i])
    predicted_id = int(y_pred[i])

    actual_label = str(
        labels[actual_id]
    )

    predicted_label = str(
        labels[predicted_id]
    )

    confidence = float(
        np.max(predictions[i])
    ) * 100

    status = (
        "CORRECT"
        if actual_id == predicted_id
        else "WRONG"
    )

    print(
        f"{i + 1:2d}. "
        f"Actual: {actual_label:10s} | "
        f"Predicted: {predicted_label:10s} | "
        f"Confidence: {confidence:6.2f}% | "
        f"{status}"
    )

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nDataset:")
print(X_PATH)

print("\nSamples:")
print(len(X))

print("\nClasses:")
print(num_classes)

print("\nInput:")
print("(30 frames, 1659 features)")

print("\nEvaluation samples:")
print(len(X_test))

print("\nAccuracy:")
print(f"{accuracy * 100:.2f}%")

print("\n" + "=" * 70)