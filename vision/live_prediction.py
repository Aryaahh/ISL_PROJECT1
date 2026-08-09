import os
import sys
from collections import deque, Counter

import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

VISION_DIR = os.path.join(
    BASE_DIR,
    "vision"
)

if VISION_DIR not in sys.path:

    sys.path.insert(
        0,
        VISION_DIR
    )


from landmark_detection import LandmarkDetector
from feature_extraction import FeatureExtractor


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "isl_bilstm_model.keras"
)

LABEL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "training",
    "labels.npy"
)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 30

FEATURE_SIZE = 1659

NUMBER_OF_CLASSES = 21

CAMERA_INDEX = 0

# Only show a prediction above this confidence.
CONFIDENCE_THRESHOLD = 0.70

# Number of recent predictions used for stabilization.
VOTE_HISTORY_SIZE = 5


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("ISL BiLSTM LIVE PREDICTION")
print("=" * 70)


# ============================================================
# LOAD LABELS
# ============================================================

if not os.path.exists(LABEL_PATH):

    raise FileNotFoundError(
        f"Labels not found:\n{LABEL_PATH}"
    )


LABELS = np.load(
    LABEL_PATH,
    allow_pickle=True
)

LABELS = [
    str(label)
    for label in LABELS
]


print()
print("Labels:")

for index, label in enumerate(LABELS):

    print(
        f"{index}: {label}"
    )

print()
print(
    "Number of classes:",
    len(LABELS)
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("-" * 70)
print("LOADING MODEL")
print("-" * 70)


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


model = tf.keras.models.load_model(
    MODEL_PATH
)


print(
    "Model loaded successfully."
)

print(
    "Model input shape :",
    model.input_shape
)

print(
    "Model output shape:",
    model.output_shape
)


# ============================================================
# COMPATIBILITY CHECK
# ============================================================

if model.input_shape != (
    None,
    SEQUENCE_LENGTH,
    FEATURE_SIZE
):

    raise ValueError(
        "\nMODEL INPUT MISMATCH\n"
        f"Model input: {model.input_shape}\n"
        f"Expected:    "
        f"(None, {SEQUENCE_LENGTH}, {FEATURE_SIZE})"
    )


if model.output_shape[-1] != NUMBER_OF_CLASSES:

    raise ValueError(
        "\nMODEL CLASS COUNT MISMATCH\n"
        f"Model classes: {model.output_shape[-1]}\n"
        f"Expected:      {NUMBER_OF_CLASSES}"
    )


if len(LABELS) != NUMBER_OF_CLASSES:

    raise ValueError(
        "\nLABEL COUNT MISMATCH\n"
        f"Labels:   {len(LABELS)}\n"
        f"Expected: {NUMBER_OF_CLASSES}"
    )


print()
print("Model compatibility check: PASS")


# ============================================================
# FEATURE EXTRACTOR
# ============================================================

print()
print("-" * 70)
print("FEATURE EXTRACTOR")
print("-" * 70)


extractor = FeatureExtractor()


if extractor.feature_size != FEATURE_SIZE:

    raise ValueError(
        f"Feature extractor returned "
        f"{extractor.feature_size}, "
        f"expected {FEATURE_SIZE}"
    )


print(
    "Feature size:",
    extractor.feature_size
)

print(
    "Feature compatibility check: PASS"
)


# ============================================================
# LANDMARK DETECTOR
# ============================================================

print()
print("-" * 70)
print("INITIALIZING MEDIAPIPE")
print("-" * 70)


detector = LandmarkDetector()


# ============================================================
# CAMERA
# ============================================================

print()
print("-" * 70)
print("STARTING CAMERA")
print("-" * 70)


cap = cv2.VideoCapture(
    CAMERA_INDEX
)


if not cap.isOpened():

    detector.close()

    raise RuntimeError(
        "Cannot open camera."
    )


print("Camera started.")

print()
print("Controls:")
print("  Q = Quit")
print("  C = Clear sequence")
print("  P = Predict current 30 frames")
print()


# ============================================================
# SEQUENCE
# ============================================================

sequence = deque(
    maxlen=SEQUENCE_LENGTH
)


# ============================================================
# PREDICTION STATE
# ============================================================

last_prediction = "Waiting..."

last_confidence = 0.0

prediction_history = deque(
    maxlen=VOTE_HISTORY_SIZE
)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "Cannot read camera frame."
        )

        break


    # ========================================================
    # IMPORTANT:
    #
    # DO NOT FLIP BEFORE MEDIAPIPE.
    #
    # Training pipeline:
    #
    # cv2.imread()
    #       ↓
    # MediaPipe
    #       ↓
    # FeatureExtractor
    #
    # Live pipeline must be:
    #
    # camera frame
    #       ↓
    # MediaPipe
    #       ↓
    # FeatureExtractor
    #
    # ========================================================

    original_frame = frame.copy()


    # ========================================================
    # MEDIAPIPE
    # ========================================================

    results = detector.detect(
        original_frame
    )


    # ========================================================
    # EXTRACT 1659 FEATURES
    # ========================================================

    try:

        features = extractor.extract_all_features(
            results
        )

    except Exception as error:

        print(
            "Feature extraction error:",
            error
        )

        display_frame = cv2.flip(
            original_frame,
            1
        )

        cv2.putText(
            display_frame,
            "Feature extraction error",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.imshow(
            "ISL BiLSTM Live Prediction",
            display_frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

        continue


    # ========================================================
    # CHECK FEATURE VECTOR
    # ========================================================

    if features.shape != (
        FEATURE_SIZE,
    ):

        print(
            "ERROR:",
            features.shape
        )

        continue


    # ========================================================
    # ADD TO SEQUENCE
    # ========================================================

    sequence.append(
        features
    )


    # ========================================================
    # DISPLAY FRAME
    #
    # Landmarks are drawn on original frame.
    # Then frame is mirrored ONLY for display.
    # ========================================================

    detector.draw_landmarks(
        original_frame,
        results
    )


    display_frame = cv2.flip(
        original_frame,
        1
    )


    # ========================================================
    # MANUAL PREDICTION
    #
    # Prediction happens ONLY when 30 frames exist.
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # ========================================================
    # P = PREDICT
    # ========================================================

    if key == ord("p"):

        if len(sequence) < SEQUENCE_LENGTH:

            print()
            print(
                f"Need "
                f"{SEQUENCE_LENGTH - len(sequence)} "
                f"more frames."
            )

            last_prediction = "Waiting..."

            last_confidence = 0.0

            prediction_history.clear()

        else:

            # ------------------------------------------------
            # Create input
            # ------------------------------------------------

            input_sequence = np.asarray(
                sequence,
                dtype=np.float32
            )

            input_sequence = np.expand_dims(
                input_sequence,
                axis=0
            )


            # ------------------------------------------------
            # Verify shape
            # ------------------------------------------------

            if input_sequence.shape != (
                1,
                SEQUENCE_LENGTH,
                FEATURE_SIZE
            ):

                raise ValueError(
                    f"Invalid model input shape: "
                    f"{input_sequence.shape}"
                )


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            probabilities = model.predict(
                input_sequence,
                verbose=0
            )[0]


            # ------------------------------------------------
            # Best class
            # ------------------------------------------------

            predicted_class = int(
                np.argmax(
                    probabilities
                )
            )


            confidence = float(
                probabilities[
                    predicted_class
                ]
            )


            predicted_label = LABELS[
                predicted_class
            ]


            print()
            print("-" * 60)

            print(
                "RAW PREDICTION"
            )

            print(
                "Class:",
                predicted_class
            )

            print(
                "Label:",
                predicted_label
            )

            print(
                "Confidence:",
                f"{confidence * 100:.2f}%"
            )


            # ------------------------------------------------
            # Top 5 predictions
            # ------------------------------------------------

            top_indices = np.argsort(
                probabilities
            )[-5:][::-1]


            print()
            print("Top 5:")

            for rank, index in enumerate(
                top_indices,
                start=1
            ):

                print(
                    f"{rank}. "
                    f"{LABELS[index]:10s} "
                    f"{probabilities[index] * 100:.2f}%"
                )


            print("-" * 60)


            # ------------------------------------------------
            # Confidence check
            # ------------------------------------------------

            if confidence >= CONFIDENCE_THRESHOLD:

                prediction_history.append(
                    predicted_label
                )


                # --------------------------------------------
                # Majority vote
                # --------------------------------------------

                counts = Counter(
                    prediction_history
                )


                stable_label, stable_count = (
                    counts.most_common(1)[0]
                )


                last_prediction = stable_label

                last_confidence = confidence


            else:

                last_prediction = "Uncertain"

                last_confidence = confidence


    # ========================================================
    # C = CLEAR
    # ========================================================

    if key == ord("c"):

        sequence.clear()

        prediction_history.clear()

        last_prediction = "Waiting..."

        last_confidence = 0.0

        print()
        print(
            "Sequence cleared."
        )


    # ========================================================
    # Q = QUIT
    # ========================================================

    if key == ord("q"):

        break


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.rectangle(
        display_frame,
        (0, 0),
        (700, 120),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        display_frame,
        f"Prediction: {last_prediction}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )


    cv2.putText(
        display_frame,
        f"Confidence: "
        f"{last_confidence * 100:.2f}%",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )


    cv2.putText(
        display_frame,
        f"Sequence: "
        f"{len(sequence)}/{SEQUENCE_LENGTH}",
        (20, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "ISL BiLSTM Live Prediction",
        display_frame
    )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

detector.close()


print()
print("=" * 70)
print("LIVE PREDICTION STOPPED")
print("=" * 70)