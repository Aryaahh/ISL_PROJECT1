import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tensorflow as tf
from collections import deque, Counter

from vision.landmark_detection import LandmarkDetector
from vision.sequence_builder import SequenceBuilder


# ====================================================
# Load Model
# ====================================================

MODEL_PATH = "models/isl_bilstm_model.h5"

model = tf.keras.models.load_model(MODEL_PATH)

print("=" * 60)
print("Model Loaded Successfully")
print("Model Input Shape:", model.input_shape)
print("=" * 60)


# ====================================================
# Load Labels
# ====================================================

LABEL_MAP_PATH = "data/sequences/label_map.npy"

label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()
reverse_label_map = {v: k for k, v in label_map.items()}


# ====================================================
# Constants
# ====================================================

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 1659


# ====================================================
# Initialize Components
# ====================================================

detector = LandmarkDetector()

builder = SequenceBuilder(
    sequence_length=SEQUENCE_LENGTH,
    feature_size=FEATURE_SIZE
)


# ====================================================
# Webcam
# ====================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam")
    exit()


# Prediction smoothing
recent_predictions = deque(maxlen=10)

predicted_sentence = "Waiting..."
confidence_text = ""
frame_counter = 0


print("REAL TIME SIGN LANGUAGE PREDICTION")
print("Press Q to Quit")
print("=" * 60)


# ====================================================
# Main Loop
# ====================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    frame = cv2.flip(frame, 1)

    # ------------------------------------------------
    # Detect landmarks
    # ------------------------------------------------

    results = detector.detect(frame)
    detector.draw_landmarks(frame, results)


    # ------------------------------------------------
    # Extract features
    # ------------------------------------------------

    features = detector.extract_landmarks(results)

    if features.shape[0] != FEATURE_SIZE:
        cv2.putText(
            frame,
            f"Feature Error: {features.shape[0]}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        cv2.imshow("ISL Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue


    # ------------------------------------------------
    # Build sequence
    # ------------------------------------------------

    builder.add_frame(features)


    # ------------------------------------------------
    # Predict every 5 frames after sequence is ready
    # ------------------------------------------------

    frame_counter += 1

    if builder.is_ready() and frame_counter % 5 == 0:

        sequence = builder.get_sequence()

        if sequence is not None:

            sequence = np.expand_dims(sequence, axis=0)

            prediction = model.predict(sequence, verbose=0)[0]

            class_id = int(np.argmax(prediction))
            confidence = float(prediction[class_id])

            predicted_word = reverse_label_map.get(class_id, "Unknown")

            recent_predictions.append(predicted_word)

            # Majority vote smoothing
            predicted_sentence = Counter(recent_predictions).most_common(1)[0][0]

            confidence_text = f"Confidence: {confidence * 100:.2f}%"

            print(f"Prediction: {predicted_sentence} ({confidence:.2f})")


    # ------------------------------------------------
    # UI Overlay
    # ------------------------------------------------

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Sign: {predicted_sentence}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        confidence_text,
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Frames: {len(builder.sequence)}/{SEQUENCE_LENGTH}",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ------------------------------------------------
    # Show window
    # ------------------------------------------------

    cv2.imshow("ISL Prediction", frame)


    # ------------------------------------------------
    # Quit
    # ------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ====================================================
# Cleanup
# ====================================================

print("Closing application...")

detector.close()
cap.release()
cv2.destroyAllWindows()