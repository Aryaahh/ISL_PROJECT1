import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
import mediapipe as mp


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "isl_bilstm_model.h5"
)

LABEL_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "sequences",
    "labels.pkl"
)

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 1659

# Confidence required before accepting a prediction
CONFIDENCE_THRESHOLD = 0.70

# Number of consecutive same predictions required
# before accepting a sign.
STABLE_PREDICTIONS_REQUIRED = 3

# Prevent the same caption from being added repeatedly
DUPLICATE_COOLDOWN = 25


# ============================================================
# FEATURE EXTRACTION
# ============================================================

class FeatureExtractor:

    def __init__(self):
        pass

    # --------------------------------------------------------
    # Hand
    # 21 landmarks x 3 = 63
    # --------------------------------------------------------

    def extract_hand(self, hand_landmarks):

        if hand_landmarks:

            values = []

            for lm in hand_landmarks.landmark:

                values.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

            return np.array(
                values,
                dtype=np.float32
            )

        return np.zeros(
            63,
            dtype=np.float32
        )

    # --------------------------------------------------------
    # Pose
    # 33 landmarks x 3 = 99
    # --------------------------------------------------------

    def extract_pose(self, pose_landmarks):

        if pose_landmarks:

            values = []

            for lm in pose_landmarks.landmark:

                values.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

            return np.array(
                values,
                dtype=np.float32
            )

        return np.zeros(
            99,
            dtype=np.float32
        )

    # --------------------------------------------------------
    # Face
    #
    # MediaPipe Holistic with refine_face_landmarks=True
    # provides 478 face landmarks.
    #
    # 478 x 3 = 1434
    # --------------------------------------------------------

    def extract_face(self, face_landmarks):

        if face_landmarks:

            values = []

            for lm in face_landmarks.landmark:

                values.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

            # Make absolutely sure the size is 1434.
            values = np.array(
                values,
                dtype=np.float32
            )

            if len(values) == 1434:
                return values

            if len(values) > 1434:
                return values[:1434]

            return np.pad(
                values,
                (0, 1434 - len(values))
            )

        return np.zeros(
            1434,
            dtype=np.float32
        )

    # --------------------------------------------------------
    # Combine
    #
    # 63 + 63 + 99 + 1434 = 1659
    # --------------------------------------------------------

    def extract_all_features(self, results):

        left_hand = self.extract_hand(
            results.left_hand_landmarks
        )

        right_hand = self.extract_hand(
            results.right_hand_landmarks
        )

        pose = self.extract_pose(
            results.pose_landmarks
        )

        face = self.extract_face(
            results.face_landmarks
        )

        features = np.concatenate([
            left_hand,
            right_hand,
            pose,
            face
        ])

        # Safety check
        if features.shape[0] != FEATURE_SIZE:

            raise ValueError(
                f"Feature size mismatch. "
                f"Expected {FEATURE_SIZE}, "
                f"got {features.shape[0]}"
            )

        return features.astype(
            np.float32
        )


# ============================================================
# SEQUENCE BUFFER
# ============================================================

class SequenceBuffer:

    def __init__(
        self,
        sequence_length=SEQUENCE_LENGTH
    ):

        self.sequence_length = sequence_length
        self.frames = []

    def add(self, feature_vector):

        self.frames.append(
            feature_vector
        )

        if len(self.frames) > self.sequence_length:

            self.frames.pop(0)

    def ready(self):

        return len(self.frames) == self.sequence_length

    def get(self):

        if not self.ready():
            return None

        sequence = np.array(
            self.frames,
            dtype=np.float32
        )

        return sequence

    def reset(self):

        self.frames = []

    def length(self):

        return len(self.frames)


# ============================================================
# MEDIA PIPE
# ============================================================

class HolisticDetector:

    def __init__(self):

        self.mp_holistic = mp.solutions.holistic
        self.mp_draw = mp.solutions.drawing_utils

        self.holistic = (
            self.mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                refine_face_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        )

    def process(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb.flags.writeable = False

        results = self.holistic.process(
            rgb
        )

        rgb.flags.writeable = True

        return results

    def draw(self, frame, results):

        # Face
        if results.face_landmarks:

            self.mp_draw.draw_landmarks(
                frame,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS
            )

        # Pose
        if results.pose_landmarks:

            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS
            )

        # Left hand
        if results.left_hand_landmarks:

            self.mp_draw.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS
            )

        # Right hand
        if results.right_hand_landmarks:

            self.mp_draw.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS
            )

        return frame

    def close(self):

        self.holistic.close()


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 60)
    print("ISL LIVE SIGN LANGUAGE CAPTION SYSTEM")
    print("=" * 60)

    print("\nLoading BiLSTM model...")

    if not os.path.exists(MODEL_PATH):

        print("\nERROR: Model not found:")
        print(MODEL_PATH)

        raise SystemExit

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")

    print(
        "\nModel input shape :",
        model.input_shape
    )

    print(
        "Model output shape:",
        model.output_shape
    )

    # Verify model
    expected_input = (
        None,
        SEQUENCE_LENGTH,
        FEATURE_SIZE
    )

    if model.input_shape != expected_input:

        print("\nWARNING:")
        print(
            "Expected model input:",
            expected_input
        )

    if model.output_shape[-1] != 101:

        print("\nWARNING:")
        print(
            "Expected 101 output classes."
        )

    return model


# ============================================================
# LOAD LABELS
# ============================================================

def load_labels():

    print("\nLoading label mapping...")

    if not os.path.exists(LABEL_PATH):

        print("\nERROR: labels.pkl not found:")
        print(LABEL_PATH)

        raise SystemExit

    print(
        "Loading labels from:"
    )

    print(LABEL_PATH)

    with open(
        LABEL_PATH,
        "rb"
    ) as f:

        label_map = pickle.load(f)

    print(
        "Number of labels:",
        len(label_map)
    )

    id_to_label = {
        int(class_id): label
        for label, class_id
        in label_map.items()
    }

    return id_to_label


# ============================================================
# PREDICT
# ============================================================

def predict_sequence(
    model,
    sequence
):

    # --------------------------------------------------------
    # Sequence should be:
    #
    # (30, 1659)
    # --------------------------------------------------------

    if sequence.shape != (
        SEQUENCE_LENGTH,
        FEATURE_SIZE
    ):

        raise ValueError(
            "Invalid sequence shape: "
            f"{sequence.shape}"
        )

    # Add batch dimension
    #
    # (30,1659)
    #
    # becomes
    #
    # (1,30,1659)

    input_data = np.expand_dims(
        sequence,
        axis=0
    ).astype(np.float32)

    # --------------------------------------------------------
    # Use direct model call instead of model.predict()
    #
    # This avoids some of the overhead that caused your
    # previous KeyboardInterrupt during repeated prediction.
    # --------------------------------------------------------

    predictions = model(
        input_data,
        training=False
    ).numpy()

    probabilities = predictions[0]

    predicted_id = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_id]
    )

    return predicted_id, confidence


# ============================================================
# DRAW TEXT
# ============================================================

def draw_text(
    frame,
    caption,
    confidence,
    buffer_length,
    stable_count
):

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # Caption background
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (0, height - 150),
        (width, height),
        (0, 0, 0),
        -1
    )

    # --------------------------------------------------------
    # Caption
    # --------------------------------------------------------

    display_caption = caption

    if len(display_caption) > 70:

        display_caption = (
            "..." +
            display_caption[-67:]
        )

    cv2.putText(
        frame,
        "CAPTION:",
        (20, height - 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        display_caption,
        (20, height - 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    status = (
        f"Frames: {buffer_length}/30   "
        f"Confidence: {confidence:.2f}   "
        f"Stable: {stable_count}/"
        f"{STABLE_PREDICTIONS_REQUIRED}"
    )

    cv2.putText(
        frame,
        status,
        (20, height - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


# ============================================================
# MAIN
# ============================================================

def main():

    model = load_model()

    id_to_label = load_labels()

    print(
        "\nStarting camera and MediaPipe..."
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print(
            "\nERROR: Camera could not be opened."
        )

        return

    print(
        "\nCamera started successfully."
    )

    print("\nInstructions:")
    print(
        "Perform signs continuously."
    )
    print(
        "Wait for the caption to appear."
    )
    print(
        "Press C to clear the caption."
    )
    print(
        "Press R to reset the 30-frame sequence."
    )
    print(
        "Press Q to quit."
    )

    detector = HolisticDetector()
    extractor = FeatureExtractor()
    buffer = SequenceBuffer()

    # --------------------------------------------------------
    # Caption state
    # --------------------------------------------------------

    caption_words = []

    last_prediction_id = None
    stable_count = 0

    last_accepted_id = None
    cooldown_counter = 0

    last_confidence = 0.0

    prediction_count = 0

    # --------------------------------------------------------
    # Main camera loop
    # --------------------------------------------------------

    try:

        while True:

            success, frame = camera.read()

            if not success:

                print(
                    "Could not read frame."
                )

                break

            # Mirror camera
            frame = cv2.flip(
                frame,
                1
            )

            # ------------------------------------------------
            # MediaPipe
            # ------------------------------------------------

            results = detector.process(
                frame
            )

            # ------------------------------------------------
            # Features
            # ------------------------------------------------

            try:

                feature_vector = (
                    extractor.extract_all_features(
                        results
                    )
                )

            except ValueError as e:

                print(
                    "\nFeature error:",
                    e
                )

                buffer.reset()

                continue

            # ------------------------------------------------
            # Add frame
            # ------------------------------------------------

            buffer.add(
                feature_vector
            )

            # ------------------------------------------------
            # Prediction
            #
            # IMPORTANT:
            # We only predict when 30 frames are available.
            # ------------------------------------------------

            if buffer.ready():

                sequence = buffer.get()

                predicted_id, confidence = (
                    predict_sequence(
                        model,
                        sequence
                    )
                )

                last_confidence = confidence

                prediction_count += 1

                # --------------------------------------------
                # Confidence filter
                # --------------------------------------------

                if confidence >= CONFIDENCE_THRESHOLD:

                    # ----------------------------------------
                    # Check whether prediction is same as
                    # previous prediction.
                    # ----------------------------------------

                    if predicted_id == last_prediction_id:

                        stable_count += 1

                    else:

                        last_prediction_id = predicted_id
                        stable_count = 1

                    # ----------------------------------------
                    # Accept only stable prediction
                    # ----------------------------------------

                    if (
                        stable_count
                        >= STABLE_PREDICTIONS_REQUIRED
                    ):

                        if cooldown_counter > 0:

                            cooldown_counter -= 1

                        else:

                            # --------------------------------
                            # Avoid immediate duplicate
                            # --------------------------------

                            if (
                                predicted_id
                                != last_accepted_id
                            ):

                                label = id_to_label.get(
                                    predicted_id,
                                    "UNKNOWN"
                                )

                                caption_words.append(
                                    label
                                )

                                last_accepted_id = (
                                    predicted_id
                                )

                                cooldown_counter = (
                                    DUPLICATE_COOLDOWN
                                )

                                print(
                                    "\nRecognized:",
                                    label
                                )

                                print(
                                    "Confidence:",
                                    f"{confidence:.3f}"
                                )

                        # Reset buffer after accepted sign
                        #
                        # This separates one sign from
                        # the next sign.

                        buffer.reset()

                        stable_count = 0
                        last_prediction_id = None

            # ------------------------------------------------
            # Caption
            # ------------------------------------------------

            caption = " ".join(
                caption_words
            )

            if not caption:

                caption = (
                    "Waiting for signs..."
                )

            # ------------------------------------------------
            # Draw landmarks
            # ------------------------------------------------

            detector.draw(
                frame,
                results
            )

            # ------------------------------------------------
            # Draw caption
            # ------------------------------------------------

            draw_text(
                frame,
                caption,
                last_confidence,
                buffer.length(),
                stable_count
            )

            # ------------------------------------------------
            # Controls
            # ------------------------------------------------

            cv2.putText(
                frame,
                "C: Clear    R: Reset    Q: Quit",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "ISL Live Caption",
                frame
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # ------------------------------------------------
            # Clear caption
            # ------------------------------------------------

            if key == ord("c"):

                caption_words = []

                last_accepted_id = None
                cooldown_counter = 0

                print(
                    "\nCaption cleared."
                )

            # ------------------------------------------------
            # Reset sequence
            # ------------------------------------------------

            elif key == ord("r"):

                buffer.reset()

                stable_count = 0
                last_prediction_id = None

                print(
                    "\nSequence reset."
                )

            # ------------------------------------------------
            # Quit
            # ------------------------------------------------

            elif key == ord("q"):

                print(
                    "\nStopping system..."
                )

                break

    finally:

        detector.close()

        camera.release()

        cv2.destroyAllWindows()

    print(
        "\nSystem stopped."
    )

    print(
        "Predictions generated:",
        prediction_count
    )

    print(
        "\nFinal caption:"
    )

    print(
        " ".join(caption_words)
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
    