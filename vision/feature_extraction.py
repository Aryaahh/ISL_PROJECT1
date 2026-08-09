import numpy as np


class FeatureExtractor:

    # ============================================================
    # FEATURE CONFIGURATION
    # ============================================================

    LEFT_HAND_SIZE = 21 * 3       # 63
    RIGHT_HAND_SIZE = 21 * 3      # 63
    POSE_SIZE = 33 * 3            # 99
    FACE_SIZE = 478 * 3            # 1434

    FEATURE_SIZE = (
        LEFT_HAND_SIZE
        + RIGHT_HAND_SIZE
        + POSE_SIZE
        + FACE_SIZE
    )

    def __init__(self):

        self.feature_size = self.FEATURE_SIZE

        print(
            f"FeatureExtractor initialized: "
            f"{self.feature_size} features"
        )

    # ============================================================
    # HAND LANDMARKS
    # ============================================================

    def extract_hand_landmarks(self, hand_landmarks):

        if hand_landmarks is None:

            return np.zeros(
                self.LEFT_HAND_SIZE,
                dtype=np.float32
            )

        landmarks = []

        for landmark in hand_landmarks.landmark:

            landmarks.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        result = np.asarray(
            landmarks,
            dtype=np.float32
        )

        if result.shape != (63,):

            raise ValueError(
                f"Hand feature shape error: "
                f"{result.shape}, expected (63,)"
            )

        return result

    # ============================================================
    # POSE LANDMARKS
    # ============================================================

    def extract_pose_landmarks(self, pose_landmarks):

        if pose_landmarks is None:

            return np.zeros(
                self.POSE_SIZE,
                dtype=np.float32
            )

        landmarks = []

        for landmark in pose_landmarks.landmark:

            landmarks.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        result = np.asarray(
            landmarks,
            dtype=np.float32
        )

        if result.shape != (99,):

            raise ValueError(
                f"Pose feature shape error: "
                f"{result.shape}, expected (99,)"
            )

        return result

    # ============================================================
    # FACE LANDMARKS
    # ============================================================

    def extract_face_landmarks(self, face_landmarks):

        if face_landmarks is None:

            return np.zeros(
                self.FACE_SIZE,
                dtype=np.float32
            )

        landmarks = []

        for landmark in face_landmarks.landmark:

            landmarks.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        result = np.asarray(
            landmarks,
            dtype=np.float32
        )

        if result.shape != (1434,):

            raise ValueError(
                f"Face feature shape error: "
                f"{result.shape}, expected (1434,)"
            )

        return result

    # ============================================================
    # ALL FEATURES
    #
    # IMPORTANT:
    #
    # The order MUST remain:
    #
    # LEFT HAND
    # RIGHT HAND
    # POSE
    # FACE
    #
    # This is the order used to create your training data.
    # ============================================================

    def extract_all_features(self, results):

        left_hand = self.extract_hand_landmarks(
            results.left_hand_landmarks
        )

        right_hand = self.extract_hand_landmarks(
            results.right_hand_landmarks
        )

        pose = self.extract_pose_landmarks(
            results.pose_landmarks
        )

        face = self.extract_face_landmarks(
            results.face_landmarks
        )

        features = np.concatenate([
            left_hand,
            right_hand,
            pose,
            face
        ]).astype(np.float32)

        if features.shape != (1659,):

            raise ValueError(
                "\nInvalid feature vector!\n"
                f"Actual:   {features.shape}\n"
                f"Expected: (1659,)"
            )

        return features


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("FEATURE EXTRACTION TEST")
    print("=" * 60)

    print()
    print("Left Hand :", FeatureExtractor.LEFT_HAND_SIZE)
    print("Right Hand:", FeatureExtractor.RIGHT_HAND_SIZE)
    print("Pose      :", FeatureExtractor.POSE_SIZE)
    print("Face      :", FeatureExtractor.FACE_SIZE)
    print()
    print(
        "Total     :",
        FeatureExtractor.FEATURE_SIZE
    )

    assert FeatureExtractor.FEATURE_SIZE == 1659

    print()
    print("Feature size check: PASS")
    print("=" * 60)
    