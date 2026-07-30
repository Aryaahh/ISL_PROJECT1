import numpy as np


class FeatureExtractor:

    def __init__(self):
        pass

    # ----------------------------------------------------
    # Left / Right Hand (21 landmarks × 3 = 63)
    # ----------------------------------------------------
    def extract_hand_landmarks(self, hand_landmarks):

        if hand_landmarks:
            landmarks = []

            for landmark in hand_landmarks.landmark:
                landmarks.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            return np.array(landmarks, dtype=np.float32)

        return np.zeros(63, dtype=np.float32)

    # ----------------------------------------------------
    # Pose (33 landmarks × 3 = 99)
    # ----------------------------------------------------
    def extract_pose_landmarks(self, pose_landmarks):

        if pose_landmarks:
            landmarks = []

            for landmark in pose_landmarks.landmark:
                landmarks.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            return np.array(landmarks, dtype=np.float32)

        return np.zeros(99, dtype=np.float32)

    # ----------------------------------------------------
    # Face (468 landmarks × 3 = 1404)
    # ----------------------------------------------------
    def extract_face_landmarks(self, face_landmarks):

        if face_landmarks:
            landmarks = []

            for landmark in face_landmarks.landmark:
                landmarks.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            return np.array(landmarks, dtype=np.float32)

        return np.zeros(1404, dtype=np.float32)

    # ----------------------------------------------------
    # Combine All Features
    # ----------------------------------------------------
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
        ])

        return features


# ----------------------------------------------------
# Test
# ----------------------------------------------------
if __name__ == "__main__":

    print("=" * 40)
    print("Feature Extraction Module Ready")
    print("=" * 40)

    print("Expected Feature Dimensions")
    print("-" * 40)

    print("Left Hand  : 63")
    print("Right Hand : 63")
    print("Pose       : 99")
    print("Face       : 1404")

    print("-" * 40)
    print("Total      : 1629")
    print("=" * 40)
    