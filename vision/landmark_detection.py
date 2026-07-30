import cv2
import mediapipe as mp
import numpy as np


class LandmarkDetector:

    def __init__(self):

        # MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic

        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_draw = mp.solutions.drawing_utils

    # ----------------------------------------
    # Detect Landmarks
    # ----------------------------------------
    def detect(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rgb.flags.writeable = False

        results = self.holistic.process(rgb)

        rgb.flags.writeable = True

        return results

    # ----------------------------------------
    # Draw Landmarks
    # ----------------------------------------
    def draw_landmarks(self, frame, results):

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

        # Left Hand
        if results.left_hand_landmarks:

            self.mp_draw.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS
            )

        # Right Hand
        if results.right_hand_landmarks:

            self.mp_draw.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS
            )

        return frame

    # ----------------------------------------
    # Extract Numerical Features
    # ----------------------------------------
    def extract_landmarks(self, results):

        landmarks = []

        # ==============================
        # Face (468 × 3 = 1404)
        # ==============================

        if results.face_landmarks:

            for lm in results.face_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

        else:
            landmarks.extend([0] * (468 * 3))

        # ==============================
        # Pose (33 × 4 = 132)
        # x, y, z, visibility
        # ==============================

        if results.pose_landmarks:

            for lm in results.pose_landmarks.landmark:
                landmarks.extend(
                    [lm.x, lm.y, lm.z, lm.visibility]
                )

        else:
            landmarks.extend([0] * (33 * 4))

        # ==============================
        # Left Hand (21 × 3 = 63)
        # ==============================

        if results.left_hand_landmarks:

            for lm in results.left_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

        else:
            landmarks.extend([0] * (21 * 3))

        # ==============================
        # Right Hand (21 × 3 = 63)
        # ==============================

        if results.right_hand_landmarks:

            for lm in results.right_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

        else:
            landmarks.extend([0] * (21 * 3))

        return np.array(landmarks, dtype=np.float32)

    # ----------------------------------------
    # Print Landmark Information
    # ----------------------------------------
    def print_summary(self, results):

        face = len(results.face_landmarks.landmark) if results.face_landmarks else 0
        pose = len(results.pose_landmarks.landmark) if results.pose_landmarks else 0
        left = len(results.left_hand_landmarks.landmark) if results.left_hand_landmarks else 0
        right = len(results.right_hand_landmarks.landmark) if results.right_hand_landmarks else 0

        total = (
            face * 3 +
            pose * 4 +
            left * 3 +
            right * 3
        )

        print(
            f"Face:{face} | "
            f"Pose:{pose} | "
            f"Left:{left} | "
            f"Right:{right} | "
            f"Features:{total}"
        )

    # ----------------------------------------
    # Release Resources
    # ----------------------------------------
    def close(self):

        self.holistic.close()


# ==========================================
# Testing
# ==========================================

if __name__ == "__main__":

    detector = LandmarkDetector()

    cap = cv2.VideoCapture(0)

    print("Press 'Q' to Quit")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        results = detector.detect(frame)

        detector.draw_landmarks(frame, results)

        detector.print_summary(results)

        cv2.imshow("Landmark Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    