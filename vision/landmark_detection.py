import cv2
import mediapipe as mp


class LandmarkDetector:

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(self):

        self.mp_holistic = mp.solutions.holistic
        self.mp_draw = mp.solutions.drawing_utils

        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        print("MediaPipe Holistic initialized.")

    # ============================================================
    # DETECT
    # ============================================================

    def detect(self, frame):

        # IMPORTANT:
        #
        # Do NOT flip the frame here.
        #
        # Training data was extracted from the original
        # frames without cv2.flip().
        #

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

    # ============================================================
    # DRAW
    # ============================================================

    def draw_landmarks(
        self,
        frame,
        results
    ):

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

    # ============================================================
    # LANDMARK SUMMARY
    # ============================================================

    def print_summary(
        self,
        results
    ):

        face = (
            len(results.face_landmarks.landmark)
            if results.face_landmarks
            else 0
        )

        pose = (
            len(results.pose_landmarks.landmark)
            if results.pose_landmarks
            else 0
        )

        left = (
            len(results.left_hand_landmarks.landmark)
            if results.left_hand_landmarks
            else 0
        )

        right = (
            len(results.right_hand_landmarks.landmark)
            if results.right_hand_landmarks
            else 0
        )

        print(
            f"Face: {face} | "
            f"Pose: {pose} | "
            f"Left: {left} | "
            f"Right: {right}"
        )

    # ============================================================
    # CLOSE
    # ============================================================

    def close(self):

        self.holistic.close()


# ============================================================
# TEST CAMERA
# ============================================================

if __name__ == "__main__":

    detector = LandmarkDetector()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open camera."
        )

    print()
    print("Camera test started.")
    print("Press Q to quit.")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # --------------------------------------------------------
        # IMPORTANT:
        #
        # Process ORIGINAL frame.
        # --------------------------------------------------------

        results = detector.detect(
            frame
        )

        detector.draw_landmarks(
            frame,
            results
        )

        # --------------------------------------------------------
        # Mirror ONLY the DISPLAY.
        #
        # This does not affect landmark extraction.
        # --------------------------------------------------------

        display_frame = cv2.flip(
            frame,
            1
        )

        cv2.imshow(
            "Landmark Detection",
            display_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    detector.close()

    cap.release()

    cv2.destroyAllWindows()