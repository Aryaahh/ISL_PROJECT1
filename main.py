import os
import cv2
import numpy as np

from vision.camera import Camera
from vision.landmark_detection import LandmarkDetector
from vision.feature_extraction import FeatureExtractor
from vision.sequence_builder import SequenceBuilder


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 1659

# Change this to the sign/sentence you are collecting
LABEL = "hello"

# Folder expected by build_sequences.py
SAVE_ROOT = os.path.join(
    "data",
    "processed_landmarks"
)


# ============================================================
# FIND NEXT SAMPLE NUMBER
# ============================================================

def get_next_sample_number(label):
    """
    Find the next available sample number.

    Example:
        sample_1
        sample_2
        sample_3

    If none exist, returns 1.
    """

    label_path = os.path.join(
        SAVE_ROOT,
        label
    )

    os.makedirs(
        label_path,
        exist_ok=True
    )

    sample_numbers = []

    for name in os.listdir(label_path):

        sample_path = os.path.join(
            label_path,
            name
        )

        if not os.path.isdir(sample_path):
            continue

        if name.startswith("sample_"):

            try:
                number = int(
                    name.replace(
                        "sample_",
                        ""
                    )
                )

                sample_numbers.append(number)

            except ValueError:
                pass

    if not sample_numbers:
        return 1

    return max(sample_numbers) + 1


# ============================================================
# SAVE SEQUENCE
# ============================================================

def save_sequence(sequence, label, sample_number):
    """
    Save a 30-frame sequence in the format expected by
    vision/build_sequences.py.

    Output:

    data/
        processed_landmarks/
            label/
                sample_1/
                    frame_001.npy
                    frame_002.npy
                    ...
                    frame_030.npy
    """

    sample_path = os.path.join(
        SAVE_ROOT,
        label,
        f"sample_{sample_number}"
    )

    os.makedirs(
        sample_path,
        exist_ok=True
    )

    print("\nSaving sequence...")
    print("Label  :", label)
    print("Sample :", sample_number)
    print("Shape  :", sequence.shape)

    for frame_number in range(SEQUENCE_LENGTH):

        frame_path = os.path.join(
            sample_path,
            f"frame_{frame_number + 1:03d}.npy"
        )

        np.save(
            frame_path,
            sequence[frame_number].astype(
                np.float32
            )
        )

    print("\nSequence saved successfully.")
    print("Location:")
    print(sample_path)

    print(
        f"Saved {SEQUENCE_LENGTH} frames."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ISL DATA COLLECTION")
    print("=" * 60)

    print("\nCollecting gesture:", LABEL)

    print("\nInstructions:")
    print("1. Perform the sign")
    print("2. Wait until 30 frames are collected")
    print("3. Press 's' to save the sequence")
    print("4. Press 'q' to exit")

    print("\nFeature size:", FEATURE_SIZE)
    print("Sequence length:", SEQUENCE_LENGTH)

    # --------------------------------------------------------
    # Initialize modules
    # --------------------------------------------------------

    camera = Camera()

    landmark_detector = LandmarkDetector()

    feature_extractor = FeatureExtractor()

    sequence_builder = SequenceBuilder(
        sequence_length=SEQUENCE_LENGTH,
        feature_size=FEATURE_SIZE
    )

    # --------------------------------------------------------
    # Find next sample number
    # --------------------------------------------------------

    sample_number = get_next_sample_number(
        LABEL
    )

    print(
        "\nNext sample number:",
        sample_number
    )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    cap = camera.cap

    if cap is None or not cap.isOpened():

        print("\nERROR: Camera could not be opened.")

        try:
            landmark_detector.close()
        except Exception:
            pass

        try:
            camera.stop()
        except Exception:
            pass

        return

    print("\nCamera started successfully.")
    print("Press 'q' to quit.")

    # ========================================================
    # CAMERA LOOP
    # ========================================================

    try:

        while True:

            # ------------------------------------------------
            # Read frame
            # ------------------------------------------------

            ret, frame = cap.read()

            if not ret:

                print(
                    "\nERROR: Could not read frame."
                )

                break

            # ------------------------------------------------
            # Mirror camera
            # ------------------------------------------------

            frame = cv2.flip(
                frame,
                1
            )

            # ------------------------------------------------
            # Detect landmarks
            #
            # IMPORTANT:
            # detect() returns only results.
            # ------------------------------------------------

            results = landmark_detector.detect(
                frame
            )

            # ------------------------------------------------
            # Extract all 1659 features
            #
            # FeatureExtractor:
            #
            # Left hand  = 63
            # Right hand = 63
            # Pose       = 99
            # Face       = 1434
            #
            # Total      = 1659
            # ------------------------------------------------

            feature_vector = (
                feature_extractor.extract_all_features(
                    results
                )
            )

            # ------------------------------------------------
            # Verify feature size
            # ------------------------------------------------

            if feature_vector.shape != (
                FEATURE_SIZE,
            ):

                print(
                    "\nERROR: Incorrect feature size!"
                )

                print(
                    "Expected:",
                    FEATURE_SIZE
                )

                print(
                    "Received:",
                    feature_vector.shape
                )

                break

            # ------------------------------------------------
            # Add frame to sequence
            # ------------------------------------------------

            sequence_builder.add_frame(
                feature_vector
            )

            # ------------------------------------------------
            # Draw landmarks
            # ------------------------------------------------

            display_frame = (
                landmark_detector.draw_landmarks(
                    frame,
                    results
                )
            )

            # ------------------------------------------------
            # Display information
            # ------------------------------------------------

            current_frames = len(
                sequence_builder.sequence
            )

            cv2.putText(
                display_frame,
                f"Frames: {current_frames}/{SEQUENCE_LENGTH}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                f"Label: {LABEL}",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # Sequence ready
            # ------------------------------------------------

            if sequence_builder.is_ready():

                cv2.putText(
                    display_frame,
                    "SEQUENCE READY - PRESS S",
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    display_frame,
                    "Perform sign...",
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

            # ------------------------------------------------
            # Show camera
            # ------------------------------------------------

            cv2.imshow(
                "ISL Data Collection",
                display_frame
            )

            # ------------------------------------------------
            # Keyboard
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            # =================================================
            # SAVE
            # =================================================

            if key == ord("s"):

                if sequence_builder.is_ready():

                    sequence = (
                        sequence_builder.get_sequence()
                    )

                    # -----------------------------------------
                    # Final validation
                    # -----------------------------------------

                    if sequence is None:

                        print(
                            "\nERROR: Sequence is None."
                        )

                        continue

                    if sequence.shape != (
                        SEQUENCE_LENGTH,
                        FEATURE_SIZE
                    ):

                        print(
                            "\nERROR: Invalid sequence shape."
                        )

                        print(
                            "Expected:",
                            (
                                SEQUENCE_LENGTH,
                                FEATURE_SIZE
                            )
                        )

                        print(
                            "Received:",
                            sequence.shape
                        )

                        continue

                    # -----------------------------------------
                    # Save
                    # -----------------------------------------

                    save_sequence(
                        sequence,
                        LABEL,
                        sample_number
                    )

                    # -----------------------------------------
                    # Prepare for next sample
                    # -----------------------------------------

                    sample_number += 1

                    sequence_builder.reset()

                    print(
                        "\nReady for next sample."
                    )

                else:

                    print(
                        "\nCollect 30 frames first."
                    )

            # =================================================
            # QUIT
            # =================================================

            elif key == ord("q"):

                print(
                    "\nExiting data collection..."
                )

                break

    finally:

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        try:
            landmark_detector.close()
        except Exception:
            pass

        try:
            camera.stop()
        except Exception:
            pass

        cv2.destroyAllWindows()

        print(
            "\nCamera and resources released."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
    