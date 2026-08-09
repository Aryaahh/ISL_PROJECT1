import os
import sys
import cv2
import numpy as np

# ============================================================
# MAKE PROJECT ROOT IMPORTABLE
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from vision.landmark_detection import LandmarkDetector
from vision.feature_extraction import FeatureExtractor


# ============================================================
# SETTINGS
# ============================================================

RAW_VIDEO_FOLDER = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw_videos"
)

OUTPUT_FOLDER = os.path.join(
    PROJECT_ROOT,
    "data",
    "word_processed_landmarks"
)

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 1659

VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# FIND NEXT SAMPLE NUMBER
# ============================================================

def get_next_sample_number(label):
    """
    Find the next available sample number for a label.

    Example:

        sample_1
        sample_2
        sample_3

    If no samples exist, returns 1.
    """

    label_folder = os.path.join(
        OUTPUT_FOLDER,
        label
    )

    os.makedirs(
        label_folder,
        exist_ok=True
    )

    sample_numbers = []

    for name in os.listdir(label_folder):

        if not name.startswith("sample_"):
            continue

        sample_path = os.path.join(
            label_folder,
            name
        )

        if not os.path.isdir(sample_path):
            continue

        try:

            number = int(
                name.replace(
                    "sample_",
                    ""
                )
            )

            sample_numbers.append(number)

        except ValueError:

            continue

    if not sample_numbers:
        return 1

    return max(sample_numbers) + 1


# ============================================================
# CHECK WHETHER VIDEO WAS ALREADY PROCESSED
# ============================================================

def already_processed(label, video_name):
    """
    Check whether this video has already been processed.

    Each processed sample contains a file called:

        source_video.txt

    containing the original video filename.
    """

    label_folder = os.path.join(
        OUTPUT_FOLDER,
        label
    )

    if not os.path.exists(label_folder):
        return False

    for sample_name in os.listdir(label_folder):

        sample_folder = os.path.join(
            label_folder,
            sample_name
        )

        if not os.path.isdir(sample_folder):
            continue

        source_file = os.path.join(
            sample_folder,
            "source_video.txt"
        )

        if not os.path.exists(source_file):
            continue

        try:

            with open(
                source_file,
                "r",
                encoding="utf-8"
            ) as f:

                saved_video_name = f.read().strip()

            if saved_video_name == video_name:
                return True

        except Exception:
            pass

    return False


# ============================================================
# SAVE PROCESSED SAMPLE
# ============================================================

def save_sequence(
    sequence,
    label,
    sample_number,
    video_name
):
    """
    Save a 30-frame sequence.

    Structure:

    data/
        word_processed_landmarks/
            label/
                sample_1/
                    frame_001.npy
                    frame_002.npy
                    ...
                    frame_030.npy
                    source_video.txt
    """

    if sequence.shape != (
        SEQUENCE_LENGTH,
        FEATURE_SIZE
    ):

        raise ValueError(
            "\nInvalid sequence shape.\n"
            f"Expected: "
            f"({SEQUENCE_LENGTH}, {FEATURE_SIZE})\n"
            f"Received: {sequence.shape}"
        )

    label_folder = os.path.join(
        OUTPUT_FOLDER,
        label
    )

    os.makedirs(
        label_folder,
        exist_ok=True
    )

    sample_folder = os.path.join(
        label_folder,
        f"sample_{sample_number}"
    )

    os.makedirs(
        sample_folder,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save frames
    # --------------------------------------------------------

    for frame_number in range(
        SEQUENCE_LENGTH
    ):

        frame_path = os.path.join(
            sample_folder,
            f"frame_{frame_number + 1:03d}.npy"
        )

        np.save(
            frame_path,
            sequence[frame_number].astype(
                np.float32
            )
        )

    # --------------------------------------------------------
    # Remember source video
    # --------------------------------------------------------

    source_file = os.path.join(
        sample_folder,
        "source_video.txt"
    )

    with open(
        source_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(video_name)

    print(
        f"      Saved sample_{sample_number}"
    )


# ============================================================
# EXTRACT 30 FRAMES FROM VIDEO
# ============================================================

def extract_sequence_from_video(
    video_path,
    landmark_detector,
    feature_extractor
):
    """
    Read a video and extract exactly 30 frames.

    Frames are selected evenly across the entire video.

    Returns:

        numpy array with shape
        (30, 1659)

    or None if the video cannot be processed.
    """

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        print(
            "      ERROR: Could not open video."
        )

        return None

    # --------------------------------------------------------
    # Read all frames
    # --------------------------------------------------------

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    # --------------------------------------------------------
    # Check number of frames
    # --------------------------------------------------------

    total_frames = len(frames)

    if total_frames < SEQUENCE_LENGTH:

        print(
            f"      SKIP: only "
            f"{total_frames} frames "
            f"(need at least {SEQUENCE_LENGTH})"
        )

        return None

    # --------------------------------------------------------
    # Select exactly 30 frames
    #
    # Instead of taking only the first 30 frames,
    # distribute 30 frames throughout the video.
    # --------------------------------------------------------

    selected_indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH,
        dtype=int
    )

    sequence = []

    # --------------------------------------------------------
    # Process selected frames
    # --------------------------------------------------------

    for frame_index in selected_indices:

        frame = frames[frame_index]

        # Convert BGR -> RGB is handled inside
        # LandmarkDetector.detect()

        results = landmark_detector.detect(
            frame
        )

        feature_vector = (
            feature_extractor.extract_all_features(
                results
            )
        )

        # ----------------------------------------------------
        # Validate feature vector
        # ----------------------------------------------------

        if feature_vector.shape != (
            FEATURE_SIZE,
        ):

            print(
                "\n      ERROR: Invalid feature size."
            )

            print(
                "      Expected:",
                FEATURE_SIZE
            )

            print(
                "      Received:",
                feature_vector.shape
            )

            return None

        sequence.append(
            feature_vector
        )

    # --------------------------------------------------------
    # Convert to NumPy array
    # --------------------------------------------------------

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if sequence.shape != (
        SEQUENCE_LENGTH,
        FEATURE_SIZE
    ):

        print(
            "\n      ERROR: Invalid final sequence shape."
        )

        print(
            "      Expected:",
            (
                SEQUENCE_LENGTH,
                FEATURE_SIZE
            )
        )

        print(
            "      Received:",
            sequence.shape
        )

        return None

    return sequence


# ============================================================
# PROCESS ONE LABEL
# ============================================================

def process_label(
    label,
    landmark_detector,
    feature_extractor
):

    label_folder = os.path.join(
        RAW_VIDEO_FOLDER,
        label
    )

    if not os.path.isdir(label_folder):

        return 0, 0

    video_files = []

    for filename in sorted(
        os.listdir(label_folder)
    ):

        if filename.lower().endswith(
            VIDEO_EXTENSIONS
        ):

            video_files.append(
                filename
            )

    if not video_files:

        return 0, 0

    print("\n" + "-" * 70)
    print(
        f"Label: {label}"
    )

    print(
        f"Videos found: {len(video_files)}"
    )

    processed_count = 0
    skipped_count = 0

    # --------------------------------------------------------
    # Find next sample number
    # --------------------------------------------------------

    sample_number = get_next_sample_number(
        label
    )

    # --------------------------------------------------------
    # Process videos
    # --------------------------------------------------------

    for video_name in video_files:

        video_path = os.path.join(
            label_folder,
            video_name
        )

        print(
            f"\n   Video: {video_name}"
        )

        # ----------------------------------------------------
        # Skip already processed videos
        # ----------------------------------------------------

        if already_processed(
            label,
            video_name
        ):

            print(
                "      Already processed - SKIP"
            )

            skipped_count += 1

            continue

        # ----------------------------------------------------
        # Process video
        # ----------------------------------------------------

        sequence = extract_sequence_from_video(
            video_path,
            landmark_detector,
            feature_extractor
        )

        if sequence is None:

            skipped_count += 1

            continue

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_sequence(
            sequence,
            label,
            sample_number,
            video_name
        )

        print(
            f"      Shape: {sequence.shape}"
        )

        processed_count += 1

        sample_number += 1

    return processed_count, skipped_count


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ISL WORD VIDEO PROCESSING")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nInput folder:")
    print(RAW_VIDEO_FOLDER)

    print("\nOutput folder:")
    print(OUTPUT_FOLDER)

    print("\nSequence length:")
    print(SEQUENCE_LENGTH)

    print("\nFeature size:")
    print(FEATURE_SIZE)

    # --------------------------------------------------------
    # Check raw video folder
    # --------------------------------------------------------

    if not os.path.exists(
        RAW_VIDEO_FOLDER
    ):

        print(
            "\nERROR: raw_videos folder was not found!"
        )

        print(
            RAW_VIDEO_FOLDER
        )

        return

    # --------------------------------------------------------
    # Find labels
    # --------------------------------------------------------

    labels = []

    for name in sorted(
        os.listdir(RAW_VIDEO_FOLDER)
    ):

        folder_path = os.path.join(
            RAW_VIDEO_FOLDER,
            name
        )

        if os.path.isdir(folder_path):

            labels.append(name)

    if not labels:

        print(
            "\nERROR: No label folders found."
        )

        return

    print(
        f"\nNumber of word classes: {len(labels)}"
    )

    # --------------------------------------------------------
    # Initialize MediaPipe
    # --------------------------------------------------------

    print(
        "\nInitializing MediaPipe..."
    )

    landmark_detector = LandmarkDetector()

    feature_extractor = FeatureExtractor()

    print(
        "MediaPipe initialized successfully."
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total_processed = 0
    total_skipped = 0

    # --------------------------------------------------------
    # Process all labels
    # --------------------------------------------------------

    try:

        for label in labels:

            processed, skipped = process_label(
                label,
                landmark_detector,
                feature_extractor
            )

            total_processed += processed
            total_skipped += skipped

    finally:

        landmark_detector.close()

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("WORD VIDEO PROCESSING COMPLETED")
    print("=" * 70)

    print(
        "\nWord classes:",
        len(labels)
    )

    print(
        "New videos processed:",
        total_processed
    )

    print(
        "Videos skipped:",
        total_skipped
    )

    print(
        "\nProcessed landmark location:"
    )

    print(
        OUTPUT_FOLDER
    )

    print("\n" + "=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
    