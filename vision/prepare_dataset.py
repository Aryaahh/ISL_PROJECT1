import os
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

DATASET_ROOT = os.path.join(
    "data",
    "word_processed_landmarks"
)

OUTPUT_DIR = os.path.join(
    "data",
    "training"
)

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 1659


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print("=" * 70)
    print("ISL DATASET PREPARATION")
    print("=" * 70)

    print("\nDataset location:")
    print(DATASET_ROOT)

    if not os.path.exists(DATASET_ROOT):

        raise FileNotFoundError(
            f"Dataset folder not found:\n{DATASET_ROOT}"
        )

    # --------------------------------------------------------
    # Find word classes
    # --------------------------------------------------------

    labels = sorted(
        [
            name
            for name in os.listdir(DATASET_ROOT)
            if os.path.isdir(
                os.path.join(DATASET_ROOT, name)
            )
        ]
    )

    if not labels:

        raise ValueError(
            "No word classes found."
        )

    print("\nWord classes found:", len(labels))

    for index, label in enumerate(labels):

        print(
            f"{index}: {label}"
        )

    # --------------------------------------------------------
    # Create label mapping
    # --------------------------------------------------------

    label_to_index = {
        label: index
        for index, label in enumerate(labels)
    }

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    X = []
    y = []

    total_samples = 0
    skipped_samples = 0

    # ========================================================
    # PROCESS EACH WORD CLASS
    # ========================================================

    for label in labels:

        label_path = os.path.join(
            DATASET_ROOT,
            label
        )

        sample_names = sorted(
            [
                name
                for name in os.listdir(label_path)
                if os.path.isdir(
                    os.path.join(
                        label_path,
                        name
                    )
                )
            ]
        )

        print("\n" + "-" * 70)
        print(
            f"Label: {label}"
        )
        print(
            f"Samples found: {len(sample_names)}"
        )

        # ----------------------------------------------------
        # Process samples
        # ----------------------------------------------------

        for sample_name in sample_names:

            sample_path = os.path.join(
                label_path,
                sample_name
            )

            frames = []

            valid_sample = True

            # -----------------------------------------------
            # Read 30 frames
            # -----------------------------------------------

            for frame_number in range(
                1,
                SEQUENCE_LENGTH + 1
            ):

                frame_filename = (
                    f"frame_{frame_number:03d}.npy"
                )

                frame_path = os.path.join(
                    sample_path,
                    frame_filename
                )

                if not os.path.exists(
                    frame_path
                ):

                    print(
                        f"SKIP {label}/{sample_name}: "
                        f"missing {frame_filename}"
                    )

                    valid_sample = False

                    break

                frame = np.load(
                    frame_path
                )

                # -------------------------------------------
                # Check frame shape
                # -------------------------------------------

                if frame.shape != (
                    FEATURE_SIZE,
                ):

                    print(
                        f"SKIP {label}/{sample_name}: "
                        f"invalid frame shape "
                        f"{frame.shape}"
                    )

                    valid_sample = False

                    break

                frames.append(
                    frame.astype(
                        np.float32
                    )
                )

            # -----------------------------------------------
            # Skip invalid sample
            # -----------------------------------------------

            if not valid_sample:

                skipped_samples += 1

                continue

            # -----------------------------------------------
            # Create sequence
            # -----------------------------------------------

            sequence = np.asarray(
                frames,
                dtype=np.float32
            )

            # -----------------------------------------------
            # Final shape check
            # -----------------------------------------------

            if sequence.shape != (
                SEQUENCE_LENGTH,
                FEATURE_SIZE
            ):

                print(
                    f"SKIP {label}/{sample_name}: "
                    f"invalid sequence shape "
                    f"{sequence.shape}"
                )

                skipped_samples += 1

                continue

            # -----------------------------------------------
            # Add sequence and label
            # -----------------------------------------------

            X.append(
                sequence
            )

            y.append(
                label_to_index[label]
            )

            total_samples += 1

            print(
                f"Loaded: {label}/{sample_name} "
                f"{sequence.shape}"
            )

    # ========================================================
    # CONVERT TO NUMPY ARRAYS
    # ========================================================

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64
    )

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(
        "\nX shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )

    print(
        "Total valid samples:",
        total_samples
    )

    print(
        "Skipped samples:",
        skipped_samples
    )

    print(
        "Number of classes:",
        len(labels)
    )

    # ========================================================
    # CHECK DATASET
    # ========================================================

    expected_shape = (
        SEQUENCE_LENGTH,
        FEATURE_SIZE
    )

    if X.ndim != 3:

        raise ValueError(
            f"Invalid X dimensions: {X.shape}"
        )

    if X.shape[1:] != expected_shape:

        raise ValueError(
            f"Invalid X shape: {X.shape}. "
            f"Expected (?, {SEQUENCE_LENGTH}, "
            f"{FEATURE_SIZE})"
        )

    if len(X) != len(y):

        raise ValueError(
            "X and y contain different "
            "numbers of samples."
        )

    # ========================================================
    # PRINT CLASS DISTRIBUTION
    # ========================================================

    print("\nClass distribution:")

    for label in labels:

        label_index = label_to_index[label]

        count = np.sum(
            y == label_index
        )

        print(
            f"{label:15s}: {count} samples"
        )

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ========================================================
    # SAVE DATASET
    # ========================================================

    X_path = os.path.join(
        OUTPUT_DIR,
        "X.npy"
    )

    y_path = os.path.join(
        OUTPUT_DIR,
        "y.npy"
    )

    labels_path = os.path.join(
        OUTPUT_DIR,
        "labels.npy"
    )

    np.save(
        X_path,
        X
    )

    np.save(
        y_path,
        y
    )

    np.save(
        labels_path,
        np.asarray(
            labels
        )
    )

    # ========================================================
    # SAVE LABEL MAPPING
    # ========================================================

    mapping_path = os.path.join(
        OUTPUT_DIR,
        "label_mapping.txt"
    )

    with open(
        mapping_path,
        "w",
        encoding="utf-8"
    ) as file:

        for label, index in label_to_index.items():

            file.write(
                f"{index}: {label}\n"
            )

    # ========================================================
    # FINISHED
    # ========================================================

    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)

    print("\nSaved files:")

    print(
        X_path
    )

    print(
        y_path
    )

    print(
        labels_path
    )

    print(
        mapping_path
    )

    print("\nFinal X shape:")
    print(
        X.shape
    )

    print("\nFinal y shape:")
    print(
        y.shape
    )

    print("\nEach sample:")
    print(
        "(30 frames, 1659 features)"
    )

    print("\nReady for model training.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    load_dataset()
    