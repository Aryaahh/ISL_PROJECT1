import os
import numpy as np
import pickle

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

LANDMARK_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    "processed_landmarks"
)

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    "sequences"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 30
FEATURE_SIZE = 1659

# ============================================================
# LABELS
# ============================================================

labels = sorted([
    d for d in os.listdir(LANDMARK_FOLDER)
    if os.path.isdir(
        os.path.join(LANDMARK_FOLDER, d)
    )
])

label_map = {
    label: index
    for index, label in enumerate(labels)
}

print("=" * 70)
print("BUILDING CLEAN ISL SEQUENCE DATASET")
print("=" * 70)

print()
print("Landmark folder:")
print(LANDMARK_FOLDER)

print()
print("Number of classes:", len(labels))

# ============================================================
# DATA
# ============================================================

X = []
y = []

total_samples = 0
valid_samples = 0
skipped_samples = 0

# ============================================================
# PROCESS EACH CLASS
# ============================================================

for label in labels:

    label_path = os.path.join(
        LANDMARK_FOLDER,
        label
    )

    print()
    print("-" * 70)
    print("Label:", label)

    sample_folders = sorted([
        d
        for d in os.listdir(label_path)
        if os.path.isdir(
            os.path.join(label_path, d)
        )
    ])

    print(
        "Source samples:",
        len(sample_folders)
    )

    for sample in sample_folders:

        total_samples += 1

        sample_path = os.path.join(
            label_path,
            sample
        )

        frame_files = sorted([
            f
            for f in os.listdir(sample_path)
            if f.lower().endswith(".npy")
        ])

        # ----------------------------------------------------
        # Need at least 30 frames
        # ----------------------------------------------------

        if len(frame_files) < SEQUENCE_LENGTH:

            print(
                "SKIP:",
                sample,
                "-> only",
                len(frame_files),
                "frames"
            )

            skipped_samples += 1
            continue

        # ----------------------------------------------------
        # Take exactly 30 frames
        # ----------------------------------------------------

        sequence_files = frame_files[
            :SEQUENCE_LENGTH
        ]

        sequence = []

        valid = True

        for frame_file in sequence_files:

            frame_path = os.path.join(
                sample_path,
                frame_file
            )

            try:

                features = np.load(
                    frame_path
                )

            except Exception as e:

                print(
                    "ERROR loading:",
                    frame_path
                )

                print(e)

                valid = False
                break

            # ------------------------------------------------
            # Check feature size
            # ------------------------------------------------

            if features.shape != (FEATURE_SIZE,):

                print(
                    "BAD FEATURE SIZE:",
                    frame_path,
                    features.shape
                )

                valid = False
                break

            sequence.append(
                features.astype(
                    np.float32
                )
            )

        if not valid:

            skipped_samples += 1
            continue

        # ----------------------------------------------------
        # Convert sequence
        # ----------------------------------------------------

        sequence = np.asarray(
            sequence,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Final shape check
        # ----------------------------------------------------

        if sequence.shape != (
            SEQUENCE_LENGTH,
            FEATURE_SIZE
        ):

            print(
                "BAD SEQUENCE:",
                sample,
                sequence.shape
            )

            skipped_samples += 1
            continue

        X.append(sequence)

        y.append(
            label_map[label]
        )

        valid_samples += 1

    print(
        "Valid samples for label:",
        sum(
            1
            for value in y
            if value == label_map[label]
        )
    )

# ============================================================
# CONVERT TO NUMPY
# ============================================================

print()
print("=" * 70)
print("CREATING NUMPY DATASET")
print("=" * 70)

X = np.asarray(
    X,
    dtype=np.float32
)

y = np.asarray(
    y,
    dtype=np.int32
)

print()
print("X shape:", X.shape)
print("y shape:", y.shape)

# ============================================================
# SAVE
# ============================================================

X_PATH = os.path.join(
    OUTPUT_FOLDER,
    "X.npy"
)

Y_PATH = os.path.join(
    OUTPUT_FOLDER,
    "y.npy"
)

LABEL_PATH = os.path.join(
    OUTPUT_FOLDER,
    "labels.pkl"
)

np.save(
    X_PATH,
    X
)

np.save(
    Y_PATH,
    y
)

with open(
    LABEL_PATH,
    "wb"
) as f:

    pickle.dump(
        label_map,
        f
    )

# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("DATASET BUILD COMPLETED")
print("=" * 70)

print()
print("Total source samples :", total_samples)
print("Valid sequences      :", valid_samples)
print("Skipped samples      :", skipped_samples)

print()
print("X:", X.shape)
print("y:", y.shape)

print()
print("Number of classes:", len(label_map))

print()
print("Label mapping saved:")
print(LABEL_PATH)

print()
print("=" * 70)