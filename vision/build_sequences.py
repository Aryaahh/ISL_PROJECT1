import os
import numpy as np

# =====================================================
# Paths
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LANDMARK_FOLDER = os.path.join(BASE_DIR, "data", "processed_landmarks")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "data", "sequences")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =====================================================
# Settings
# =====================================================

SEQUENCE_LENGTH = 30

X = []
y = []

labels = sorted([
    d for d in os.listdir(LANDMARK_FOLDER)
    if os.path.isdir(os.path.join(LANDMARK_FOLDER, d))
])

label_map = {label: i for i, label in enumerate(labels)}

print("=" * 60)
print("BUILDING SEQUENCES")
print("=" * 60)

# =====================================================
# Read every sentence
# =====================================================

for label in labels:

    label_path = os.path.join(LANDMARK_FOLDER, label)

    print(f"\nSentence : {label}")

    samples = sorted(os.listdir(label_path))

    for sample in samples:

        sample_path = os.path.join(label_path, sample)

        if not os.path.isdir(sample_path):
            continue

        files = sorted([
            f for f in os.listdir(sample_path)
            if f.endswith(".npy")
        ])

        if len(files) < SEQUENCE_LENGTH:
            continue

        landmarks = []

        for file in files[:SEQUENCE_LENGTH]:
            data = np.load(os.path.join(sample_path, file))
            landmarks.append(data)

        X.append(np.array(landmarks))
        y.append(label_map[label])

print("\nSaving Dataset...")

X = np.array(X)
y = np.array(y)

np.save(os.path.join(OUTPUT_FOLDER, "X.npy"), X)
np.save(os.path.join(OUTPUT_FOLDER, "y.npy"), y)
np.save(os.path.join(OUTPUT_FOLDER, "label_map.npy"), label_map)

print("=" * 60)
print("Sequence Building Completed")
print("=" * 60)
print("Total Sequences :", len(X))
print("Shape of X      :", X.shape)
print("Shape of y      :", y.shape)
print("=" * 60)
