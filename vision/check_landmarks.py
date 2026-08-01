import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LANDMARK_FOLDER = os.path.join(BASE_DIR, "data", "processed_landmarks")

print("=" * 60)
print("CHECKING LANDMARK FILES")
print("=" * 60)

shapes = {}

for sentence in sorted(os.listdir(LANDMARK_FOLDER)):

    sentence_path = os.path.join(LANDMARK_FOLDER, sentence)

    if not os.path.isdir(sentence_path):
        continue

    for video in sorted(os.listdir(sentence_path)):

        video_path = os.path.join(sentence_path, video)

        if not os.path.isdir(video_path):
            continue

        for file in sorted(os.listdir(video_path)):

            if not file.endswith(".npy"):
                continue

            path = os.path.join(video_path, file)

            data = np.load(path)

            shape = data.shape

            if shape not in shapes:
                shapes[shape] = 0

            shapes[shape] += 1

print("\nDifferent feature shapes found:\n")

for shape, count in shapes.items():
    print(f"{shape}  --> {count} files")
    