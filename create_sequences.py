import os
import numpy as np
import pickle

DATA_PATH = "data/processed_landmarks"
SEQ_LENGTH = 30

X = []
y = []

labels = sorted(os.listdir(DATA_PATH))

label_map = {label:i for i, label in enumerate(labels)}

print("Classes:", len(labels))

for label in labels:
    label_path = os.path.join(DATA_PATH, label)

    if not os.path.isdir(label_path):
        continue

    for video in os.listdir(label_path):
        video_path = os.path.join(label_path, video)

        if not os.path.isdir(video_path):
            continue

        frames = []

        files = sorted(
            [f for f in os.listdir(video_path) if f.endswith(".npy")]
        )

        for f in files:
            frames.append(
                np.load(os.path.join(video_path,f))
            )

        # create sliding sequences
        for i in range(len(frames)-SEQ_LENGTH+1):
            sequence = frames[i:i+SEQ_LENGTH]

            X.append(sequence)
            y.append(label_map[label])


X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)


os.makedirs("data/sequences", exist_ok=True)

np.save("data/sequences/X.npy", X)
np.save("data/sequences/y.npy", y)

with open("data/sequences/labels.pkl","wb") as f:
    pickle.dump(label_map,f)

print("Sequence creation completed")
