import os
import numpy as np

path = "data/processed_landmarks"

fixed = 0

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".npy"):
            fp = os.path.join(root, file)

            arr = np.load(fp)

            if arr.shape == (1629,):
                arr = np.pad(arr, (0, 30))
                np.save(fp, arr)
                fixed += 1

print("Fixed files:", fixed)
