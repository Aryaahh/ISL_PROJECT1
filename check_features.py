import os
import numpy as np

ROOT = r"data\processed_landmarks"

counts = {}
examples = {}

for root, dirs, files in os.walk(ROOT):
    for file in files:
        if file.endswith(".npy"):
            path = os.path.join(root, file)

            try:
                data = np.load(path)
                shape = data.shape

                counts[shape] = counts.get(shape, 0) + 1

                if shape not in examples:
                    examples[shape] = path

            except Exception as e:
                print("ERROR:", path, e)

print("\nFeature shapes found:")
print("=" * 50)

for shape, count in counts.items():
    print(f"Shape : {shape}")
    print(f"Count : {count}")
    print(f"Example: {examples[shape]}")
    print("-" * 50)
