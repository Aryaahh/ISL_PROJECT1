import os
import numpy as np

# ==========================================================
# Configuration
# ==========================================================

# Get the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LANDMARK_FOLDER = os.path.join(BASE_DIR, "data", "processed_landmarks")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "data", "sequences")

SEQUENCE_LENGTH = 30

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("=" * 60)
print("BUILDING SEQUENCES")
print("=" * 60)

total_sentences = 0
total_videos = 0
total_sequences = 0

# ==========================================================
# Loop through all sentence folders
# ==========================================================

for sentence in sorted(os.listdir(LANDMARK_FOLDER)):

    sentence_path = os.path.join(LANDMARK_FOLDER, sentence)

    if not os.path.isdir(sentence_path):
        continue

    total_sentences += 1
    print(f"\nSentence : {sentence}")

    output_sentence = os.path.join(OUTPUT_FOLDER, sentence)
    os.makedirs(output_sentence, exist_ok=True)

    # ======================================================
    # Loop through all video folders
    # ======================================================

    for video in sorted(os.listdir(sentence_path)):

        video_path = os.path.join(sentence_path, video)

        if not os.path.isdir(video_path):
            continue

        total_videos += 1

        output_video = os.path.join(output_sentence, video)
        os.makedirs(output_video, exist_ok=True)

        # Get all landmark files
        landmark_files = sorted([
            file for file in os.listdir(video_path)
            if file.endswith(".npy")
        ])

        sequence_count = 0

        # ==================================================
        # Create sequences
        # ==================================================

        for start in range(0, len(landmark_files), SEQUENCE_LENGTH):

            current_files = landmark_files[start:start + SEQUENCE_LENGTH]

            # Ignore incomplete sequences
            if len(current_files) < SEQUENCE_LENGTH:
                continue

            sequence = []

            for landmark_file in current_files:

                landmark_path = os.path.join(
                    video_path,
                    landmark_file
                )

                features = np.load(landmark_path)

                sequence.append(features)

            sequence = np.array(sequence, dtype=np.float32)

            save_name = f"sequence_{sequence_count:03d}.npy"

            save_path = os.path.join(
                output_video,
                save_name
            )

            np.save(save_path, sequence)

            sequence_count += 1
            total_sequences += 1

        print(f"   {video} --> {sequence_count} sequences")

print("\n" + "=" * 60)
print("SEQUENCE BUILDING COMPLETED")
print("=" * 60)
print(f"Sentences : {total_sentences}")
print(f"Videos    : {total_videos}")
print(f"Sequences : {total_sequences}")
print("=" * 60)
