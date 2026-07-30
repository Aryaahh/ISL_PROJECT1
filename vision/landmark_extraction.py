import os
import cv2
import numpy as np

from landmark_detection import LandmarkDetector
from feature_extraction import FeatureExtractor


# ==========================================
# Paths
# ==========================================

INPUT_FOLDER = "data/processed_frames"
OUTPUT_FOLDER = "data/processed_landmarks"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================================
# Initialize
# ==========================================

detector = LandmarkDetector()
extractor = FeatureExtractor()


total_sentences = 0
total_videos = 0
total_frames = 0

print("=" * 60)
print("LANDMARK EXTRACTION STARTED")
print("=" * 60)


# ==========================================
# Loop through every sentence
# ==========================================

for sentence in sorted(os.listdir(INPUT_FOLDER)):

    sentence_path = os.path.join(INPUT_FOLDER, sentence)

    if not os.path.isdir(sentence_path):
        continue

    total_sentences += 1

    print(f"\nSentence : {sentence}")

    output_sentence = os.path.join(
        OUTPUT_FOLDER,
        sentence
    )

    os.makedirs(output_sentence, exist_ok=True)

    # ======================================
    # Loop through every video
    # ======================================

    for video in sorted(os.listdir(sentence_path)):

        video_path = os.path.join(sentence_path, video)

        if not os.path.isdir(video_path):
            continue

        total_videos += 1

        output_video = os.path.join(
            output_sentence,
            video
        )

        os.makedirs(output_video, exist_ok=True)

        frame_counter = 0

        # ==================================
        # Loop through every frame
        # ==================================

        frame_files = sorted([
            f for f in os.listdir(video_path)
            if f.endswith(".jpg")
        ])

        for frame_name in frame_files:

            frame_path = os.path.join(
                video_path,
                frame_name
            )

            frame = cv2.imread(frame_path)

            if frame is None:
                continue

            # ------------------------------
            # Landmark Detection
            # ------------------------------

            results = detector.detect(frame)

            # ------------------------------
            # Feature Extraction
            # ------------------------------

            features = extractor.extract_all_features(results)

            # ------------------------------
            # Save Landmark
            # ------------------------------

            save_name = frame_name.replace(
                ".jpg",
                ".npy"
            )

            save_path = os.path.join(
                output_video,
                save_name
            )

            np.save(save_path, features)

            frame_counter += 1
            total_frames += 1

        print(f"   {video} --> {frame_counter} Frames")


# ==========================================
# Finish
# ==========================================

detector.close()

print("\n" + "=" * 60)
print("LANDMARK EXTRACTION COMPLETED")
print("=" * 60)

print(f"Sentences : {total_sentences}")
print(f"Videos    : {total_videos}")
print(f"Frames    : {total_frames}")

print("=" * 60)
