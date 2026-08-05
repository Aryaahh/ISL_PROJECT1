import os
import cv2

# =====================================================
# Project Paths
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIDEO_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    "archive (2)",
    "ISL_CSLRT_Corpus",
    "ISL_CSLRT_Corpus",
    "Videos_Sentence_Level"
)

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    "processed_frames"
)

# =====================================================
# Check Dataset
# =====================================================

if not os.path.exists(VIDEO_FOLDER):
    print("=" * 60)
    print("ERROR")
    print("=" * 60)
    print("Dataset folder not found!")
    print("\nExpected location:")
    print(VIDEO_FOLDER)
    exit()

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("=" * 60)
print("FRAME EXTRACTION STARTED")
print("=" * 60)

sentence_count = 0
video_count = 0

# =====================================================
# Extract Frames
# =====================================================

for sentence in sorted(os.listdir(VIDEO_FOLDER)):

    sentence_path = os.path.join(VIDEO_FOLDER, sentence)

    if not os.path.isdir(sentence_path):
        continue

    sentence_count += 1

    print(f"\nSentence : {sentence}")

    output_sentence = os.path.join(
        OUTPUT_FOLDER,
        sentence
    )

    os.makedirs(output_sentence, exist_ok=True)

    for video in sorted(os.listdir(sentence_path)):

        if not video.lower().endswith((".mp4", ".avi", ".mov")):
            continue

        video_count += 1

        video_path = os.path.join(sentence_path, video)

        sample_name = os.path.splitext(video)[0]

        output_sample = os.path.join(
            output_sentence,
            sample_name
        )

        os.makedirs(output_sample, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        frame_no = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_file = os.path.join(
                output_sample,
                f"frame_{frame_no:04d}.jpg"
            )

            cv2.imwrite(frame_file, frame)

            frame_no += 1

        cap.release()

        print(f"   {video}  -->  {frame_no} frames")

print("\n" + "=" * 60)
print("FRAME EXTRACTION COMPLETED")
print("=" * 60)
print(f"Total Sentences : {sentence_count}")
print(f"Total Videos    : {video_count}")
print(f"Frames saved in : {OUTPUT_FOLDER}")
print("=" * 60)
