import os
import cv2

# ==========================
# Dataset Paths
# ==========================

VIDEO_FOLDER = r"E:\ISL_PROJECT1\data\archive\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Videos_Sentence_Level"

OUTPUT_FOLDER = r"E:\ISL_PROJECT1\data\processed_frames"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("=" * 60)
print("FRAME EXTRACTION STARTED")
print("=" * 60)

sentence_count = 0
video_count = 0

# ==========================
# Loop through every sentence
# ==========================

for sentence in sorted(os.listdir(VIDEO_FOLDER)):

    sentence_path = os.path.join(VIDEO_FOLDER, sentence)

    if not os.path.isdir(sentence_path):
        continue

    sentence_count += 1

    print(f"\nSentence: {sentence}")

    output_sentence = os.path.join(OUTPUT_FOLDER, sentence)
    os.makedirs(output_sentence, exist_ok=True)

    videos = sorted(os.listdir(sentence_path))

    for video in videos:

        if not video.lower().endswith((".mp4", ".avi", ".mov")):
            continue

        video_count += 1

        video_path = os.path.join(sentence_path, video)

        sample_name = os.path.splitext(video)[0]

        output_sample = os.path.join(output_sentence, sample_name)
        os.makedirs(output_sample, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        frame_no = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_name = f"frame_{frame_no:04d}.jpg"

            cv2.imwrite(
                os.path.join(output_sample, frame_name),
                frame
            )

            frame_no += 1

        cap.release()

        print(f"   {video} --> {frame_no} frames")

print("\n" + "=" * 60)
print("Frame Extraction Completed")
print(f"Sentences : {sentence_count}")
print(f"Videos    : {video_count}")
print("=" * 60)
