import os
import cv2

# Change this path to your dataset folder
DATASET_PATH = r"E:\ISL_PROJECT1\data\archive\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Videos_Sentence_Level"
print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

total_videos = 0

for sentence in sorted(os.listdir(DATASET_PATH)):

    sentence_path = os.path.join(DATASET_PATH, sentence)

    if not os.path.isdir(sentence_path):
        continue

    videos = [v for v in os.listdir(sentence_path)
              if v.endswith((".mp4", ".avi", ".mov"))]

    print(f"\nSentence : {sentence}")
    print(f"Number of Videos : {len(videos)}")

    total_videos += len(videos)

    # Show information for each video
    for video in videos:

        video_path = os.path.join(sentence_path, video)

        cap = cv2.VideoCapture(video_path)

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"  {video}")
        print(f"     Frames : {frames}")
        print(f"     FPS    : {fps:.2f}")
        print(f"     Size   : {width} x {height}")

        cap.release()

print("\n" + "=" * 60)
print(f"Total Videos : {total_videos}")
print("=" * 60)