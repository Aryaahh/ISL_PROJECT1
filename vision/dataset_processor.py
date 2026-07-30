import os


class DatasetProcessor:

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path


    def get_sentence_folders(self):

        sentence_folders = []

        for folder in sorted(os.listdir(self.dataset_path)):

            folder_path = os.path.join(self.dataset_path, folder)

            if os.path.isdir(folder_path):
                sentence_folders.append(folder_path)

        return sentence_folders


    def get_sample_folders(self, sentence_folder):

        sample_folders = []

        for folder in sorted(os.listdir(sentence_folder)):

            folder_path = os.path.join(sentence_folder, folder)

            if os.path.isdir(folder_path):
                sample_folders.append(folder_path)

        return sample_folders


    def get_frame_files(self, sample_folder):

        frame_files = []

        for file in sorted(os.listdir(sample_folder)):

            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                frame_files.append(
                    os.path.join(sample_folder, file)
                )

        return frame_files


if __name__ == "__main__":

    dataset_path = (
        "data/archive/ISL_CSLRT_Corpus/"
        "ISL_CSLRT_Corpus/"
        "Frames_Sentence_Level"
    )

    processor = DatasetProcessor(dataset_path)

    sentence_folders = processor.get_sentence_folders()

    print("\nDataset Summary\n")

    for sentence in sentence_folders:

        samples = processor.get_sample_folders(sentence)

        print(f"\nSentence : {os.path.basename(sentence)}")

        for sample in samples:

            frames = processor.get_frame_files(sample)

            print(
                f"   Sample {os.path.basename(sample)}"
                f" --> {len(frames)} frames"
            )

    print("\nFinished.")