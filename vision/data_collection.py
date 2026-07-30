import os
import numpy as np


class DataCollector:

    def __init__(self, save_path="data"):

        self.save_path = save_path

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)


    def save_sequence(self, sequence, label, sample_number):

        label_path = os.path.join(
            self.save_path,
            label
        )

        if not os.path.exists(label_path):
            os.makedirs(label_path)


        file_path = os.path.join(
            label_path,
            f"sample_{sample_number}.npy"
        )


        np.save(
            file_path,
            sequence
        )


        print("Saved:", file_path)



    def load_sequence(self, file_path):

        sequence = np.load(file_path)

        return sequence



if __name__ == "__main__":

    collector = DataCollector()

    print("Data Collection Module Ready")