import numpy as np


class SequenceBuilder:

    def __init__(self, sequence_length=30, feature_size=1629):

        self.sequence_length = sequence_length
        self.feature_size = feature_size
        self.sequence = []


    def add_frame(self, feature_vector):

        self.sequence.append(feature_vector)

        # Keep only last N frames
        if len(self.sequence) > self.sequence_length:
            self.sequence.pop(0)


    def is_ready(self):

        return len(self.sequence) == self.sequence_length


    def get_sequence(self):

        if self.is_ready():

            return np.array(self.sequence)

        return None


    def reset(self):

        self.sequence = []


if __name__ == "__main__":

    builder = SequenceBuilder()

    print("Sequence Builder Ready")