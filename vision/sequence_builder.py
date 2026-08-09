import numpy as np


class SequenceBuilder:

    def __init__(self, sequence_length=30, feature_size=1659):
        self.sequence_length = sequence_length
        self.feature_size = feature_size
        self.sequence = []

    def add_frame(self, feature_vector):

        feature_vector = np.asarray(
            feature_vector,
            dtype=np.float32
        )

        # Check feature dimension
        if feature_vector.shape != (self.feature_size,):
            raise ValueError(
                f"Invalid feature shape: {feature_vector.shape}. "
                f"Expected: ({self.feature_size},)"
            )

        self.sequence.append(feature_vector)

        # Keep only the latest 30 frames
        if len(self.sequence) > self.sequence_length:
            self.sequence.pop(0)

    def is_ready(self):

        return len(self.sequence) == self.sequence_length

    def get_sequence(self):

        if self.is_ready():

            sequence = np.array(
                self.sequence,
                dtype=np.float32
            )

            # Final safety check
            if sequence.shape != (
                self.sequence_length,
                self.feature_size
            ):
                raise ValueError(
                    f"Invalid sequence shape: {sequence.shape}. "
                    f"Expected: "
                    f"({self.sequence_length}, {self.feature_size})"
                )

            return sequence

        return None

    def reset(self):

        self.sequence = []


if __name__ == "__main__":

    builder = SequenceBuilder()

    print("Sequence Builder Ready")
    print("Sequence length :", builder.sequence_length)
    print("Feature size    :", builder.feature_size)