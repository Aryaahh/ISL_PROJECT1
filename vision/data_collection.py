import os
import numpy as np


class DataCollector:

    def __init__(self, save_path="data"):

        self.save_path = save_path

        os.makedirs(
            self.save_path,
            exist_ok=True
        )

    # ========================================================
    # SAVE SEQUENCE
    # ========================================================

    def save_sequence(
        self,
        sequence,
        label,
        sample_number
    ):

        # ----------------------------------------------------
        # Check sequence shape
        # ----------------------------------------------------

        sequence = np.asarray(
            sequence,
            dtype=np.float32
        )

        if sequence.ndim != 2:

            raise ValueError(
                "Sequence must have shape "
                "(frames, features). "
                f"Received: {sequence.shape}"
            )

        if sequence.shape[0] != 30:

            raise ValueError(
                "Sequence must contain exactly "
                f"30 frames. Received: {sequence.shape[0]}"
            )

        if sequence.shape[1] != 1659:

            raise ValueError(
                "Each frame must contain exactly "
                f"1659 features. Received: {sequence.shape[1]}"
            )

        # ----------------------------------------------------
        # Label directory
        # ----------------------------------------------------

        label_path = os.path.join(
            self.save_path,
            "processed_landmarks",
            label
        )

        os.makedirs(
            label_path,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Sample directory
        # ----------------------------------------------------

        sample_path = os.path.join(
            label_path,
            f"sample_{sample_number}"
        )

        os.makedirs(
            sample_path,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Save each frame separately
        # ----------------------------------------------------

        for frame_number in range(30):

            frame = sequence[frame_number]

            frame_path = os.path.join(
                sample_path,
                f"frame_{frame_number + 1:03d}.npy"
            )

            np.save(
                frame_path,
                frame
            )

        print(
            f"Saved sequence: "
            f"{label}/sample_{sample_number}"
        )

        print(
            "Frames       : 30"
        )

        print(
            "Features/frame: 1659"
        )

        print(
            "Shape        :",
            sequence.shape
        )

        print(
            "Location     :",
            sample_path
        )

    # ========================================================
    # LOAD COMPLETE SEQUENCE
    # ========================================================

    def load_sequence(self, file_path):

        sequence = np.load(
            file_path
        )

        return sequence


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    collector = DataCollector()

    print("=" * 50)
    print("Data Collection Module Ready")
    print("=" * 50)
    