import cv2
import numpy as np

from vision.camera import Camera
from vision.preprocessing import Preprocessor
from vision.landmark_detection import LandmarkDetector
from vision.feature_extraction import FeatureExtractor
from vision.sequence_builder import SequenceBuilder
from vision.data_collection import DataCollector


def main():

    camera = Camera()
    preprocessor = Preprocessor()
    landmark_detector = LandmarkDetector()
    feature_extractor = FeatureExtractor()

    sequence_builder = SequenceBuilder(
        sequence_length=30,
        feature_size=1629
    )

    data_collector = DataCollector(
        save_path="data"
    )

    label = "hello"
    sample_number = 1

    cap = camera.cap

    if not cap.isOpened():
        print("Camera not available")
        return

    print("ISL Data Collection Started")
    print("Collecting gesture:", label)
    print("Press 's' to save sequence")
    print("Press 'q' to exit")


    while True:

        ret, frame = cap.read()

        if not ret:
            break


        frame = cv2.flip(frame, 1)


        processed_frame = preprocessor.process(frame)


        _, results = landmark_detector.detect(
            processed_frame
        )


        left_hand, right_hand, pose, face = (
            feature_extractor.extract_all(results)
        )


        feature_vector = np.concatenate([
            left_hand,
            right_hand,
            pose,
            face
        ])


        sequence_builder.add_frame(
            feature_vector
        )


        if sequence_builder.is_ready():

            print(
                "Sequence Ready:",
                sequence_builder.get_sequence().shape
            )


        display_frame = landmark_detector.draw_landmarks(
            processed_frame,
            results
        )


        display_frame = cv2.cvtColor(
            display_frame,
            cv2.COLOR_RGB2BGR
        )


        cv2.imshow(
            "ISL Data Collection",
            display_frame
        )


        key = cv2.waitKey(1) & 0xFF


        # Save collected sequence
        if key == ord("s"):

            if sequence_builder.is_ready():

                sequence = sequence_builder.get_sequence()


                data_collector.save_sequence(
                    sequence,
                    label,
                    sample_number
                )


                sample_number += 1

                sequence_builder.reset()

            else:

                print("Collect 30 frames first")


        if key == ord("q"):
            break


    camera.stop()


if __name__ == "__main__":
    main()