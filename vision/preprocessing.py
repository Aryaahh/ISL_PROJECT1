import cv2


class Preprocessor:

    def __init__(self):
        pass

    def resize_frame(self, frame, width=640, height=480):
        resized = cv2.resize(frame, (width, height))
        return resized

    def flip_frame(self, frame):
        flipped = cv2.flip(frame, 1)
        return flipped

    def bgr_to_rgb(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb_frame

    def process(self, frame):
        frame = self.resize_frame(frame)
        frame = self.flip_frame(frame)
        rgb_frame = self.bgr_to_rgb(frame)

        return rgb_frame


if __name__ == "__main__":
    print("Preprocessing module ready")