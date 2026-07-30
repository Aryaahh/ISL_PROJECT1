import cv2


class Camera:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)

    def start(self):
        if not self.cap.isOpened():
            print("Error: Cannot open camera")
            return

        print("Camera started")
        print("Press 'q' to exit")

        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Error: Cannot read frame")
                break

            cv2.imshow("ISL Vision - Camera Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop()

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    camera = Camera()
    camera.start()