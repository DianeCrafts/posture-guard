import cv2

class Camera:
    def __init__(self, index: int = 0, width: int | None = 1280, height: int | None = 720):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")

        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        return self.cap.read()  # (ok, frame)

    def release(self):
        self.cap.release()
