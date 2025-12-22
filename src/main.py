import cv2
from video.camera import Camera

def main():
    cam = Camera(index=0)

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break

            cv2.imshow("PostureGuard - Camera", frame)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
