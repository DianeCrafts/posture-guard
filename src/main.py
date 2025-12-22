import cv2
from video.camera import Camera
from pose.estimator import PoseEstimator
from ui.renderer import Renderer

def main():
    cam = Camera(index=0)
    pose_estimator = PoseEstimator()
    renderer = Renderer()

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break

            pose_result = pose_estimator.infer(frame)
            frame = renderer.draw_pose(frame, pose_result)

            cv2.imshow("PostureGuard - Pose Estimation", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
