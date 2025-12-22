import cv2
from video.camera import Camera
from pose.estimator import PoseEstimator
from analysis.posture_analyzer import PostureAnalyzer
from ui.renderer import Renderer

def main():
    cam = Camera(index=0)
    pose_estimator = PoseEstimator()
    analyzer = PostureAnalyzer()
    renderer = Renderer()

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break

            pose_result = pose_estimator.infer(frame)
            metrics = analyzer.analyze(pose_result)

            frame = renderer.draw_pose(frame, pose_result)
            frame = renderer.draw_metrics(frame, metrics)

            cv2.imshow("PostureGuard - Metrics", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
