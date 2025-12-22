import cv2
from video.camera import Camera
from pose.estimator import PoseEstimator
from analysis.posture_analyzer import PostureAnalyzer
from decision.posture_classifier import PostureClassifier
from ui.renderer import Renderer
from decision.temporal_smoother import TemporalSmoother
def main():
    cam = Camera(index=0)
    pose_estimator = PoseEstimator()
    analyzer = PostureAnalyzer()
    classifier = PostureClassifier()
    smoother = TemporalSmoother(confirm_time=1.5)
    renderer = Renderer()

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break

            pose_result = pose_estimator.infer(frame)
            metrics = analyzer.analyze(pose_result)
            classification = classifier.classify(metrics)
            smoothed = smoother.update(classification)

            frame = renderer.draw_pose(frame, pose_result)
            frame = renderer.draw_metrics(frame, metrics)
            frame = renderer.draw_posture_state(frame, classification)
            frame = renderer.draw_stable_state(frame, smoothed)

            cv2.imshow("PostureGuard - Classification", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
