import cv2
import time

from video.camera import Camera
from pose.estimator import PoseEstimator
from analysis.posture_analyzer import PostureAnalyzer
from decision.posture_classifier import PostureClassifier
from decision.temporal_smoother import TemporalSmoother
from decision.alert_manager import AlertManager
from ui.renderer import Renderer
from data.recorder import DataRecorder


def trigger_alert():
    try:
        import winsound
        winsound.Beep(1000, 500)
    except ImportError:
        print("ALERT: Bad posture detected!")


def main():
    cam = Camera(index=0)
    pose_estimator = PoseEstimator()
    analyzer = PostureAnalyzer()
    classifier = PostureClassifier()
    smoother = TemporalSmoother(confirm_time=1.5)
    alert_manager = AlertManager(
        bad_posture_time=5.0,
        cooldown_time=15.0
    )
    renderer = Renderer()
    recorder = DataRecorder(sample_hz=10)
    # visual alert timing
    alert_visible_until = 0.0

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break

            pose_result = pose_estimator.infer(frame)
            metrics = analyzer.analyze(pose_result)
            classification = classifier.classify(metrics)
            smoothed = smoother.update(classification)

            if alert_manager.check(smoothed):
                trigger_alert()
                alert_visible_until = time.time() + 2.0

            alert_active = time.time() < alert_visible_until

            frame = renderer.draw_pose(frame, pose_result)
            frame = renderer.draw_metrics(frame, metrics)
            frame = renderer.draw_posture_state(frame, classification)
            frame = renderer.draw_stable_state(frame, smoothed)
            frame = renderer.draw_alert(frame, alert_active)

            cv2.imshow("PostureGuard - Classification", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("g"):
                recorder.set_label("GOOD")
            elif key == ord("b"):
                recorder.set_label("BAD")
            elif key == ord("u"):
                recorder.set_label("UNKNOWN")
            elif key == ord("q"):
                break

            recorder.update(metrics)


    finally:
        cam.release()
        cv2.destroyAllWindows()
        recorder.close()


if __name__ == "__main__":
    main()
