import cv2
import mediapipe as mp

class Renderer:
    def __init__(self):
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def draw_pose(self, frame, pose_result):
        if pose_result.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                pose_result.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
            )
        return frame

    def draw_metrics(self, frame, metrics):
        if not metrics:
            return frame

        y = 30
        for key, value in metrics.items():
            text = f"{key}: {value:.2f}"
            cv2.putText(
                frame,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            y += 25

        return frame
    

    def draw_posture_state(self, frame, classification):
        if not classification:
            return frame

        state = classification["state"]

        color_map = {
            "GOOD": (0, 255, 0),
            "WARNING": (0, 255, 255),
            "BAD": (0, 0, 255),
        }

        cv2.putText(
            frame,
            f"Posture: {state}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color_map[state],
            3,
        )

        return frame



    def draw_stable_state(self, frame, smoothed):
        if not smoothed:
            return frame

        state = smoothed["stable_state"]
        duration = smoothed["duration_sec"]

        color_map = {
            "GOOD": (0, 255, 0),
            "WARNING": (0, 255, 255),
            "BAD": (0, 0, 255),
        }

        text = f"Stable posture: {state} ({duration:.1f}s)"

        cv2.putText(
            frame,
            text,
            (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color_map[state],
            2,
        )

        return frame
    
    def draw_alert(self, frame, alert_active):
        if alert_active:
            cv2.putText(
                frame,
                "FIX YOUR POSTURE!!",
                (300, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )
        return frame

