from analysis import metrics


class PostureAnalyzer:
    def analyze(self, pose_result):
        if not pose_result.pose_landmarks:
            return None

        lm = pose_result.pose_landmarks.landmark

        return {
            "neck_angle_deg": metrics.neck_angle(lm),
            "shoulder_diff": metrics.shoulder_height_diff(lm),
        }
