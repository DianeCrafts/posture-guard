class PostureClassifier:
    def __init__(self):
        # Thresholds (tunable)
        self.neck_good = 15
        self.neck_bad = 25

        self.shoulder_good = 0.03
        self.shoulder_bad = 0.06

    def classify(self, metrics):
        if not metrics:
            return None

        neck_state = self._classify_neck(metrics["neck_angle_deg"])
        shoulder_state = self._classify_shoulders(metrics["shoulder_diff"])

        # Overall posture state
        if "BAD" in (neck_state, shoulder_state):
            overall = "BAD"
        elif "WARNING" in (neck_state, shoulder_state):
            overall = "WARNING"
        else:
            overall = "GOOD"

        return {
            "state": overall,
            "details": {
                "neck_angle": neck_state,
                "shoulders": shoulder_state,
            },
        }

    def _classify_neck(self, angle):
        if angle < self.neck_good:
            return "GOOD"
        elif angle < self.neck_bad:
            return "WARNING"
        else:
            return "BAD"

    def _classify_shoulders(self, diff):
        if diff < self.shoulder_good:
            return "GOOD"
        elif diff < self.shoulder_bad:
            return "WARNING"
        else:
            return "BAD"
