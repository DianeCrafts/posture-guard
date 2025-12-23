import joblib
import numpy as np
from pathlib import Path


class MLPostureClassifier:
    def __init__(self, model_path="models/posture_model.pkl", threshold=0.5):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = joblib.load(model_path)
        self.threshold = threshold

    def classify(self, metrics):
        if not metrics:
            return None

        X = np.array([[
            metrics["neck_angle_deg"],
            metrics["shoulder_diff"],
        ]])

        prob_bad = self.model.predict_proba(X)[0][1]
        state = "BAD" if prob_bad >= self.threshold else "GOOD"

        return {
            "state": state,
            "probability": prob_bad
        }
