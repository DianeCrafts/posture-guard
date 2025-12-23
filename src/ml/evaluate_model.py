import warnings
warnings.simplefilter("ignore", UserWarning)
import pandas as pd
import joblib
from pathlib import Path

from src.decision.posture_classifier import PostureClassifier
from src.ml.ml_classifier import MLPostureClassifier

from sklearn.metrics import classification_report, confusion_matrix


DATA_DIR = Path("data")
MODEL_PATH = Path("models/posture_model.pkl")


def load_data():
    csv_files = sorted(DATA_DIR.glob("posture_session_*.csv"))
    if not csv_files:
        raise FileNotFoundError("No posture CSV files found.")

    df = pd.read_csv(csv_files[-1])

    # Keep only labeled rows
    df = df[df["label"].isin(["GOOD", "BAD"])]

    # Encode labels
    df["label_bin"] = df["label"].map({"GOOD": 0, "BAD": 1})

    return df


def evaluate_rule_based(df):
    classifier = PostureClassifier()

    preds = []
    for _, row in df.iterrows():
        metrics = {
            "neck_angle_deg": row["neck_angle_deg"],
            "shoulder_diff": row["shoulder_diff"],
        }
        result = classifier.classify(metrics)
        preds.append(1 if result["state"] == "BAD" else 0)

    return preds


def evaluate_ml_based(df):
    classifier = MLPostureClassifier(model_path=MODEL_PATH)

    preds = []
    for _, row in df.iterrows():
        metrics = {
            "neck_angle_deg": row["neck_angle_deg"],
            "shoulder_diff": row["shoulder_diff"],
        }
        result = classifier.classify(metrics)
        preds.append(1 if result["state"] == "BAD" else 0)

    return preds


def main():
    df = load_data()
    y_true = df["label_bin"].tolist()

    print("\n===== RULE-BASED CLASSIFIER =====")
    y_rule = evaluate_rule_based(df)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_rule))
    print(classification_report(y_true, y_rule, target_names=["GOOD", "BAD"]))

    print("\n===== ML-BASED CLASSIFIER =====")
    y_ml = evaluate_ml_based(df)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_ml))
    print(classification_report(y_true, y_ml, target_names=["GOOD", "BAD"]))

    # Agreement analysis
    agreement = sum(r == m for r, m in zip(y_rule, y_ml)) / len(y_rule)
    print(f"\nRule vs ML agreement rate: {agreement:.2%}")


if __name__ == "__main__":
    main()
