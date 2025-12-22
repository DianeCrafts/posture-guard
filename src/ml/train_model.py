import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def load_data():
    # Load the most recent CSV
    csv_files = sorted(DATA_DIR.glob("posture_session_*.csv"))
    if not csv_files:
        raise FileNotFoundError("No posture CSV files found.")

    df = pd.read_csv(csv_files[-1])

    # Keep only labeled data
    df = df[df["label"].isin(["GOOD", "BAD"])]

    # Encode labels
    df["label"] = df["label"].map({"GOOD": 0, "BAD": 1})

    return df


def main():
    df = load_data()

    X = df[["neck_angle_deg", "shoulder_diff"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["GOOD", "BAD"]))

    # Save model
    model_path = MODEL_DIR / "posture_model.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

    # Show learned importance
    print("\nFeature importance (coefficients):")
    for feature, coef in zip(X.columns, model.coef_[0]):
        print(f"  {feature}: {coef:.3f}")


if __name__ == "__main__":
    main()
