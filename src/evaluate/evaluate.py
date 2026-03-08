"""Model evaluation and validation checks."""

import os
import sys

import pandas as pd
import yaml
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report


def load_config(config_path: str = "configs/model_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Minimum performance thresholds for promotion
ACCURACY_THRESHOLD = 0.70
F1_THRESHOLD = 0.65


def evaluate(config_path: str = "configs/model_config.yaml") -> dict:
    """Evaluate the trained model against quality gates.

    Returns:
        Dict of metrics. Exits with code 1 if thresholds are not met.
    """
    config = load_config(config_path)
    processed_path = config["data"]["processed_path"]

    X_test = pd.read_csv(os.path.join(processed_path, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_path, "y_test.csv")).squeeze()
    label_encoder = joblib.load("models/label_encoder.joblib")
    model = joblib.load("models/model.joblib")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    class_names = label_encoder.classes_.tolist()
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(report)

    # Quality gates
    passed = True
    if accuracy < ACCURACY_THRESHOLD:
        print(f"FAIL: Accuracy {accuracy:.4f} below threshold {ACCURACY_THRESHOLD}")
        passed = False
    else:
        print(f"PASS: Accuracy {accuracy:.4f} >= {ACCURACY_THRESHOLD}")

    if f1_weighted < F1_THRESHOLD:
        print(f"FAIL: F1 {f1_weighted:.4f} below threshold {F1_THRESHOLD}")
        passed = False
    else:
        print(f"PASS: F1 (weighted) {f1_weighted:.4f} >= {F1_THRESHOLD}")

    if not passed:
        print("Model did not meet quality gates. Blocking deployment.")
        sys.exit(1)

    print("All quality gates passed.")
    return {"accuracy": accuracy, "f1_weighted": f1_weighted}


if __name__ == "__main__":
    evaluate()
