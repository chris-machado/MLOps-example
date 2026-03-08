"""Model training with MLflow experiment tracking."""

import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score


def load_config(config_path: str = "configs/model_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(config_path: str = "configs/model_config.yaml") -> str:
    """Train the model and log to MLflow.

    Returns:
        The MLflow run ID.
    """
    config = load_config(config_path)
    processed_path = config["data"]["processed_path"]
    model_params = config["model"]["params"]
    experiment_name = config["mlflow"]["experiment_name"]

    # Load processed data
    X_train = pd.read_csv(os.path.join(processed_path, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_path, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_path, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(processed_path, "y_test.csv")).squeeze()
    label_encoder = joblib.load("models/label_encoder.joblib")

    # Set up MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params(model_params)
        mlflow.log_param("model_type", config["model"]["type"])
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        # Train
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        f1_macro = f1_score(y_test, y_pred, average="macro")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_weighted", f1_weighted)
        mlflow.log_metric("f1_macro", f1_macro)

        # Log classification report
        class_names = label_encoder.classes_.tolist()
        report = classification_report(y_test, y_pred, target_names=class_names)
        print(report)
        mlflow.log_text(report, "classification_report.txt")

        # Log feature importances
        importances = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        mlflow.log_text(importances.to_csv(index=False), "feature_importances.csv")

        # Log model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="steel-fault-classifier",
        )

        # Save model locally for serving
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

        print(f"Run ID: {run.info.run_id}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 (weighted): {f1_weighted:.4f}")
        print(f"F1 (macro): {f1_macro:.4f}")

        return run.info.run_id


if __name__ == "__main__":
    train()
