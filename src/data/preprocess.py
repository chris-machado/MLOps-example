"""Data preprocessing and feature engineering pipeline."""

import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


def load_config(config_path: str = "configs/model_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def preprocess(config_path: str = "configs/model_config.yaml") -> dict:
    """Run the full preprocessing pipeline.

    Steps:
        1. Load raw data
        2. Handle missing values
        3. Encode target labels
        4. Scale features
        5. Train/test split
        6. Save processed artifacts
    """
    config = load_config(config_path)
    raw_path = config["data"]["raw_path"]
    processed_path = config["data"]["processed_path"]
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]

    os.makedirs(processed_path, exist_ok=True)

    # Load
    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df)} rows from {raw_path}")

    # Separate features and target
    X = df.drop(columns=["fault_type"])
    y = df["fault_type"]

    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )

    # Save processed data
    X_train_scaled.to_csv(os.path.join(processed_path, "X_train.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(processed_path, "X_test.csv"), index=False)
    pd.Series(y_train).to_csv(os.path.join(processed_path, "y_train.csv"), index=False)
    pd.Series(y_test).to_csv(os.path.join(processed_path, "y_test.csv"), index=False)

    # Save artifacts for serving
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, os.path.join("models", "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join("models", "label_encoder.joblib"))

    # Save reference data for monitoring (train set with labels)
    train_ref = X_train_scaled.copy()
    train_ref["fault_type"] = y_train
    train_ref.to_csv(os.path.join(processed_path, "train.csv"), index=False)

    print(f"Train set: {len(X_train_scaled)} rows")
    print(f"Test set: {len(X_test_scaled)} rows")
    print(f"Artifacts saved to {processed_path} and models/")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "label_encoder": label_encoder,
    }


if __name__ == "__main__":
    preprocess()
