"""Tests for the model serving API."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create a test client with mock model artifacts."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    # Create mock artifacts
    n_features = 27
    classes = [
        "Bumps",
        "Dirtiness",
        "K_Scratch",
        "Other_Faults",
        "Pastry",
        "Stains",
        "Z_Scratch",
    ]

    le = LabelEncoder()
    le.fit(classes)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    scaler = StandardScaler()
    scaler.fit(pd.DataFrame(np.random.randn(50, n_features), columns=feature_names))

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X_mock = scaler.transform(np.random.randn(50, n_features))
    y_mock = np.random.choice(range(len(classes)), size=50)
    model.fit(X_mock, y_mock)

    joblib.dump(model, tmp_path / "model.joblib")
    joblib.dump(scaler, tmp_path / "scaler.joblib")
    joblib.dump(le, tmp_path / "label_encoder.joblib")

    monkeypatch.setenv("MODEL_DIR", str(tmp_path))

    # Force reimport so the module loads from tmp_path
    import importlib

    import src.serve.app as app_module

    importlib.reload(app_module)

    return TestClient(app_module.app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_valid(client):
    features = np.random.randn(27).tolist()
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert 0 <= data["confidence"] <= 1


def test_predict_wrong_feature_count(client):
    response = client.post("/predict", json={"features": [1.0, 2.0]})
    assert response.status_code == 422
