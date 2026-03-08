"""Tests for data preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.preprocess import preprocess


@pytest.fixture
def sample_data(tmp_path):
    """Create a minimal steel plates dataset for testing."""
    np.random.seed(42)
    n = 100
    feature_names = [
        "X_Minimum",
        "X_Maximum",
        "Y_Minimum",
        "Y_Maximum",
        "Pixels_Areas",
        "X_Perimeter",
        "Y_Perimeter",
        "Sum_of_Luminosity",
        "Minimum_of_Luminosity",
        "Maximum_of_Luminosity",
        "Length_of_Conveyer",
        "TypeOfSteel_A300",
        "TypeOfSteel_A400",
        "Steel_Plate_Thickness",
        "Edges_Index",
        "Empty_Index",
        "Square_Index",
        "Outside_X_Index",
        "Edges_X_Index",
        "Edges_Y_Index",
        "Outside_Global_Index",
        "LogOfAreas",
        "Log_X_Index",
        "Log_Y_Index",
        "Orientation_Index",
        "Luminosity_Index",
        "SigmoidOfAreas",
    ]
    data = {name: np.random.randn(n) for name in feature_names}
    data["fault_type"] = np.random.choice(["Pastry", "Z_Scratch", "K_Scratch", "Bumps"], size=n)
    df = pd.DataFrame(data)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    raw_path = raw_dir / "steel_plates_faults.csv"
    df.to_csv(raw_path, index=False)

    config = {
        "data": {
            "raw_path": str(raw_path),
            "processed_path": str(processed_dir),
            "test_size": 0.2,
            "random_state": 42,
        },
        "model": {
            "type": "random_forest",
            "params": {"n_estimators": 10, "random_state": 42},
        },
        "mlflow": {"experiment_name": "test", "tracking_uri": "sqlite:///test.db"},
        "serving": {
            "host": "0.0.0.0",
            "port": 8000,
            "model_path": "models/model.joblib",
        },
        "monitoring": {
            "reference_data_path": str(processed_dir / "train.csv"),
            "drift_threshold": 0.05,
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return tmp_path, str(config_path)


def test_preprocess_creates_splits(sample_data):
    tmp_path, config_path = sample_data
    result = preprocess(config_path)

    assert result["X_train"] is not None
    assert result["X_test"] is not None
    assert len(result["X_train"]) == 80
    assert len(result["X_test"]) == 20


def test_preprocess_saves_artifacts(sample_data):
    tmp_path, config_path = sample_data
    preprocess(config_path)

    processed_dir = tmp_path / "processed"
    assert (processed_dir / "X_train.csv").exists()
    assert (processed_dir / "X_test.csv").exists()
    assert (processed_dir / "y_train.csv").exists()
    assert (processed_dir / "y_test.csv").exists()


def test_preprocess_scales_features(sample_data):
    tmp_path, config_path = sample_data
    result = preprocess(config_path)

    # Scaled training data should have mean ~0 and std ~1
    means = result["X_train"].mean()
    assert all(abs(m) < 0.2 for m in means), "Scaled means should be near zero"
