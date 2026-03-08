"""Data drift detection using Evidently."""

import os

import pandas as pd
import yaml
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def load_config(config_path: str = "configs/model_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def detect_drift(
    current_data_path: str,
    config_path: str = "configs/model_config.yaml",
) -> dict:
    """Compare current data against the training reference data for drift.

    Args:
        current_data_path: Path to new production data CSV.
        config_path: Path to config file.

    Returns:
        Dict with drift results.
    """
    config = load_config(config_path)
    reference_path = config["monitoring"]["reference_data_path"]

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_data_path)

    # Drop target column for drift detection
    if "fault_type" in reference.columns:
        reference = reference.drop(columns=["fault_type"])
    if "fault_type" in current.columns:
        current = current.drop(columns=["fault_type"])

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    result = report.as_dict()

    # Extract summary
    drift_summary = result["metrics"][0]["result"]
    n_drifted = drift_summary["number_of_drifted_columns"]
    n_columns = drift_summary["number_of_columns"]
    dataset_drift = drift_summary["dataset_drift"]

    print(f"Columns with drift: {n_drifted}/{n_columns}")
    print(f"Dataset drift detected: {dataset_drift}")

    # Save HTML report
    os.makedirs("monitoring/reports", exist_ok=True)
    report.save_html("monitoring/reports/drift_report.html")
    print("Report saved to monitoring/reports/drift_report.html")

    return {
        "dataset_drift": dataset_drift,
        "n_drifted_columns": n_drifted,
        "n_total_columns": n_columns,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python drift_detection.py <current_data.csv>")
        sys.exit(1)

    detect_drift(sys.argv[1])
