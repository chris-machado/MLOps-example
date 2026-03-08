"""Fetch the UCI Steel Plates Faults dataset."""

import os

import pandas as pd
from ucimlrepo import fetch_ucirepo


def fetch_steel_plates_data(
    output_path: str = "data/raw/steel_plates_faults.csv",
) -> str:
    """Download the Steel Plates Faults dataset from UCI ML Repository.

    Dataset: https://archive.ics.uci.edu/dataset/198/steel+plates+faults
    1,941 instances | 27 features | 7 fault types (multiclass)

    Manufacturing context: Classifying surface defects on steel plates
    using measurements from an image-based inspection system.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dataset = fetch_ucirepo(id=198)

    features = dataset.data.features
    targets = dataset.data.targets

    # Targets are one-hot encoded across 7 columns — convert to single label
    fault_labels = targets.idxmax(axis=1)

    df = pd.concat([features, fault_labels.rename("fault_type")], axis=1)
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Fault distribution:\n{df['fault_type'].value_counts()}")

    return output_path


if __name__ == "__main__":
    fetch_steel_plates_data()
