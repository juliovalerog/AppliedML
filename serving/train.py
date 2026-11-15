"""Run from the repository root: python -m serving.train."""

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from serving.pipeline import build_model


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "serving" / "artifacts" / "telco.joblib"
MODEL_VERSION = "telco-s6-v1"


def main():
    data_path = ROOT / "data" / "telco.csv"
    fingerprint = hashlib.sha256(data_path.read_bytes()).hexdigest()
    assert fingerprint == "88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a"
    data = pd.read_csv(data_path)
    # This parsing is row-wise; learned imputation remains inside the pipeline.
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    X = data.drop(columns=["Churn", "customerID"])
    y = data["Churn"].eq("Yes").astype(int)
    X_dev, _, y_dev, _ = train_test_split(
        X, y, test_size=.20, stratify=y, random_state=42
    )
    # Session 6 already selected raw probabilities and the model recipe on validation.
    # Refit on development, without selecting anything from the classroom holdout.
    pipeline = build_model(X.columns).fit(X_dev, y_dev)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    metadata = {
        "model_version": MODEL_VERSION,
        "training_customers": len(X_dev),
        "dataset_sha256": fingerprint,
        "python": platform.python_version(),
        "packages": {},
    }
    for package in ["scikit-learn", "numpy", "pandas", "scipy", "joblib"]:
        metadata["packages"][package] = version(package)
    MODEL_PATH.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {MODEL_PATH.name}: {len(X_dev):,} development customers")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
