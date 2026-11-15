"""Run from the repository root: python -m serving.train."""

# Check that training reads the same CSV bytes used in the classroom experiment.
import hashlib
# Write readable artifact metadata alongside the binary model.
import json
# Record the Python version needed to reproduce this local build.
import platform
# Read installed package versions rather than relying on the broad requirements ranges.
from importlib.metadata import version
# Resolve data and output paths from this module's location.
from pathlib import Path

# Save the complete fitted Python Pipeline, including learned preparation statistics.
import joblib
# Load customer data and parse the raw charge column.
import pandas as pd
# Recreate the course's development/holdout boundary before any fitting.
from sklearn.model_selection import train_test_split

# Import the recipe from a stable module so its saved log transform can be loaded elsewhere.
from serving.pipeline import build_model


# Move from serving/train.py to the repository root independently of the caller's directory.
ROOT = Path(__file__).resolve().parents[1]
# Store generated files together in the directory excluded from Git.
MODEL_PATH = ROOT / "serving" / "artifacts" / "telco.joblib"
# Identify the fixed teaching recipe; the metadata below records each local environment.
MODEL_VERSION = "telco-s6-v1"


def main():
    # Use the dataset committed with the course instead of requiring an external download.
    data_path = ROOT / "data" / "telco.csv"
    # Hash the original bytes before parsing so the dataset identity is unambiguous.
    fingerprint = hashlib.sha256(data_path.read_bytes()).hexdigest()
    # Stop if the source changed rather than silently claiming to reproduce the same model.
    assert fingerprint == "88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a"
    # Read one row per customer while retaining the original raw column names.
    data = pd.read_csv(data_path)
    # This parsing is row-wise; learned imputation remains inside the pipeline.
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    # Exclude the known outcome and customer identifier from model inputs.
    X = data.drop(columns=["Churn", "customerID"])
    # Encode observed churn as the positive class so probability column 1 means leaving.
    y = data["Churn"].eq("Yes").astype(int)
    # Retain the same 80% development set; underscore variables deliberately discard holdout here.
    X_dev, _, y_dev, _ = train_test_split(
        X, y, test_size=.20, stratify=y, random_state=42
    )
    # Session 6 already selected raw probabilities and the model recipe on validation.
    # Refit on development, without selecting anything from the classroom holdout.
    pipeline = build_model(X.columns).fit(X_dev, y_dev)
    # Create the generated-artifact folder on the first run and reuse it on later runs.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Persist fitted medians, scales, categories and classifier together, not just model weights.
    joblib.dump(pipeline, MODEL_PATH)

    # Keep model identity, training support and source/environment information auditable.
    metadata = {
        "model_version": MODEL_VERSION,
        "training_customers": len(X_dev),
        "dataset_sha256": fingerprint,
        "python": platform.python_version(),
        "packages": {},
    }
    # Record the libraries needed to load and reproduce the fitted object in another process.
    for package in ["scikit-learn", "numpy", "pandas", "scipy", "joblib"]:
        # Store actual installed versions; these can differ from the instructor's environment.
        metadata["packages"][package] = version(package)
    # Save a human-readable JSON sidecar with the same stem as the binary artifact.
    MODEL_PATH.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    # Confirm which artifact was written and how many customers supplied its fitted statistics.
    print(f"Saved {MODEL_PATH.name}: {len(X_dev):,} development customers")
    # Show provenance immediately so students can inspect it before starting the API.
    print(json.dumps(metadata, indent=2))


# Train only when this module is run as a command; importing its constants must not retrain.
if __name__ == "__main__":
    main()
