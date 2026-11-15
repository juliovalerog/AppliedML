"""Keep the training recipe and its log transform importable after saving."""

# Calculate row-wise logarithms without learning statistics from other customers.
import numpy as np
# Send numeric measurements and categorical fields through different preparation branches.
from sklearn.compose import ColumnTransformer
# Learn numeric replacements from training data, then reuse those fitted values at prediction time.
from sklearn.impute import SimpleImputer
# Reuse the same classifier and settings selected in Session 6.
from sklearn.linear_model import LogisticRegression
# Keep all fitted preparation steps attached to the classifier when saving the model.
from sklearn.pipeline import Pipeline
# Scale measurements, encode categories and adapt the log function to the Pipeline interface.
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


# Name measured quantities explicitly; the stored 0/1 SeniorCitizen field stays categorical.
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]


def add_logs(frame):
    # Missing charges stay missing until the fitted imputer replaces them.
    result = frame.copy()
    # log1p means log(1 + x), so a customer with zero months still has a defined value.
    result["log_tenure"] = np.log1p(result["tenure"])
    # Compress accumulated charges while retaining the original charge column too.
    result["log_total"] = np.log1p(result["TotalCharges"])
    # Return the extended frame without modifying the caller's raw inputs.
    return result


def build_model(columns):
    """Rebuild Session 6's uncorrected Logistic Regression finalist."""
    # Build a category list in the original column order rather than guessing from data types.
    categorical = []
    # Inspect each raw field before adding any derived numeric columns.
    for column in columns:
        # Everything outside the three measured quantities follows the category branch.
        if column not in NUMERIC:
            # Preserve input order so fitted one-hot output positions remain reproducible.
            categorical.append(column)
    # Learn replacements first so the scaler receives complete numeric inputs.
    numeric_preprocessing = Pipeline([
        # Reuse training medians for missing raw charges and their missing log values.
        ("impute", SimpleImputer(strategy="median")),
        # Learn training means and standard deviations instead of rescaling each request.
        ("scale", StandardScaler()),
    ])
    # Route original measurements and both logs together; encode other raw fields separately.
    preprocessing = ColumnTransformer([
        ("num", numeric_preprocessing, NUMERIC + ["log_tenure", "log_total"]),
        # Keep the course encoder recipe; the API separately restricts accepted raw categories.
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    # The saved object includes the formulas, training medians, scales and categories.
    return Pipeline([
        # Run the deterministic log formulas before selecting preprocessing columns.
        ("features", FunctionTransformer(add_logs)),
        # Fit the two preparation branches when training, then only transform at prediction time.
        ("preprocess", preprocessing),
        # Keep regularization and solver fixed rather than selecting new settings during serving.
        ("model", LogisticRegression(C=1, solver="lbfgs", max_iter=2000)),
    ])
