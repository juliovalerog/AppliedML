"""Keep the training recipe and its log transform importable after saving."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]


def add_logs(frame):
    # Missing charges stay missing until the fitted imputer replaces them.
    result = frame.copy()
    result["log_tenure"] = np.log1p(result["tenure"])
    result["log_total"] = np.log1p(result["TotalCharges"])
    return result


def build_model(columns):
    """Rebuild Session 6's uncorrected Logistic Regression finalist."""
    categorical = []
    for column in columns:
        if column not in NUMERIC:
            categorical.append(column)
    numeric_preprocessing = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    preprocessing = ColumnTransformer([
        ("num", numeric_preprocessing, NUMERIC + ["log_tenure", "log_total"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    # The saved object includes the formulas, training medians, scales and categories.
    return Pipeline([
        ("features", FunctionTransformer(add_logs)),
        ("preprocess", preprocessing),
        ("model", LogisticRegression(C=1, solver="lbfgs", max_iter=2000)),
    ])
