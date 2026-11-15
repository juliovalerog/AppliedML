"""Check the running HTTP service: python -m serving.check_api."""

import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd

from serving.train import MODEL_PATH, MODEL_VERSION


def post(payload):
    request = Request(
        "http://127.0.0.1:8000/predict", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        response = urlopen(request, timeout=10)
    except HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response)


def main():
    customer = json.loads(Path(__file__).with_name("customer.json").read_text())
    pipeline = joblib.load(MODEL_PATH)
    expected = float(pipeline.predict_proba(pd.DataFrame([customer]))[0, 1])
    status, body = post(customer)
    assert status == 200, (status, body)
    assert body["model_version"] == MODEL_VERSION
    np.testing.assert_allclose(body["churn_probability"], expected, atol=1e-12, rtol=0)
    # Compare approximately with class; the direct pipeline above is the exact check.
    np.testing.assert_allclose(expected, .3598181970496265, atol=1e-3, rtol=0)
    print(f"Valid customer: HTTP 200, probability {expected:.8f}; matches saved pipeline")

    missing_contract = customer.copy()
    del missing_contract["Contract"]
    status, body = post(missing_contract)
    assert status == 422, (status, body)
    assert any(e["loc"] == ["body", "Contract"] and e["type"] == "missing"
               for e in body["detail"])
    print("Missing Contract: HTTP 422, required field identified")

    # These requests are the same failure examples shown in the Session 8 lesson.
    for field, value in [("MonthlyCharges", "34.65"), ("Contract", "Flex 2027")]:
        invalid = customer.copy()
        invalid[field] = value
        status, body = post(invalid)
        assert status == 422, (status, body)
        assert any(e["loc"] == ["body", field] for e in body["detail"])
        print(f"Invalid {field}: HTTP 422")

    # Unknown total charges follow the same fitted imputation path as training.
    missing_charge = customer.copy()
    missing_charge["TotalCharges"] = None
    status, body = post(missing_charge)
    row = pd.DataFrame([missing_charge])
    row["TotalCharges"] = pd.to_numeric(row["TotalCharges"], errors="coerce")
    expected_missing = pipeline.predict_proba(row)[0, 1]
    assert status == 200, (status, body)
    np.testing.assert_allclose(body["churn_probability"], expected_missing, atol=1e-12, rtol=0)
    print("Null TotalCharges: HTTP 200, matches fitted imputation")


if __name__ == "__main__":
    main()
