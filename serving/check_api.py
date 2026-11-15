"""Check the running HTTP service: python -m serving.check_api."""

# Encode request bodies and decode successful predictions or validation-error details.
import json
# Read the same stored customer JSON used in the student walkthrough.
from pathlib import Path
# Capture an HTTP validation response while retaining access to its status and JSON body.
from urllib.error import HTTPError
# Exercise the real HTTP endpoint using Python's standard library.
from urllib.request import Request, urlopen

# Load a separate copy of the artifact for an independent direct-scoring reference.
import joblib
# Compare floating-point results with explicit numerical tolerances.
import numpy as np
# Build exactly the same labelled customer row expected by the fitted Pipeline.
import pandas as pd

# Keep the direct-scoring check tied to the same artifact and model label as the server.
from serving.train import MODEL_PATH, MODEL_VERSION


def post(payload):
    # Serialize the supplied dictionary to JSON bytes and send it to the local POST route.
    request = Request(
        "http://127.0.0.1:8000/predict", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    # Handle both success and validation failure through the same response-reading steps.
    try:
        # Bound waiting time so a stopped or unresponsive local server does not hang the exercise.
        response = urlopen(request, timeout=10)
    except HTTPError as error:
        # HTTP 422 arrives as an exception object but still contains a readable response body.
        response = error
    # Close the stream after reading it, including streams attached to error responses.
    with response:
        # Return status and decoded JSON separately so callers can verify both.
        return response.status, json.load(response)


def main():
    # Start from the full valid class customer before constructing incomplete or invalid variants.
    customer = json.loads(Path(__file__).with_name("customer.json").read_text())
    # Load outside the service process to check HTTP scoring against the actual saved model.
    pipeline = joblib.load(MODEL_PATH)
    # Wrap the customer as one raw row and take the probability of the churn class.
    expected = float(pipeline.predict_proba(pd.DataFrame([customer]))[0, 1])
    # Send the same unaltered customer over HTTP to the running Uvicorn server.
    status, body = post(customer)
    # Require a successful prediction response; include the body if the assertion fails.
    assert status == 200, (status, body)
    # Check model identity separately from the numerical result.
    assert body["model_version"] == MODEL_VERSION
    # Permit only tiny floating-point differences between HTTP and direct artifact scoring.
    np.testing.assert_allclose(body["churn_probability"], expected, atol=1e-12, rtol=0)
    # Compare approximately with class; the direct pipeline above is the exact check.
    np.testing.assert_allclose(expected, .3598181970496265, atol=1e-3, rtol=0)
    # Expose the matched score rather than silently finishing the successful case.
    print(f"Valid customer: HTTP 200, probability {expected:.8f}; matches saved pipeline")

    # Make a copy so the valid reference remains intact for later tests.
    missing_contract = customer.copy()
    # Remove only the required contract field to isolate missing-field validation.
    del missing_contract["Contract"]
    # Submit the incomplete customer through the real request-validation boundary.
    status, body = post(missing_contract)
    # Require a validation error rather than an unexpected prediction or server failure.
    assert status == 422, (status, body)
    # Confirm that the error specifically identifies a missing Contract in the JSON body.
    assert any(e["loc"] == ["body", "Contract"] and e["type"] == "missing"
               for e in body["detail"])
    # Report which required-field check passed for the student running this command.
    print("Missing Contract: HTTP 422, required field identified")

    # These requests are the same failure examples shown in the Session 8 lesson.
    for field, value in [("MonthlyCharges", "34.65"), ("Contract", "Flex 2027")]:
        # Restart from a complete valid customer so errors do not accumulate across cases.
        invalid = customer.copy()
        # Introduce either the wrong numeric type or an unknown contract category.
        invalid[field] = value
        # Let the service validate the altered value before the model can score it.
        status, body = post(invalid)
        # Both changes must be rejected at the input boundary.
        assert status == 422, (status, body)
        # Check that the error points to the field actually altered in this iteration.
        assert any(e["loc"] == ["body", field] for e in body["detail"])
        # Identify the rejected field so each validation example is visible in the output.
        print(f"Invalid {field}: HTTP 422")

    # Unknown total charges follow the same fitted imputation path as training.
    missing_charge = customer.copy()
    # Explicit None serializes to JSON null; it is different from omitting a required field.
    missing_charge["TotalCharges"] = None
    # Ask the service to score this supported missing-value case.
    status, body = post(missing_charge)
    # Rebuild the same raw row for direct scoring outside the service.
    row = pd.DataFrame([missing_charge])
    # Turn the all-missing single charge column into numeric NaN, matching the route's preparation.
    row["TotalCharges"] = pd.to_numeric(row["TotalCharges"], errors="coerce")
    # Let the fitted training median, not a statistic from this request, fill the missing charge.
    expected_missing = pipeline.predict_proba(row)[0, 1]
    # Explicit null is allowed here, so this customer must receive a successful prediction.
    assert status == 200, (status, body)
    # Check that HTTP and direct scoring use the same fitted imputation behavior.
    np.testing.assert_allclose(body["churn_probability"], expected_missing, atol=1e-12, rtol=0)
    # Confirm that the missing-value example matched, not merely that its request was accepted.
    print("Null TotalCharges: HTTP 200, matches fitted imputation")


# Run the checks when requested as a command; importing the helper must not send requests.
if __name__ == "__main__":
    main()
