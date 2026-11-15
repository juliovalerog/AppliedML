# Session 8 — A first local prediction service

**Preview of next semester:** This lesson is available in advance of the dedicated **MLOps & Model Deployment** course. Here we serve one trained model locally and obtain a prediction over HTTP. Production deployment, infrastructure, automation and operational monitoring belong to that course.

**Objective:** A colleague with a customer on the phone can send raw Telco fields to `POST /predict` and receive a churn probability. Read [notebook 14](../notebooks/14_telco_local_fastapi.ipynb) for the guided walkthrough. No cloud account is needed.

## What travels out of the notebook?

Keep the whole fitted Pipeline: the two row-wise logs, numeric medians and scales, one-hot categories, and Logistic Regression. Recomputing a median on today's requests would change the input meaning for the trained model. Calling `fit` inside a prediction route would teach preprocessing from the request instead of reusing training statistics.

We reuse Session 6's finalist: original inputs plus `log1p(tenure)` and `log1p(TotalCharges)`, with raw probabilities. Session 6 selected this probability method on validation. The training command refits that fixed recipe on the same 5,634 development customers (20% stratified holdout, seed 42). It does not tune or select on holdout. This classroom holdout has already been inspected in the course; serving is not fresh performance evidence.

[`pipeline.py`](pipeline.py) puts `add_logs` in an importable module. A saved function refers to its module; a function defined only in a notebook's `__main__` can fail to load in another process. Keep the `serving` package and use the module commands below from the repository root.

## 1. Install the dependencies

Follow the root [environment setup](../README.md#set-up-your-python-environment), including Python 3.12 and activation of `.venv`. In the root folder containing `README.md`, run:

```bash
python -m pip install -r serving/requirements.txt
python -m pip check
```

This includes the course requirements and adds FastAPI, Pydantic and Uvicorn. Joblib is declared directly because we use it directly, although scikit-learn already installs it. Interpretability uses the linear SHAP formula and XGBoost's built-in TreeSHAP, so no separate SHAP package is required.

## 2. Prepare the model

```bash
python -m serving.train
```

The command reads `data/telco.csv`, fits the Pipeline and writes `serving/artifacts/telco.joblib`. The adjacent `telco.json` records the model label, training count, dataset hash, Python version and model-library versions. Generated artifacts are ignored by Git. Keep the same environment when saving and loading; regenerate locally after a library change. Joblib files can execute Python during loading: load only artifacts you trust. See [scikit-learn's persistence guidance](https://scikit-learn.org/stable/model_persistence.html).

`telco-s6-v1` names this teaching recipe. The sidecar records the actual local build; it is not a production artifact registry.

## 3. Start the server

In the same terminal, with `.venv` active and the repository root as the working directory:

```bash
python -m uvicorn serving.app:app --host 127.0.0.1 --port 8000
```

Wait for `Application startup complete`. Leave this terminal running. FastAPI's [`lifespan`](https://fastapi.tiangolo.com/advanced/events/) loads the artifact **once per server process**, before accepting requests. Each call uses `predict_proba`, without fitting. If the artifact is missing, startup tells you to run `python -m serving.train`.

Open <http://127.0.0.1:8000/docs> to inspect the schema or try the route. `GET /` is not defined; use `/docs` or `POST /predict`. If port 8000 is busy, stop the other local service or use a free port and change the client URL accordingly. Stop this server with `Ctrl+C` when finished.

## 4. Send a customer

Open a **second terminal**, return to the repository root and activate the same `.venv`. The full body is already stored in [`customer.json`](customer.json):

```json
{
  "gender": "Male",
  "SeniorCitizen": 1,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 23,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Bank transfer (automatic)",
  "MonthlyCharges": 34.65,
  "TotalCharges": 768.45
}
```

The client sends **raw fields**, including charges in euros and tenure in months. It sends neither `customerID`, `Churn`, logs nor one-hot columns. The fitted Pipeline owns those transformations.

The Python client works on Windows, macOS and Linux without extra packages:

```bash
python -m serving.client
```

Alternatively, in macOS/Linux:

```bash
curl -i -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @serving/customer.json
```

In Windows PowerShell, use `curl.exe` to avoid the older PowerShell `curl` alias:

```powershell
curl.exe -i -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@serving/customer.json"
```

Expected status: **HTTP 200**. The response is approximately:

```json
{
  "churn_probability": 0.3598182,
  "model_version": "telco-s6-v1"
}
```

This is about **36% estimated churn risk**, not a certainty and not a measured effect of an offer. The final decimals may vary with numerical-library versions. The automated check below compares HTTP with the actual saved pipeline in your environment, as well as checking proximity to class.

The Session 6 policy considers an offer at risk **at least 0.20**, under its margin, discount and response assumptions. This customer crosses that cutoff; at 0.40 the same score would not trigger an offer. A 20% risk cutoff is not a limit of 20% of customers contacted. This endpoint returns the risk; the caller applies the business policy. Changing offer cost or capacity may require a different policy even if the probability stays the same.

## 5. Check what validation does

[`app.py`](app.py) declares all 19 raw fields with [Pydantic](https://docs.pydantic.dev/latest/concepts/strict_mode/). FastAPI [validates the request body](https://fastapi.tiangolo.com/tutorial/body/) before entering `predict`:

| Request | Result | Reading |
| --- | --- | --- |
| Full `customer.json` | 200 | Score the validated customer. |
| Remove `Contract` | 422 | A required field is missing; no prediction is made. |
| Send `"34.65"` as `MonthlyCharges` | 422 | Strict validation rejects a numeric string. Send the JSON number `34.65`. |
| Set `Contract` to `"Flex 2027"` | 422 | The category is outside the agreed input contract. |
| Set `TotalCharges` to `null` | 200 | Unknown charge becomes `NaN`, then the training median fills it. |

All fields must be present, including nullable `TotalCharges`. Extra fields are rejected. Tenure is an integer between 0 and 72; monthly charges are finite numbers from 0 to 200 euros and total charges from 0 to 20,000 euros. These are explicit guardrails for this classroom population, not universal Telco rules. Extending the service to new populations requires reviewing them. The model's encoder can ignore unknown categories, but this service intentionally rejects categories outside its documented contract.

For the missing contract, the real error body includes this entry (other fields such as the supplied input may also appear):

```json
{
  "detail": [
    {"type": "missing", "loc": ["body", "Contract"], "msg": "Field required"}
  ]
}
```

A valid number can still mean the wrong thing: 100 cents and 100 euros have the same JSON type. This small schema does not check cross-field service consistency or establish that the source data are correct.

With Uvicorn still running, execute the repeatable HTTP check:

```bash
python -m serving.check_api
```

It checks a valid prediction against a separately loaded pipeline, the classroom example within 0.001, a missing required field, wrong type, unknown category, and fitted imputation for an explicit null. It exits with an error if any expectation fails. It uses real HTTP to the running service.

## What would we watch next semester?

The Session 8 lesson separates three clocks. **Today**, inspect request validation, latency, failures, missingness and input distributions. **After outcomes mature**, inspect ranking, calibration and detection by contract group, accounting for who received an offer. **With a randomized comparison**, measure whether the policy improves retention and net value.

Data drift means the input distribution changes, for example more monthly contracts. Concept drift means comparable profiles have a different relationship with churn; input drift alone cannot establish it. The lesson's two-sample KS test compares cumulative numeric distributions, and PSI summarizes changes in shares within fixed bins. KS significance depends on sample size and its assumptions; PSI depends on bins and treatment of empty cells. Neither establishes business harm or automatically calls for retraining. A mix shift can change aggregate scores even if within-group behavior holds.

A future service and CRM could link prediction and action events with a request ID, model and policy versions, timestamps and permitted monitoring fields. Decide what information to retain and protect before logging customer records. This local example leaves operational logging, drift jobs, alerting, retraining, production hosting, infrastructure and automation to the dedicated course. First investigate whether an alert comes from a broken feed, a changed customer mix, changed risks or changed offer economics.

**Question:** If the offer stops paying, what evidence would distinguish a data problem, a probability problem and a policy problem?
