"""Run from the repository root: python -m uvicorn serving.app:app."""

# Separate one-time startup and shutdown work from per-request prediction.
from contextlib import asynccontextmanager
# Declare permitted category values without assigning them an artificial numeric order.
from typing import Literal

# Restore the fitted Pipeline instead of rebuilding preparation statistics from requests.
import joblib
# Represent a validated customer as the labelled one-row table expected by the Pipeline.
import pandas as pd
# Define the HTTP application and its prediction route.
from fastapi import FastAPI
# Describe field types, accepted ranges and validation settings for request and response bodies.
from pydantic import BaseModel, ConfigDict, Field

# Reuse the training command's artifact location and model label without invoking training.
from serving.train import MODEL_PATH, MODEL_VERSION


# Reuse the same allowed values across fields that share a binary service indicator.
YesNo = Literal["Yes", "No"]
# Distinguish no add-on from no internet connection; neither category means an unknown value.
InternetOption = Literal["Yes", "No", "No internet service"]


# FastAPI validates this complete raw-input contract before entering the prediction function.
class TelcoInput(BaseModel):
    # Reject silent text-to-number conversion, misspelled fields and non-finite charges.
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    # Keep the category spellings used during training; these declarations have no defaults.
    gender: Literal["Female", "Male"]
    # Accept only integer 0/1 values, retaining the course's stored categorical representation.
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    # Require months as an integer within the supported classroom population's range.
    tenure: int = Field(ge=0, le=72)
    # Service categories include explicit absence states rather than unknown free text.
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetOption
    OnlineBackup: InternetOption
    DeviceProtection: InternetOption
    TechSupport: InternetOption
    StreamingTV: InternetOption
    StreamingMovies: InternetOption
    # Reject a new contract label instead of silently letting an unseen category pass encoding.
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    # Accept the exact payment categories stored in the training data.
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    # Charges are JSON numbers in euros; the upper bound is a teaching guardrail, not a universal rule.
    MonthlyCharges: float = Field(ge=0, le=200)
    # Explicit null means unknown. Omitting the field still fails validation.
    TotalCharges: float | None = Field(ge=0, le=20000)


# Define the response separately: a validated input is not itself a model prediction.
class Prediction(BaseModel):
    # Ensure the returned risk is in probability units rather than log odds or a class label.
    churn_probability: float = Field(ge=0, le=1)
    # Include the model label so the caller can identify the recipe behind the result.
    model_version: str


# FastAPI runs the code before yield at startup and the code after it at shutdown.
@asynccontextmanager
async def lifespan(app):
    # Fail before accepting requests if the student has not prepared the artifact yet.
    if not MODEL_PATH.exists():
        # Provide the exact command needed to complete the missing preparation step.
        raise RuntimeError("Prepare the model first: python -m serving.train")
    # Load once per server process; every request reuses the fitted preprocessing.
    app.state.pipeline = joblib.load(MODEL_PATH)
    # Hand control back to FastAPI while retaining the fitted model for incoming requests.
    yield
    # Release the application's reference when this server process shuts down.
    del app.state.pipeline


# Attach startup loading to the application that Uvicorn imports and serves.
app = FastAPI(title="Telco churn: local course example", lifespan=lifespan)


# Accept customer JSON through POST and validate the returned body against Prediction.
@app.post("/predict", response_model=Prediction)
def predict(customer: TelcoInput):
    # model_dump returns validated raw fields; the list wrapper makes them exactly one table row.
    row = pd.DataFrame([customer.model_dump()])
    # Ensure null becomes numeric NaN even for a single all-missing charge column.
    row["TotalCharges"] = pd.to_numeric(row["TotalCharges"], errors="coerce")
    # Reuse all fitted preparation and select row 0, positive-class probability column 1.
    probability = float(app.state.pipeline.predict_proba(row)[0, 1])
    # A risk estimate leaves the offer decision to the caller's policy.
    return Prediction(churn_probability=probability, model_version=MODEL_VERSION)
