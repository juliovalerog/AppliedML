"""Run from the repository root: python -m uvicorn serving.app:app."""

from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from serving.train import MODEL_PATH, MODEL_VERSION


YesNo = Literal["Yes", "No"]
InternetOption = Literal["Yes", "No", "No internet service"]


class TelcoInput(BaseModel):
    # Reject silent text-to-number conversion, misspelled fields and non-finite charges.
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=72)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetOption
    OnlineBackup: InternetOption
    DeviceProtection: InternetOption
    TechSupport: InternetOption
    StreamingTV: InternetOption
    StreamingMovies: InternetOption
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=200)
    # Explicit null means unknown. Omitting the field still fails validation.
    TotalCharges: float | None = Field(ge=0, le=20000)


class Prediction(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    model_version: str


@asynccontextmanager
async def lifespan(app):
    if not MODEL_PATH.exists():
        raise RuntimeError("Prepare the model first: python -m serving.train")
    # Load once per server process; every request reuses the fitted preprocessing.
    app.state.pipeline = joblib.load(MODEL_PATH)
    yield
    del app.state.pipeline


app = FastAPI(title="Telco churn: local course example", lifespan=lifespan)


@app.post("/predict", response_model=Prediction)
def predict(customer: TelcoInput):
    row = pd.DataFrame([customer.model_dump()])
    # Ensure null becomes numeric NaN even for a single all-missing charge column.
    row["TotalCharges"] = pd.to_numeric(row["TotalCharges"], errors="coerce")
    probability = float(app.state.pipeline.predict_proba(row)[0, 1])
    # A risk estimate leaves the offer decision to the caller's policy.
    return Prediction(churn_probability=probability, model_version=MODEL_VERSION)
