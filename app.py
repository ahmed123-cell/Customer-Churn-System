"""
app.py

FastAPI service for the Telco Customer Churn model.

Serves the ONNX model exported by save_model.py, using the same fitted
StandardScaler and feature ordering it was trained with (also written by
save_model.py), and provides:

    - Pydantic request validation: malformed or out-of-range input is
      rejected before it ever reaches preprocessing or the model.
    - GET /health: liveness/readiness check.
    - The model, scaler, and feature order are loaded exactly ONCE at
      startup (via FastAPI's lifespan handler) and reused across every
      request — never reloaded per-call.
    - Async request handlers, with the CPU-bound ONNX inference offloaded
      to a worker thread (asyncio.to_thread) so it never blocks the event
      loop while other requests are waiting.
    - A typed Pydantic response schema.
    - Auto-generated interactive docs, built into FastAPI, at /docs
      (Swagger UI) and /redoc.

Requires:
    pip install fastapi uvicorn pydantic onnxruntime joblib pyyaml pandas

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Prerequisite (produces the three artifacts this app loads):
    python save_model.py --data data/Telco-Customer-Churn.csv --model random_forest
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from preprocessing import preprocess_data


def load_app_config(path: str = "configs/app_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


_APP_CFG = load_app_config()


# ---------------------------------------------------------------------------
# Model state — populated once at startup, read (never mutated) per-request.
# ---------------------------------------------------------------------------

class ModelState:
    session: ort.InferenceSession | None = None
    scaler = None
    feature_names: list[str] = []
    input_name: str = ""
    model_path: str = ""


model_state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ONNX model, scaler, and feature ordering exactly once when
    the process starts, and hold them in `model_state` for its whole
    lifetime. Runs before the app accepts any traffic, and again only if
    the process itself restarts — never on a per-request basis.
    """
    model_state.model_path = _APP_CFG["model_path"]
    model_state.session = ort.InferenceSession(
        model_state.model_path, providers=["CPUExecutionProvider"]
    )
    model_state.input_name = model_state.session.get_inputs()[0].name
    model_state.scaler = joblib.load(_APP_CFG["scaler_path"])
    with open(_APP_CFG["feature_names_path"]) as f:
        model_state.feature_names = json.load(f)

    print(f"Loaded model '{model_state.model_path}' "
          f"({len(model_state.feature_names)} features)")

    yield

    # No teardown needed: the ONNX Runtime session and in-memory scaler
    # hold no external connections and are released automatically on exit.


app = FastAPI(
    title="Telco Customer Churn API",
    description="Predicts whether a customer is likely to churn from their "
                 "account and service attributes.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

YesNo = Literal["Yes", "No"]
ServiceOption = Literal["No", "No internet service", "Yes"]

_EXAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 845.50,
}


class CustomerData(BaseModel):
    """One customer's raw account/service attributes, in the same shape as
    a row of the original Telco Customer Churn CSV (minus customerID and
    Churn). Every categorical field is restricted to the exact categories
    the model was trained on (see configs/data_config.yaml), so an
    unrecognized value is rejected with a 422 instead of silently reaching
    the model.
    """

    model_config = ConfigDict(json_schema_extra={"example": _EXAMPLE_CUSTOMER})

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100, description="Months as a customer")
    PhoneService: YesNo
    MultipleLines: Literal["No", "No phone service", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: ServiceOption
    OnlineBackup: ServiceOption
    DeviceProtection: ServiceOption
    TechSupport: ServiceOption
    StreamingTV: ServiceOption
    StreamingMovies: ServiceOption
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]
    MonthlyCharges: float = Field(ge=0, description="Current monthly charge amount")
    TotalCharges: float = Field(ge=0, description="Total amount charged to date")


class PredictionResponse(BaseModel):
    """Model output for a single customer."""

    churn_prediction: Literal[0, 1] = Field(
        description="1 = predicted to churn, 0 = predicted to stay"
    )
    churn_probability: float = Field(ge=0, le=1, description="Predicted probability of churn")
    model_path: str


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_path: str
    n_features: int


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_inference(df_row: pd.DataFrame) -> tuple[int, float]:
    """Preprocess one row with the already-fitted scaler/feature layout,
    run it through the ONNX session, and return (predicted_label,
    churn_probability).

    This is synchronous and CPU-bound (both the pandas preprocessing and
    the ONNX Runtime call release the GIL only partially, if at all), so
    the calling endpoint runs it via asyncio.to_thread rather than
    awaiting it directly.
    """
    X, _, _, feature_names = preprocess_data(
        df_row, scaler=model_state.scaler, fit_scaler=False
    )

    if feature_names != model_state.feature_names:
        # Shouldn't happen with a valid CustomerData payload and matching
        # artifacts, but fail loudly rather than feed the model a
        # misaligned feature vector.
        raise HTTPException(
            status_code=500,
            detail="Feature layout mismatch between the request and the loaded model.",
        )

    X = np.asarray(X, dtype=np.float32)
    outputs = model_state.session.run(None, {model_state.input_name: X})

    label = int(np.asarray(outputs[0]).reshape(-1)[0])

    # outputs[1] is a plain probability tensor if the model was exported
    # with zipmap=False (save_model.py's default), or a list of
    # {class: prob} dicts if exported with zipmap=True — handle both.
    proba_output = outputs[1]
    if isinstance(proba_output, list):
        probability = float(proba_output[0][1])
    else:
        probability = float(np.asarray(proba_output).reshape(-1, 2)[0][1])

    return label, probability


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    """Liveness/readiness check — confirms the model is loaded and reports
    which artifact is currently serving traffic.
    """
    loaded = model_state.session is not None
    return HealthResponse(
        status="ok" if loaded else "unavailable",
        model_loaded=loaded,
        model_path=model_state.model_path,
        n_features=len(model_state.feature_names),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(customer: CustomerData) -> PredictionResponse:
    """Predict churn for a single customer. `customer` is already validated
    against `CustomerData`'s field types/ranges/categories by the time this
    handler runs — invalid payloads never reach this code.
    """
    if model_state.session is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    df_row = pd.DataFrame([customer.model_dump()])
    label, probability = await asyncio.to_thread(_run_inference, df_row)

    return PredictionResponse(
        churn_prediction=label,
        churn_probability=round(probability, 4),
        model_path=model_state.model_path,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=_APP_CFG["host"], port=_APP_CFG["port"], reload=True)