"""
tests/test_api.py

API-contract tests for app.py, using FastAPI's TestClient (pytest + httpx
under the hood).

These tests mock the model layer entirely — onnxruntime.InferenceSession,
joblib.load, the feature_names.json read, and preprocess_data — so they run
without any real trained model, scaler, or ONNX file on disk, and stay fast
and deterministic. What they verify is the API's own contract:
    - request validation (rejecting bad input before it reaches the model)
    - status codes for the happy path and failure paths
    - response schema shape
    - health reporting
    - that /docs and /openapi.json are actually served

Preprocessing/model correctness itself (does preprocess_data encode
columns correctly, does the exported ONNX model match the original) is out
of scope here — that belongs in its own test module, e.g. test_preprocessing.py.

Requires:
    pip install pytest httpx

Run (from the project root, so `app.py` is importable):
    pytest tests/test_api.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import mock_open

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Make the project root (one level up from tests/) importable regardless of
# where pytest is invoked from, since app.py lives there, not under tests/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_module  # noqa: E402  (import after sys.path fix, intentional)


VALID_CUSTOMER = {
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

FAKE_FEATURE_NAMES = ["feature_0", "feature_1", "feature_2"]


class _FakeOnnxInput:
    name = "input"


class FakeSession:
    """Stand-in for onnxruntime.InferenceSession. Always returns the same
    fixed (label, probability) pair, and records the last input it was
    called with in case a test wants to assert on it.
    """

    def __init__(self, label: int = 1, probability: float = 0.73):
        self._label = label
        self._probability = probability
        self.last_input = None

    def get_inputs(self):
        return [_FakeOnnxInput()]

    def run(self, output_names, input_feed):
        self.last_input = input_feed["input"]
        labels = np.array([self._label])
        probabilities = np.array([[1 - self._probability, self._probability]])
        return [labels, probabilities]


def _fake_preprocess_data(df_row, scaler=None, fit_scaler=True):
    """Stub for preprocessing.preprocess_data: skips real column encoding
    entirely and returns a dummy matrix whose feature names match whatever
    model_state.feature_names is set to, so the layout-mismatch check in
    app._run_inference passes by default.
    """
    return (
        np.zeros((len(df_row), len(FAKE_FEATURE_NAMES))),
        None,
        scaler,
        FAKE_FEATURE_NAMES,
    )


@pytest.fixture
def client(monkeypatch):
    """A TestClient whose startup (FastAPI lifespan) never touches disk:
    ort.InferenceSession, joblib.load, and the feature_names.json read are
    all mocked before the app starts, and preprocess_data is stubbed for
    every request made through this client.
    """
    fake_session = FakeSession()

    monkeypatch.setattr(app_module.ort, "InferenceSession", lambda *a, **kw: fake_session)
    monkeypatch.setattr(app_module.joblib, "load", lambda path: object())
    monkeypatch.setattr(app_module, "open", mock_open(read_data=json.dumps(FAKE_FEATURE_NAMES)), raising=False)
    monkeypatch.setattr(app_module, "preprocess_data", _fake_preprocess_data)

    with TestClient(app_module.app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_reports_loaded_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["n_features"] == len(FAKE_FEATURE_NAMES)


def test_health_reports_unavailable_when_model_missing(client):
    app_module.model_state.session = None

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["model_loaded"] is False


# ---------------------------------------------------------------------------
# /predict — happy path and failure paths
# ---------------------------------------------------------------------------

def test_predict_returns_prediction_for_valid_customer(client):
    response = client.post("/predict", json=VALID_CUSTOMER)

    assert response.status_code == 200
    body = response.json()
    assert body["churn_prediction"] == 1
    assert body["churn_probability"] == pytest.approx(0.73, abs=1e-4)
    assert "model_path" in body


def test_predict_returns_503_when_model_not_loaded(client):
    app_module.model_state.session = None

    response = client.post("/predict", json=VALID_CUSTOMER)

    assert response.status_code == 503


def test_predict_returns_500_on_feature_layout_mismatch(client, monkeypatch):
    def mismatched_preprocess_data(df_row, scaler=None, fit_scaler=True):
        return np.zeros((len(df_row), 2)), None, scaler, ["wrong_feature"]

    monkeypatch.setattr(app_module, "preprocess_data", mismatched_preprocess_data)

    response = client.post("/predict", json=VALID_CUSTOMER)

    assert response.status_code == 500


# ---------------------------------------------------------------------------
# /predict — input validation (422s)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("gender", "Other"),
        ("Partner", "Maybe"),
        ("MultipleLines", "Sometimes"),
        ("InternetService", "Satellite"),
        ("OnlineSecurity", "Unknown"),
        ("Contract", "Three year"),
        ("PaymentMethod", "Cash"),
        ("SeniorCitizen", 2),
    ],
)
def test_predict_rejects_invalid_categorical_value(client, field, bad_value):
    payload = {**VALID_CUSTOMER, field: bad_value}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("tenure", -1),
        ("tenure", 200),
        ("MonthlyCharges", -10.0),
        ("TotalCharges", -1.0),
    ],
)
def test_predict_rejects_out_of_range_numeric_value(client, field, bad_value):
    payload = {**VALID_CUSTOMER, field: bad_value}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_missing_required_field(client):
    payload = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_wrong_type(client):
    payload = {**VALID_CUSTOMER, "tenure": "twelve"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Auto-generated docs
# ---------------------------------------------------------------------------

def test_openapi_schema_lists_both_routes(client):
    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/predict" in paths
    assert "/health" in paths


def test_docs_ui_is_served(client):
    response = client.get("/docs")

    assert response.status_code == 200