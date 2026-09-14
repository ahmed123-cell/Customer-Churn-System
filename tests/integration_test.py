"""
tests/integration_test.py

End-to-end integration test tying together every stage the rest of the
suite covers in isolation:

    test_data.py   -> is the raw CSV itself trustworthy?
    test_model.py  -> do the model builders behave correctly on their own?
    test_api.py    -> does the API behave correctly, given a MOCKED model?

This file runs the REAL pipeline start to finish, with nothing mocked:
    1. Load the real CSV, split it (same as train.py).
    2. preprocess_data() on both splits.
    3. Train a real model via models_builder.MODEL_BUILDERS.
    4. Export it with save_model.save_model_to_onnx and sanity-check it
       with save_model.verify_onnx_model.
    5. Point app.py's REAL FastAPI lifespan at that exact .onnx file,
       scaler, and feature order (no mocking of onnxruntime, joblib, or
       preprocess_data, unlike test_api.py).
    6. Call /predict over HTTP for real, held-out customers, and confirm
       the API's answer matches the original (pre-export) model's answer
       on those same customers.

It's slower than the rest of the suite (it actually trains models), so
it's tagged with the `integration` marker. Run it on its own with:

    pytest tests/integration_test.py -v -m integration

or exclude it from a fast local loop with:

    pytest -m "not integration"

Register the marker once (otherwise pytest just warns about an unknown
marker — harmless, but easy to silence) in pyproject.toml:

    [tool.pytest.ini_options]
    markers = ["integration: slow, end-to-end tests against the real data/model pipeline"]

Skips entirely if the real CSV isn't present, matching the pattern used by
test_model.py's own end-to-end test.

Run from the project root (or anywhere) with:
    pytest tests/integration_test.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

# Make the project root (parent of tests/) importable, so app.py /
# preprocessing.py / models_builder.py / save_model.py are importable
# without needing them installed as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_module
from models_builder import MODEL_BUILDERS
from preprocessing import preprocess_data
from save_model import save_model_to_onnx, verify_onnx_model

# NOTE: test_data.py and test_model.py both hardcode this path as the
# Windows literal Path("data\\Telco-Customer-Churn.csv"), which only
# resolves correctly on Windows — on macOS/Linux, pathlib treats the
# backslashes as literal characters rather than a path separator, so the
# file is never found there and those tests silently skip. Using an
# OS-agnostic path here so this file's skip behavior actually reflects
# whether the data exists, on any platform. Worth fixing in the other two
# files too.
DATA_PATH = Path("data") / "Telco-Customer-Churn.csv"

pytestmark = pytest.mark.integration

skip_if_no_data = pytest.mark.skipif(
    not DATA_PATH.exists(), reason="Real Telco CSV not found in data/"
)

# One model that goes through the skl2onnx export path (LogisticRegression)
# and one that goes through the onnxmltools path (LightGBM) — enough to
# exercise both code paths in save_model.py without training and exporting
# all seven builders here; per-model correctness itself is test_model.py's job.
INTEGRATION_MODELS = ["logistic_regression", "lightgbm"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def split_data(raw_data):
    train_df, test_df = train_test_split(
        raw_data, test_size=0.2, random_state=42, stratify=raw_data["Churn"]
    )
    return train_df, test_df


@pytest.fixture(params=INTEGRATION_MODELS)
def trained_pipeline(request, split_data, tmp_path):
    """Run the real pipeline for one model end-to-end (train -> export ->
    verify) and return everything a test needs: the ONNX/scaler/feature
    artifact paths, plus a few real held-out rows with the ORIGINAL
    (pre-export) model's predictions on them as ground truth.
    """
    model_name = request.param
    train_df, test_df = split_data

    X_train, y_train, scaler, feature_names = preprocess_data(train_df)

    model = MODEL_BUILDERS[model_name](X_train, y_train)

    onnx_path = save_model_to_onnx(
        model,
        n_features=X_train.shape[1],
        output_path=str(tmp_path / f"{model_name}.onnx"),
    )

    sample_df = test_df.head(5)
    X_sample, _, _, _ = preprocess_data(sample_df, scaler=scaler, fit_scaler=False)

    # Sanity-check the export itself before trusting it in the API test below.
    verify_onnx_model(onnx_path, X_sample, model)

    scaler_path = tmp_path / "scaler.joblib"
    feature_names_path = tmp_path / "feature_names.json"
    joblib.dump(scaler, scaler_path)
    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f)

    expected_labels = model.predict(X_sample)
    expected_probabilities = (
        model.predict_proba(X_sample)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_sample)
    )

    return {
        "model_name": model_name,
        "onnx_path": str(onnx_path),
        "scaler_path": str(scaler_path),
        "feature_names_path": str(feature_names_path),
        "n_features": len(feature_names),
        "sample_customers": sample_df.drop(columns=["customerID", "Churn"]),
        "expected_labels": expected_labels,
        "expected_probabilities": expected_probabilities,
    }


@pytest.fixture
def api_client(trained_pipeline, monkeypatch):
    """A TestClient wired to the REAL artifacts from `trained_pipeline` —
    unlike test_api.py, nothing about the model layer is mocked: this
    exercises app.py's actual lifespan loading, actual preprocess_data
    call, and actual ONNX Runtime inference.
    """
    monkeypatch.setitem(app_module._APP_CFG, "model_path", trained_pipeline["onnx_path"])
    monkeypatch.setitem(app_module._APP_CFG, "scaler_path", trained_pipeline["scaler_path"])
    monkeypatch.setitem(
        app_module._APP_CFG, "feature_names_path", trained_pipeline["feature_names_path"]
    )

    with TestClient(app_module.app) as client:
        yield client


def _row_to_customer_payload(row: pd.Series) -> dict:
    """Convert a raw DataFrame row into the JSON-serializable dict shape
    app.CustomerData expects (plain Python types, not numpy scalars, and
    TotalCharges coerced the same way preprocessing.py does).
    """
    payload = row.to_dict()
    payload["SeniorCitizen"] = int(payload["SeniorCitizen"])
    payload["tenure"] = int(payload["tenure"])
    payload["MonthlyCharges"] = float(payload["MonthlyCharges"])
    total_charges = pd.to_numeric(payload["TotalCharges"], errors="coerce")
    payload["TotalCharges"] = float(total_charges) if pd.notna(total_charges) else 0.0
    return payload


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@skip_if_no_data
def test_health_reflects_real_loaded_model(api_client, trained_pipeline):
    response = api_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["n_features"] == trained_pipeline["n_features"]


@skip_if_no_data
def test_api_predictions_match_original_model_on_real_customers(api_client, trained_pipeline):
    """The assertion this whole file exists for: for real, held-out
    customers, the API's answer (raw JSON in -> ONNX Runtime inference)
    must agree with the original model's answer on the same rows.
    """
    sample_customers = trained_pipeline["sample_customers"]
    expected_labels = trained_pipeline["expected_labels"]
    expected_probabilities = trained_pipeline["expected_probabilities"]

    for i, (_, row) in enumerate(sample_customers.iterrows()):
        payload = _row_to_customer_payload(row)

        response = api_client.post("/predict", json=payload)
        assert response.status_code == 200, (
            f"[{trained_pipeline['model_name']}] row {i} failed validation: {response.json()}"
        )

        body = response.json()
        assert body["churn_prediction"] == int(expected_labels[i]), (
            f"[{trained_pipeline['model_name']}] row {i}: API label "
            f"{body['churn_prediction']} != model label {expected_labels[i]}"
        )
        # Loose tolerance: ONNX Runtime's floating-point path can differ
        # slightly from scikit-learn's/LightGBM's own predict_proba without
        # indicating a real conversion bug (verify_onnx_model above already
        # checks label agreement more rigorously); this just confirms the
        # probability lands in the right ballpark, not bit-for-bit.
        assert body["churn_probability"] == pytest.approx(
            float(expected_probabilities[i]), abs=0.05
        ), f"[{trained_pipeline['model_name']}] row {i}: probability mismatch"


@skip_if_no_data
def test_api_rejects_real_customer_with_corrupted_category(api_client, trained_pipeline):
    """One negative-path check against real data: take a genuine row and
    corrupt a single categorical field, confirming validation still catches
    it even when everything else about the payload is realistic.
    """
    sample_customers = trained_pipeline["sample_customers"]
    payload = _row_to_customer_payload(sample_customers.iloc[0])
    payload["Contract"] = "Five year"  # not a real category

    response = api_client.post("/predict", json=payload)

    assert response.status_code == 422