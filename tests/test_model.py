"""
tests/test_models.py

Pytest suite that validates the model-building functions in
models_builder.py: correct fitted-estimator interface, correct handling of
class imbalance (class_weight / scale_pos_weight), hyperparameter overrides,
reproducibility, and robustness to extreme imbalance.

Uses a small synthetic dataset (via sklearn.datasets.make_classification)
rather than the real Telco CSV, so these tests are fast and independent of
the data folder. A separate integration test at the bottom exercises the
real preprocessing -> model pipeline end-to-end, and is skipped automatically
if the data file isn't available.

Run from the project root (or anywhere) with:
    pytest tests/test_models.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Make the project root (parent of tests/) importable, so we can import
# models_builder.py / preprocessing.py without needing them installed as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models_builder import (
    MODEL_BUILDERS,
    _compute_scale_pos_weight,
    build_decision_tree,
    build_gradient_boosting,
    build_lightgbm,
    build_linear_svm,
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
)

DATA_PATH = Path("data\\Telco-Customer-Churn.csv")

# Models that support scikit-learn's `class_weight` parameter.
CLASS_WEIGHT_MODELS = {
    "logistic_regression": build_logistic_regression,
    "linear_svm": build_linear_svm,
    "decision_tree": build_decision_tree,
    "random_forest": build_random_forest,
}

# Models that use `scale_pos_weight` instead (XGBoost / LightGBM).
SCALE_POS_WEIGHT_MODELS = {
    "xgboost": build_xgboost,
    "lightgbm": build_lightgbm,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_data():
    """A small, moderately imbalanced (~20% positive) synthetic binary
    classification dataset — fast to train on, but realistic enough to
    exercise class-imbalance handling similar to the churn problem
    (~26.5% positive in the real dataset).
    """
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        weights=[0.80, 0.20],
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def extreme_imbalance_data():
    """A severely imbalanced (~2% positive) dataset, used to check that
    every model builder still fits and predicts without crashing or
    producing NaNs — a common failure mode when reweighting logic divides
    by a near-zero positive count.
    """
    X, y = make_classification(
        n_samples=400,
        n_features=8,
        n_informative=5,
        weights=[0.98, 0.02],
        random_state=0,
    )
    return X, y


# ---------------------------------------------------------------------------
# Registry sanity checks
# ---------------------------------------------------------------------------

def test_model_builders_registry_has_expected_models():
    expected = {
        "logistic_regression", "linear_svm", "decision_tree",
        "random_forest", "gradient_boosting", "xgboost", "lightgbm",
    }
    assert set(MODEL_BUILDERS.keys()) == expected


def test_model_builders_registry_naming_matches_functions():
    """Each registry entry's function should be literally named
    build_<key>, so the registry can't silently point to the wrong builder.
    """
    for name, builder in MODEL_BUILDERS.items():
        assert builder.__name__ == f"build_{name}", (
            f"MODEL_BUILDERS['{name}'] points to '{builder.__name__}', "
            f"expected 'build_{name}'"
        )


# ---------------------------------------------------------------------------
# Interface / fitted-estimator checks (parametrized across every model)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, builder", MODEL_BUILDERS.items())
def test_builder_returns_fitted_estimator_with_predict(name, builder, synthetic_data):
    X_train, X_test, y_train, y_test = synthetic_data
    model = builder(X_train, y_train)

    assert hasattr(model, "predict"), f"{name} model has no predict()"
    preds = model.predict(X_test)

    assert preds.shape == y_test.shape, f"{name} prediction shape mismatch"
    assert set(np.unique(preds)) <= {0, 1}, f"{name} predicted non-binary labels: {np.unique(preds)}"


@pytest.mark.parametrize("name, builder", MODEL_BUILDERS.items())
def test_builder_predictions_are_not_nan(name, builder, synthetic_data):
    X_train, X_test, y_train, y_test = synthetic_data
    model = builder(X_train, y_train)
    preds = model.predict(X_test)
    assert not np.isnan(preds).any(), f"{name} produced NaN prediction(s)"


@pytest.mark.parametrize("name, builder", MODEL_BUILDERS.items())
def test_builder_probability_or_score_output_is_valid(name, builder, synthetic_data):
    """Every model should expose either predict_proba (valid probabilities
    in [0, 1], rows summing to 1) or decision_function (finite scores) —
    both are needed downstream for ROC-AUC in train.py.
    """
    X_train, X_test, y_train, y_test = synthetic_data
    model = builder(X_train, y_train)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        assert proba.shape == (X_test.shape[0], 2), f"{name} predict_proba wrong shape"
        assert np.isfinite(proba).all(), f"{name} predict_proba has non-finite values"
        assert np.allclose(proba.sum(axis=1), 1.0), f"{name} predict_proba rows don't sum to 1"
        assert ((proba >= 0) & (proba <= 1)).all(), f"{name} predict_proba has out-of-range values"
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        assert np.isfinite(scores).all(), f"{name} decision_function has non-finite values"
    else:
        pytest.fail(f"{name} has neither predict_proba nor decision_function")


@pytest.mark.parametrize("name, builder", MODEL_BUILDERS.items())
def test_builder_is_reproducible_with_fixed_random_state(name, builder, synthetic_data):
    """Calling the same builder twice on the same data (with the default
    fixed random_state) should produce identical predictions.
    """
    X_train, X_test, y_train, y_test = synthetic_data
    model_a = builder(X_train, y_train)
    model_b = builder(X_train, y_train)

    preds_a = model_a.predict(X_test)
    preds_b = model_b.predict(X_test)
    assert np.array_equal(preds_a, preds_b), f"{name} is not reproducible across identical calls"


# ---------------------------------------------------------------------------
# Class-imbalance handling: class_weight (sklearn models)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, builder", CLASS_WEIGHT_MODELS.items())
def test_class_weight_defaults_to_balanced(name, builder, synthetic_data):
    X_train, _, y_train, _ = synthetic_data
    model = builder(X_train, y_train)
    assert model.get_params()["class_weight"] == "balanced", (
        f"{name} should default to class_weight='balanced'"
    )


@pytest.mark.parametrize("name, builder", CLASS_WEIGHT_MODELS.items())
def test_class_weight_override_is_respected(name, builder, synthetic_data):
    X_train, _, y_train, _ = synthetic_data
    model = builder(X_train, y_train, class_weight=None)
    assert model.get_params()["class_weight"] is None, (
        f"{name} did not respect class_weight=None override"
    )


def test_gradient_boosting_has_no_class_weight_param(synthetic_data):
    """GradientBoostingClassifier doesn't support class_weight at all —
    documented as a known limitation in build_gradient_boosting's docstring.
    Confirms passing it raises, so a caller can't silently think it worked.
    """
    X_train, _, y_train, _ = synthetic_data
    with pytest.raises(TypeError):
        build_gradient_boosting(X_train, y_train, class_weight="balanced")


# ---------------------------------------------------------------------------
# Class-imbalance handling: scale_pos_weight (XGBoost / LightGBM)
# ---------------------------------------------------------------------------

def test_compute_scale_pos_weight_matches_manual_ratio(synthetic_data):
    X_train, _, y_train, _ = synthetic_data
    expected = (y_train == 0).sum() / (y_train == 1).sum()
    assert _compute_scale_pos_weight(y_train) == pytest.approx(expected)


def test_compute_scale_pos_weight_handles_no_positive_class():
    """Should return 1.0 (no reweighting) rather than raising a
    ZeroDivisionError when the positive class is entirely absent.
    """
    y_all_negative = np.zeros(50, dtype=int)
    assert _compute_scale_pos_weight(y_all_negative) == 1.0


@pytest.mark.parametrize("name, builder", SCALE_POS_WEIGHT_MODELS.items())
def test_scale_pos_weight_auto_computed_by_default(name, builder, synthetic_data):
    X_train, _, y_train, _ = synthetic_data
    model = builder(X_train, y_train)
    expected = _compute_scale_pos_weight(y_train)
    assert model.get_params()["scale_pos_weight"] == pytest.approx(expected), (
        f"{name} did not auto-compute scale_pos_weight correctly"
    )


@pytest.mark.parametrize("name, builder", SCALE_POS_WEIGHT_MODELS.items())
def test_scale_pos_weight_override_is_respected(name, builder, synthetic_data):
    X_train, _, y_train, _ = synthetic_data
    model = builder(X_train, y_train, scale_pos_weight=1.0)
    assert model.get_params()["scale_pos_weight"] == pytest.approx(1.0), (
        f"{name} did not respect an explicit scale_pos_weight override"
    )


# ---------------------------------------------------------------------------
# Hyperparameter override checks
# ---------------------------------------------------------------------------

def test_random_forest_n_estimators_override(synthetic_data):
    X_train, _, y_train, _ = synthetic_data
    model = build_random_forest(X_train, y_train, n_estimators=7)
    assert model.n_estimators == 7
    assert len(model.estimators_) == 7


def test_decision_tree_max_depth_override(synthetic_data):
    X_train, _, y_train, _ = synthetic_data
    model = build_decision_tree(X_train, y_train, max_depth=2)
    assert model.get_depth() <= 2


def test_xgboost_hyperparameters_override(synthetic_data):
    X_train, _, y_train, _ = synthetic_data
    model = build_xgboost(X_train, y_train, n_estimators=15, max_depth=2, learning_rate=0.3)
    params = model.get_params()
    assert params["n_estimators"] == 15
    assert params["max_depth"] == 2
    assert params["learning_rate"] == pytest.approx(0.3)


def test_lightgbm_hyperparameters_override(synthetic_data):
    X_train, _, y_train, _ = synthetic_data
    model = build_lightgbm(X_train, y_train, n_estimators=15, num_leaves=5)
    params = model.get_params()
    assert params["n_estimators"] == 15
    assert params["num_leaves"] == 5


# ---------------------------------------------------------------------------
# Robustness under extreme class imbalance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, builder", MODEL_BUILDERS.items())
def test_builder_handles_extreme_imbalance_without_crashing(name, builder, extreme_imbalance_data):
    X, y = extreme_imbalance_data
    # Skip if the split has too few positive samples to be meaningful.
    if (y == 1).sum() < 2:
        pytest.skip("Not enough positive samples in this synthetic draw")

    model = builder(X, y)
    preds = model.predict(X)
    assert not np.isnan(preds).any(), f"{name} produced NaNs under extreme imbalance"
    assert set(np.unique(preds)) <= {0, 1}


# ---------------------------------------------------------------------------
# End-to-end integration test with the real preprocessing pipeline
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not DATA_PATH.exists(), reason="Real Telco CSV not found in data/")
def test_end_to_end_pipeline_on_real_data():
    """Smoke test: preprocess the real dataset and build one model of each
    family (linear + tree-based) end-to-end, checking they train and score
    better than random guessing.
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from preprocessing import preprocess_data

    raw = pd.read_csv(DATA_PATH)
    train_df, test_df = train_test_split(
        raw, test_size=0.2, random_state=42, stratify=raw["Churn"]
    )
    X_train, y_train, scaler, _ = preprocess_data(train_df)
    X_test, y_test, _, _ = preprocess_data(test_df, scaler=scaler, fit_scaler=False)

    for name, builder in MODEL_BUILDERS.items():
        model = builder(X_train, y_train)
        scores = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X_test)
        )
        auc = roc_auc_score(y_test, scores)
        assert auc > 0.5, f"{name} performed no better than random guessing (AUC={auc:.3f})"