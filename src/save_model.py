"""
save_model.py

Export a fitted classifier from models_builder.py (scikit-learn,
XGBoost, or LightGBM) to the ONNX format, so it can be served or run
outside of Python (e.g. ONNX Runtime in another language, or a
model-serving platform) instead of relying on pickle/joblib.

Requires:
    pip install skl2onnx onnxmltools onnx onnxruntime

Usage:
    from save_model import save_model_to_onnx, verify_onnx_model

    # model is any fitted estimator returned by a models_builder.py builder
    # n_features must match X_train.shape[1] used to fit it
    onnx_path = save_model_to_onnx(model, n_features=X_train.shape[1],
                                    output_path="artifacts/random_forest.onnx")

    # optional: confirm the ONNX model's predictions match the original
    verify_onnx_model(onnx_path, X_test, model)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import numpy as np
import structlog


def configure_logging(level: str = "INFO") -> None:
    """Wire up structlog on top of the standard `logging` module, so every
    log call below emits one structured event (key=value fields in a
    terminal, or a single JSON object per line when stdout isn't a tty —
    e.g. piped into a log aggregator) instead of a free-text printed string.

    Mirrors train.py's configure_logging so both scripts produce logs in
    the same format.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=log_level)

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer()
            if sys.stdout.isatty()
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


log = structlog.get_logger("save_model")


def _initial_type(n_features: int):
    """Build the ONNX input signature: a float32 tensor of shape
    (batch_size, n_features), with batch_size left dynamic (`None`).
    """
    from skl2onnx.common.data_types import FloatTensorType

    return [("input", FloatTensorType([None, n_features]))]


def _convert_to_onnx(model, n_features: int, zipmap: bool = False):
    """Dispatch to the right converter library based on the model's origin
    module, and return the in-memory ONNX ModelProto.

    - scikit-learn estimators (LogisticRegression, LinearSVC,
      DecisionTreeClassifier, RandomForestClassifier,
      GradientBoostingClassifier) -> skl2onnx.
    - XGBoost (XGBClassifier) and LightGBM (LGBMClassifier) -> onnxmltools,
      which has dedicated converters for these external tree libraries
      (skl2onnx's own converter only covers scikit-learn's native classes).

    Parameters
    ----------
    model : fitted estimator
        Any model produced by a `build_*` function in models_builder.py.
    n_features : int
        Number of input features the model was trained on (X_train.shape[1]).
    zipmap : bool, default False
        scikit-learn classifiers, by default, have their probability output
        wrapped by skl2onnx in a "ZipMap" (a list of {class: prob} dicts),
        mirroring predict_proba's usual shape. Setting False instead
        outputs a plain float tensor, which is simpler to consume from
        ONNX Runtime in most non-Python contexts. Ignored for XGBoost/
        LightGBM, which don't use ZipMap in onnxmltools.
    """
    initial_type = _initial_type(n_features)
    module_name = type(model).__module__

    if module_name.startswith("xgboost"):
        from onnxmltools import convert_xgboost

        return convert_xgboost(model, initial_types=initial_type)

    if module_name.startswith("lightgbm"):
        from onnxmltools import convert_lightgbm

        return convert_lightgbm(model, initial_types=initial_type)

    # Default: any scikit-learn-compatible estimator (LogisticRegression,
    # LinearSVC, DecisionTreeClassifier, RandomForestClassifier,
    # GradientBoostingClassifier, etc.)
    from skl2onnx import convert_sklearn

    options = {id(model): {"zipmap": False}} if not zipmap else None
    return convert_sklearn(model, initial_types=initial_type, options=options)


def save_model_to_onnx(
    model,
    n_features: int,
    output_path: str,
    zipmap: bool = False,
) -> str:
    """Convert a fitted model to ONNX and write it to disk.

    Parameters
    ----------
    model : fitted estimator
        Any model produced by a `build_*` function in models_builder.py
        (already fitted — this function does not call .fit()).
    n_features : int
        Number of input features the model was trained on, e.g.
        `X_train.shape[1]` or `len(feature_names)` from preprocess_data().
    output_path : str
        Destination path for the .onnx file. Parent directories are
        created automatically if they don't exist.
    zipmap : bool, default False
        See `_convert_to_onnx` — set True to keep scikit-learn's default
        dict-style probability output instead of a plain float tensor.

    Returns
    -------
    str
        The `output_path` the model was saved to (for convenience/chaining).
    """
    onnx_model = _convert_to_onnx(model, n_features=n_features, zipmap=zipmap)

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    return output_path


def verify_onnx_model(
    onnx_path: str,
    X_sample: np.ndarray,
    original_model,
    atol: float = 1e-4,
) -> bool:
    """Sanity-check an exported ONNX model by comparing its predicted
    class labels against the original (pre-conversion) model's predictions
    on the same input. Useful right after export to catch conversion bugs
    (e.g. a wrong `n_features`, or a model type ONNX handles imprecisely)
    before relying on the .onnx file elsewhere.

    Parameters
    ----------
    onnx_path : str
        Path to the .onnx file written by `save_model_to_onnx`.
    X_sample : np.ndarray
        Feature matrix to run through both models (e.g. X_test, or a
        smaller held-out slice of it). Cast to float32 for ONNX Runtime.
    original_model : fitted estimator
        The same model instance that was passed to `save_model_to_onnx`.
    atol : float, default 1e-4
        Unused for label comparison directly, kept for callers who want to
        extend this to compare probability outputs numerically.

    Returns
    -------
    bool
        True if predicted labels match exactly between the original model
        and the ONNX model on `X_sample`.

    Raises
    ------
    AssertionError
        If predicted labels differ, with a message showing how many rows
        mismatched (helps decide whether it's a rounding issue vs a real
        conversion problem).
    """
    import onnxruntime as ort

    X_sample = np.asarray(X_sample, dtype=np.float32)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_outputs = session.run(None, {input_name: X_sample})
    onnx_labels = np.asarray(onnx_outputs[0]).reshape(-1)

    original_labels = np.asarray(original_model.predict(X_sample)).reshape(-1)

    mismatches = int((onnx_labels != original_labels).sum())
    if mismatches:
        raise AssertionError(
            f"ONNX model predictions differ from the original model on "
            f"{mismatches}/{len(original_labels)} sample rows."
        )

    return True


def _build_arg_parser():
    import argparse
    from models_builder import MODEL_BUILDERS

    parser = argparse.ArgumentParser(
        description="Train a model from models_builder.py on the given data "
                     "and export it to ONNX."
    )
    parser.add_argument("--data", required=True,
                         help="Path to the raw Telco Customer Churn CSV file.")
    parser.add_argument("--model", required=True, choices=list(MODEL_BUILDERS.keys()),
                         help="Which model builder to train and export.")
    parser.add_argument("--params", nargs="*", default=[], metavar="KEY=VALUE",
                         help="Hyperparameters passed straight to the chosen "
                              "model's builder function, e.g. "
                              "--params n_estimators=500 max_depth=6.")
    parser.add_argument("--output", default=None,
                         help="Destination .onnx path. Defaults to "
                              "artifacts/<model>.onnx")
    parser.add_argument("--zipmap", action="store_true",
                         help="Keep scikit-learn's default dict-style "
                              "probability output instead of a plain tensor. "
                              "Ignored for XGBoost/LightGBM.")
    parser.add_argument("--no-verify", action="store_true",
                         help="Skip the post-export ONNX Runtime sanity check.")
    parser.add_argument("--log-level", default="INFO",
                         choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                         help="Logging verbosity for structlog/logging output.")
    return parser


def _parse_param_value(value: str):
    """Cast a CLI-supplied string to the most sensible Python type: bool,
    None, int, float, and falling back to the raw string. Mirrors
    train.py's parse_param_value so --params behaves identically here.
    """
    lowered = value.lower()
    if lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_params(param_list) -> dict:
    """Parse ["key=value", ...] CLI strings into a kwargs dict."""
    params = {}
    for item in param_list or []:
        if "=" not in item:
            raise ValueError(f"Invalid --params entry '{item}', expected format key=value")
        key, value = item.split("=", 1)
        params[key] = _parse_param_value(value)
    return params


def main():
    import json
    import joblib
    import pandas as pd
    from preprocessing import preprocess_data
    from models_builder import MODEL_BUILDERS

    args = _build_arg_parser().parse_args()
    configure_logging(args.log_level)
    params = _parse_params(args.params)

    raw = pd.read_csv(args.data)
    log.info("data_loaded", data_path=args.data, rows=raw.shape[0], columns=raw.shape[1])

    X, y, scaler, feature_names = preprocess_data(raw)
    log.info("preprocessing_complete", n_features=len(feature_names))

    log.info("training_model", model=args.model, params=params or None)
    model = MODEL_BUILDERS[args.model](X, y, **params)

    output_path = args.output or f"artifacts/{args.model}.onnx"
    onnx_path = save_model_to_onnx(
        model, n_features=X.shape[1], output_path=output_path, zipmap=args.zipmap
    )
    log.info("model_exported", model=args.model, onnx_path=onnx_path)

    if not args.no_verify:
        verify_onnx_model(onnx_path, X[:50], model)
        log.info("onnx_verification_passed", onnx_path=onnx_path, sample_size=50)

    # Persist the fitted scaler and feature ordering alongside the model so
    # a serving layer (e.g. app.py) can preprocess new requests exactly the
    # way this model was trained, without needing to refit anything.
    parent = os.path.dirname(onnx_path) or "."
    scaler_path = os.path.join(parent, "scaler.joblib")
    feature_names_path = os.path.join(parent, "feature_names.json")

    joblib.dump(scaler, scaler_path)
    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f)

    log.info("scaler_saved", scaler_path=scaler_path)
    log.info("feature_names_saved", feature_names_path=feature_names_path)


if __name__ == "__main__":
    main()