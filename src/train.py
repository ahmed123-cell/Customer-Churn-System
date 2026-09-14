"""
train.py

End-to-end training script for the Telco Customer Churn problem:
    1. Load the raw CSV.
    2. Split into train/test BEFORE preprocessing (to avoid data leakage).
    3. Preprocess train and test sets (fit the scaler on train only, then
       reuse it to transform test) using preprocess_data() from preprocessing.py.
    4. Build every model registered in models_builder.MODEL_BUILDERS.
    5. Evaluate each model on both the training and test sets using
       Precision, Recall, and ROC-AUC.
    6. Print a comparison table, sorted by test ROC-AUC.

Usage:
    python train.py
    python train.py --data Telco-Customer-Churn.csv --test-size 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import structlog
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from preprocessing import preprocess_data
from models_builder import MODEL_BUILDERS
import yaml

def load_train_config(path: str = "configs/train_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

_TRAIN_CFG = load_train_config()


def configure_logging(level: str = "INFO") -> None:
    """Wire up structlog on top of the standard `logging` module, so every
    log call below emits one structured event (key=value fields in a
    terminal, or a single JSON object per line when stdout isn't a tty —
    e.g. piped into a log aggregator) instead of a free-text printed string.

    Safe to call more than once (e.g. if this module is both run as a
    script and imported elsewhere); structlog.configure() simply replaces
    the prior configuration.
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


log = structlog.get_logger("train")


def new_experiment_name(prefix: str) -> str:
    """Build a unique experiment name by suffixing `prefix` with the current
    timestamp, so every run of this script gets its own fresh MLflow
    experiment instead of appending runs to a shared/previous one.
    """
    return f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}"


def log_model_to_mlflow(model, artifact_path: str) -> None:
    """Log a fitted model to the active MLflow run using the flavor that
    matches its origin library, so it's loadable later with the matching
    mlflow.<flavor>.load_model() (and gets proper model-schema tracking).
    """
    module_name = type(model).__module__
    if module_name.startswith("xgboost"):
        mlflow.xgboost.log_model(model, artifact_path=artifact_path)
    elif module_name.startswith("lightgbm"):
        mlflow.lightgbm.log_model(model, artifact_path=artifact_path)
    else:
        mlflow.sklearn.log_model(model, artifact_path=artifact_path)


def get_scores(model, X) -> np.ndarray:
    """Return continuous scores for ROC-AUC, using whichever interface the
    model supports:
    - predict_proba (most classifiers) -> probability of the positive class.
    - decision_function (e.g. LinearSVC, which has no predict_proba) -> raw
      margin score, which ROC-AUC handles just fine (it only needs a
      consistent ranking, not calibrated probabilities).
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    # Last-resort fallback: hard class predictions (degrades AUC quality,
    # but keeps the pipeline from crashing on an unexpected model type).
    return model.predict(X)


def evaluate_model(model, X, y) -> dict:
    """Compute Precision, Recall, and ROC-AUC for a fitted model on (X, y)."""
    y_pred = model.predict(X)
    y_score = get_scores(model, X)
    return {
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_score),
    }


def build_and_evaluate(name, builder, X_train, y_train, X_test, y_test, **params) -> dict:
    """Build one model (with optional hyperparameter overrides), evaluate it,
    and log the run (params, metrics, model artifact) to MLflow. Returns a
    single results row with its train/test Precision, Recall, and ROC-AUC.

    Runs as a nested MLflow run when called from `train_and_evaluate_all`
    (there's already an active parent run for the whole comparison), or as
    its own top-level run when training a single --model directly.
    """
    log.info("training_model", model=name, params=params or None)

    with mlflow.start_run(run_name=name, nested=mlflow.active_run() is not None):
        model = builder(X_train, y_train, **params)

        train_metrics = evaluate_model(model, X_train, y_train)
        test_metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_param("model_name", name)
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        log_model_to_mlflow(model, artifact_path=name)

    return {
        "model": name,
        "train_precision": train_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "train_roc_auc": train_metrics["roc_auc"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_roc_auc": test_metrics["roc_auc"],
    }


def train_and_evaluate_all(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Build every model in MODEL_BUILDERS (with default hyperparameters),
    then evaluate each one on both the training and test sets. Returns a
    tidy comparison DataFrame. Each model's run is logged as a nested
    MLflow run under the parent "all_models" run started by the caller.
    """
    rows = [
        build_and_evaluate(name, builder, X_train, y_train, X_test, y_test)
        for name, builder in MODEL_BUILDERS.items()
    ]
    results = pd.DataFrame(rows).set_index("model")
    return results.sort_values("test_roc_auc", ascending=False)


def parse_param_value(value: str):
    """Cast a CLI-supplied string to the most sensible Python type: bool,
    None, int, float, and falling back to the raw string.
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


def parse_params(param_list: list[str] | None) -> dict:
    """Parse a list of "key=value" CLI strings (as produced by --params)
    into a keyword-argument dict, e.g. ["n_estimators=500", "max_depth=6"]
    -> {"n_estimators": 500, "max_depth": 6}.
    """
    params = {}
    for item in param_list or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid --params entry '{item}', expected format key=value"
            )
        key, value = item.split("=", 1)
        params[key] = parse_param_value(value)
    return params


def main(
    data_path: str,
    test_size: float,
    seed: int,
    model_name: str | None = None,
    params: dict | None = None,
) -> pd.DataFrame:
    params = params or {}

    # 0. Point MLflow at a fresh experiment for this run (optionally a
    #    remote tracking server, per config), so every execution of this
    #    script gets its own experiment instead of accumulating runs into
    #    one shared experiment over time.
    if _TRAIN_CFG.get("mlflow_tracking_uri"):
        mlflow.set_tracking_uri(_TRAIN_CFG["mlflow_tracking_uri"])
    experiment_name = new_experiment_name(_TRAIN_CFG.get("mlflow_experiment_prefix", "telco_churn"))
    mlflow.set_experiment(experiment_name)
    log.info("mlflow_experiment_set", experiment_name=experiment_name)

    # 1. Load raw data.
    raw = pd.read_csv(data_path)
    log.info("data_loaded", data_path=data_path, rows=raw.shape[0], columns=raw.shape[1])

    # 2. Split BEFORE preprocessing, stratified on the target so both splits
    #    keep the same ~26.5% churn rate.
    train_df, test_df = train_test_split(
        raw, test_size=test_size, random_state=seed, stratify=raw["Churn"]
    )
    log.info(
        "train_test_split_complete",
        train_rows=train_df.shape[0],
        test_rows=test_df.shape[0],
        test_size=test_size,
        seed=seed,
    )

    # 3. Preprocess: fit the scaler on the training set only, then reuse it
    #    (fit_scaler=False) to transform the test set, so no information
    #    from the test set leaks into training.
    X_train, y_train, scaler, feature_names = preprocess_data(train_df)
    X_test, y_test, _, _ = preprocess_data(test_df, scaler=scaler, fit_scaler=False)
    log.info("preprocessing_complete", n_features=len(feature_names))

    run_params = {"data_path": data_path, "test_size": test_size, "seed": seed,
                  "n_features": len(feature_names)}

    # 4-5. Train either a single requested model (with optional hyperparameter
    #      overrides) or every model in the registry with defaults.
    if model_name is not None:
        with mlflow.start_run(run_name=f"{model_name}_run"):
            mlflow.log_params(run_params)
            row = build_and_evaluate(
                model_name, MODEL_BUILDERS[model_name], X_train, y_train, X_test, y_test, **params
            )
        results = pd.DataFrame([row]).set_index("model")
    else:
        with mlflow.start_run(run_name="all_models") as parent_run:
            mlflow.log_params(run_params)
            results = train_and_evaluate_all(X_train, y_train, X_test, y_test)

            best_name = results.sort_values("test_roc_auc", ascending=False).index[0]
            mlflow.log_metric("best_test_roc_auc", results.loc[best_name, "test_roc_auc"])
            mlflow.set_tag("best_model", best_name)

            results_path = "artifacts/model_comparison.csv"
            import os
            os.makedirs("artifacts", exist_ok=True)
            results.to_csv(results_path)
            mlflow.log_artifact(results_path)

    # 6. Report.
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", None)

    log.info(
        "model_comparison" if model_name is None else "model_result",
        table=results.to_string(),
    )

    best_model_name = results.index[0]
    log.info(
        "best_model_selected",
        model=best_model_name,
        test_roc_auc=round(float(results.loc[best_model_name, "test_roc_auc"]), 3),
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate churn models.")
    parser.add_argument("--data", default=_TRAIN_CFG["data_path"],
                         help="Path to the raw Telco Customer Churn CSV file.")
    parser.add_argument("--test-size", type=float, default=_TRAIN_CFG["test_size"],
                         help="Fraction of data held out for testing.")
    parser.add_argument("--seed", type=int, default=_TRAIN_CFG["seed"],
                         help="Random seed for the train/test split.")
    parser.add_argument("--model", choices=list(MODEL_BUILDERS.keys()), default=None,
                         help="Train & evaluate only this model. Omit to run every "
                              "model in models_builder.MODEL_BUILDERS.")
    parser.add_argument("--params", nargs="*", default=[], metavar="KEY=VALUE",
                         help="Hyperparameters passed straight to the chosen model's "
                              "builder function, e.g. --params n_estimators=500 max_depth=6. "
                              "Only used together with --model.")
    parser.add_argument("--log-level", default="INFO",
                         choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                         help="Logging verbosity for structlog/logging output.")
    args = parser.parse_args()

    if args.params and args.model is None:
        parser.error("--params requires --model to be set (which model should they apply to?).")

    configure_logging(args.log_level)
    warnings.filterwarnings("ignore")  # keep console output focused on results
    main(args.data, args.test_size, args.seed, args.model, parse_params(args.params))