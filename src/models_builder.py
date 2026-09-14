"""
models_builder.py

Functions to build (instantiate + fit) classification models for the Telco
Customer Churn problem, covering both linear and tree-based model families.

Every function has the same minimal signature contract:
    build_<model_name>(X_train, y_train, **hyperparameters) -> fitted estimator

`X_train`/`y_train` are expected to already be preprocessed (see
preprocessing.py) — i.e. numeric, scaled where appropriate, and one-hot
encoded. All other hyperparameters have sensible defaults but can be
overridden by the caller.

Requires: scikit-learn, xgboost, lightgbm
    pip install scikit-learn xgboost lightgbm

Note on class imbalance: EDA showed churn is imbalanced (~26.5% positive
class). Models that support `class_weight` default to `class_weight="balanced"`
so the minority (churned) class isn't ignored. XGBoost and LightGBM don't use
`class_weight` — instead they use `scale_pos_weight`, which is computed
automatically from `y_train` (ratio of negative to positive samples) unless
explicitly overridden. This can be turned off by passing `class_weight=None`
(sklearn models) or `scale_pos_weight=1` (XGBoost/LightGBM).

Usage:
    from models_builder import build_logistic_regression, build_random_forest, build_xgboost

    log_reg = build_logistic_regression(X_train, y_train)
    rf = build_random_forest(X_train, y_train, n_estimators=500)
    xgb = build_xgboost(X_train, y_train)  # scale_pos_weight auto-computed from y_train
"""

from __future__ import annotations

import numpy as np
import yaml
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def load_model_config(path: str = "configs/model_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

_MODEL_CFG = load_model_config()

def _defaults(name: str) -> dict:
    return _MODEL_CFG[name]["params"]


def _compute_scale_pos_weight(y_train) -> float:
    """Compute the negative/positive class ratio from the training labels,
    for use as `scale_pos_weight` in XGBoost/LightGBM. This up-weights the
    minority (churn = 1) class proportionally to how rare it is, analogous
    to `class_weight="balanced"` in scikit-learn models.

    Returns 1.0 (no reweighting) if the positive class is missing or the
    data is already balanced/degenerate, to avoid a division by zero.
    """
    y_train = np.asarray(y_train)
    n_positive = int((y_train == 1).sum())
    n_negative = int((y_train == 0).sum())
    if n_positive == 0:
        return 1.0
    return n_negative / n_positive


# ---------------------------------------------------------------------------
# Linear models
# ---------------------------------------------------------------------------

def build_logistic_regression(
    X_train, y_train,
    C: float = _defaults("logistic_regression")["C"],
    solver: str = _defaults("logistic_regression")["solver"],
    max_iter: int = _defaults("logistic_regression")["max_iter"],
    class_weight: str | dict | None = _defaults("logistic_regression")["class_weight"],
    random_state: int = _defaults("logistic_regression")["random_state"],
) -> LogisticRegression:
    """Build and fit a Logistic Regression classifier.

    A good baseline linear model: fast, interpretable via coefficients, and
    works well once features are scaled and one-hot encoded (see
    preprocessing.py). Uses L2 regularization (scikit-learn's default).

    Parameters
    ----------
    C : float, default 1.0
        Inverse of regularization strength (smaller = stronger regularization).
    solver : str, default "lbfgs"
        Optimization algorithm (e.g. "lbfgs", "liblinear", "saga").
    max_iter : int, default 1000
        Increased from sklearn's default (100) since one-hot encoded data
        with many features can need more iterations to converge.
    class_weight : "balanced", dict, or None, default "balanced"
        Set to "balanced" to automatically up-weight the minority (churn) class.
    random_state : int, default 42
        Seed for reproducibility.
    """
    model = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def build_linear_svm(
    X_train, y_train,
    C: float = _defaults("linear_svm")["C"],
    class_weight: str | dict | None = _defaults("linear_svm")["class_weight"],
    max_iter: int =  _defaults("linear_svm")["max_iter"],
    random_state: int = _defaults("linear_svm")["random_state"],
) -> LinearSVC:
    """Build and fit a Linear Support Vector Classifier.

    Another linear model, useful as a comparison point to Logistic
    Regression — tends to perform well when classes are close to linearly
    separable in the (scaled, one-hot encoded) feature space.

    Parameters
    ----------
    C : float, default 1.0
        Regularization strength (smaller = stronger regularization).
    class_weight : "balanced", dict, or None, default "balanced"
        Set to "balanced" to automatically up-weight the minority (churn) class.
    max_iter : int, default 5000
        Increased from sklearn's default (1000) to help convergence on
        one-hot encoded data.
    random_state : int, default 42
        Seed for reproducibility.
    """
    model = LinearSVC(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Tree-based models
# ---------------------------------------------------------------------------

def build_decision_tree(
    X_train, y_train,
    max_depth: int | None = _defaults("decision_tree")["max_depth"],
    min_samples_leaf: int = _defaults("decision_tree")["min_samples_leaf"],
    class_weight: str | dict | None = _defaults("decision_tree")["class_weight"],
    random_state: int = _defaults("decision_tree")["random_state"],
) -> DecisionTreeClassifier:
    """Build and fit a single Decision Tree classifier.

    Kept shallow by default (max_depth=5, min_samples_leaf=20) to avoid
    overfitting and to stay easily interpretable/visualizable.

    Parameters
    ----------
    max_depth : int or None, default 5
        Maximum tree depth. None grows the tree until leaves are pure
        (much higher overfitting risk) — reduce for more regularization.
    min_samples_leaf : int, default 20
        Minimum samples required in a leaf node; higher values regularize
        the tree further.
    class_weight : "balanced", dict, or None, default "balanced"
        Set to "balanced" to automatically up-weight the minority (churn) class.
    random_state : int, default 42
        Seed for reproducibility.
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def build_random_forest(
    X_train, y_train,
    n_estimators: int = _defaults("random_forest")["n_estimators"],
    max_depth: int | None =  _defaults("random_forest")["max_depth"],
    min_samples_leaf: int =  _defaults("random_forest")["min_samples_leaf"],
    class_weight: str | dict | None =  _defaults("random_forest")["class_weight"],
    n_jobs: int =  _defaults("random_forest")["n_jobs"],
    random_state: int =  _defaults("random_forest")["random_state"],
) -> RandomForestClassifier:
    """Build and fit a Random Forest classifier.

    An ensemble of decision trees that generally outperforms a single tree
    and is fairly robust to overfitting and to feature scaling choices.

    Parameters
    ----------
    n_estimators : int, default 300
        Number of trees in the forest.
    max_depth : int or None, default None
        Maximum depth per tree. None lets trees grow fully (bagging across
        many trees mitigates the overfitting risk this would carry alone).
    min_samples_leaf : int, default 5
        Minimum samples required in a leaf node; regularizes each tree.
    class_weight : "balanced", dict, or None, default "balanced"
        Set to "balanced" to automatically up-weight the minority (churn) class.
    n_jobs : int, default -1
        Number of parallel jobs (-1 uses all available CPU cores).
    random_state : int, default 42
        Seed for reproducibility.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def build_gradient_boosting(
    X_train, y_train,
    n_estimators: int = _defaults("gradient_boosting")["n_estimators"],
    learning_rate: float = _defaults("gradient_boosting")["learning_rate"],
    max_depth: int = _defaults("gradient_boosting")["max_depth"],
    subsample: float = _defaults("gradient_boosting")["subsample"],
    random_state: int = _defaults("gradient_boosting")["random_state"],
) -> GradientBoostingClassifier:
    """Build and fit a Gradient Boosting classifier.

    Builds trees sequentially, each correcting the errors of the previous
    ones — typically the strongest of these tree-based options on tabular
    data like this, at the cost of longer training time and more sensitivity
    to hyperparameters.

    Note: scikit-learn's GradientBoostingClassifier does not support
    `class_weight`. To handle class imbalance here, either pass
    `sample_weight` to `.fit()` yourself on the returned model, or resample
    the training data (e.g. SMOTE) before calling this function.

    Parameters
    ----------
    n_estimators : int, default 200
        Number of boosting stages (trees) to fit.
    learning_rate : float, default 0.05
        Shrinks the contribution of each tree; lower values need more
        estimators but generalize better.
    max_depth : int, default 3
        Maximum depth of each individual tree (kept shallow, as is typical
        for boosting).
    subsample : float, default 0.8
        Fraction of samples used per tree (< 1.0 adds randomness, reducing
        overfitting, similar to stochastic gradient boosting).
    random_state : int, default 42
        Seed for reproducibility.
    """
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Boosted tree models (external libraries)
# ---------------------------------------------------------------------------

def build_xgboost(
    X_train, y_train,
    n_estimators: int = _defaults("xgboost")["n_estimators"],
    learning_rate: float = _defaults("xgboost")["learning_rate"],
    max_depth: int = _defaults("xgboost")["max_depth"],
    subsample: float = _defaults("xgboost")["subsample"],
    colsample_bytree: float = _defaults("xgboost")["colsample_bytree"],
    scale_pos_weight: float | None = _defaults("xgboost")["scale_pos_weight"],
    random_state: int = _defaults("xgboost")["random_state"],
    n_jobs: int = _defaults("xgboost")["n_jobs"],
) -> XGBClassifier:
    """Build and fit an XGBoost classifier.

    A gradient-boosted tree implementation that's typically faster and often
    more accurate than scikit-learn's GradientBoostingClassifier, thanks to
    regularization terms and a more efficient tree-building algorithm.

    Parameters
    ----------
    n_estimators : int, default 300
        Number of boosting rounds (trees).
    learning_rate : float, default 0.05
        Shrinks each tree's contribution; lower values need more estimators
        but generalize better.
    max_depth : int, default 4
        Maximum depth of each tree.
    subsample : float, default 0.8
        Fraction of training rows sampled per tree (adds randomness, reduces
        overfitting).
    colsample_bytree : float, default 0.8
        Fraction of features sampled per tree (adds randomness, reduces
        overfitting).
    scale_pos_weight : float or None, default None
        Weight multiplier for the positive (churn = 1) class, used to
        counteract class imbalance. If None (default), it is computed
        automatically from `y_train` as (# negative / # positive) — the
        standard XGBoost recommendation for imbalanced binary
        classification. Pass 1.0 explicitly to disable reweighting.
    random_state : int, default 42
        Seed for reproducibility.
    n_jobs : int, default -1
        Number of parallel threads (-1 uses all available CPU cores).
    """
    if scale_pos_weight is None:
        scale_pos_weight = _compute_scale_pos_weight(y_train)

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model


def build_lightgbm(
    X_train, y_train,
    n_estimators: int = _defaults("lightgbm")["n_estimators"],
    learning_rate: float = _defaults("lightgbm")["learning_rate"],
    max_depth: int = _defaults("lightgbm")["max_depth"],
    num_leaves: int = _defaults("lightgbm")["num_leaves"],
    subsample: float = _defaults("lightgbm")["subsample"],
    colsample_bytree: float = _defaults("lightgbm")["colsample_bytree"],
    scale_pos_weight: float | None = _defaults("lightgbm")["scale_pos_weight"],
    random_state: int = _defaults("lightgbm")["random_state"],
    n_jobs: int = _defaults("lightgbm")["n_jobs"],
) -> LGBMClassifier:
    """Build and fit a LightGBM classifier.

    A gradient-boosted tree implementation optimized for speed on larger
    datasets via histogram-based splitting and leaf-wise tree growth —
    usually a strong, fast-training alternative/companion to XGBoost.

    Parameters
    ----------
    n_estimators : int, default 300
        Number of boosting rounds (trees).
    learning_rate : float, default 0.05
        Shrinks each tree's contribution; lower values need more estimators
        but generalize better.
    max_depth : int, default -1
        Maximum tree depth (-1 means no limit; LightGBM instead controls
        complexity mainly via `num_leaves`).
    num_leaves : int, default 31
        Maximum number of leaves per tree — LightGBM's primary complexity
        control knob since it grows trees leaf-wise rather than level-wise.
    subsample : float, default 0.8
        Fraction of training rows sampled per tree (adds randomness, reduces
        overfitting).
    colsample_bytree : float, default 0.8
        Fraction of features sampled per tree (adds randomness, reduces
        overfitting).
    scale_pos_weight : float or None, default None
        Weight multiplier for the positive (churn = 1) class, used to
        counteract class imbalance. If None (default), it is computed
        automatically from `y_train` as (# negative / # positive). Pass 1.0
        explicitly to disable reweighting.
    random_state : int, default 42
        Seed for reproducibility.
    n_jobs : int, default -1
        Number of parallel threads (-1 uses all available CPU cores).
    """
    if scale_pos_weight is None:
        scale_pos_weight = _compute_scale_pos_weight(y_train)

    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Convenience registry
# ---------------------------------------------------------------------------

# Maps a readable model name to its builder function — handy for looping
# over every model in a training/evaluation script without importing each
# function by name individually.
MODEL_BUILDERS = {name: fn for name, fn in {
    "logistic_regression": build_logistic_regression,
    "linear_svm": build_linear_svm,
    "decision_tree": build_decision_tree,
    "random_forest": build_random_forest,
    "gradient_boosting": build_gradient_boosting,
    "xgboost": build_xgboost,
    "lightgbm": build_lightgbm,
}.items() if _MODEL_CFG[name]["enabled"]}