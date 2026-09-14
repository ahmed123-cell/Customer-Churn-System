"""
preprocessing.py

Preprocessing pipeline for the Telco Customer Churn dataset, producing model-ready
features for BOTH linear models (Logistic Regression, SVM, etc.) and tree-based
models (Random Forest, XGBoost, LightGBM, etc.).

Design choices (and why they work for both model families):
- Numeric features are scaled with StandardScaler.
    * Linear/distance-based models need this to converge properly and to keep
      coefficients comparable.
    * Tree-based models are scale-invariant, so scaling is harmless for them.
- Categorical features are one-hot encoded (with drop_first=True).
    * Linear models require this to avoid the dummy-variable trap
      (perfect multicollinearity).
    * Tree-based models handle one-hot columns fine too.
- Binary Yes/No-style columns are mapped to 0/1 instead of one-hot encoded,
  to avoid creating redundant columns.
- The target `Churn` is encoded to 0/1 in the target module and returned
  separately from the features.

Usage:
    from preprocessing import preprocess_data
    X, y, scaler, feature_names = preprocess_data(df)

    # To apply the SAME transformation to new/unseen data (e.g. a test set),
    # reuse the fitted scaler and column layout:
    X_new, y_new, _, _ = preprocess_data(new_df, scaler=scaler, fit_scaler=False)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yaml
from pathlib import Path

def load_data_config(path: str = "configs/data_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

_CFG = load_data_config()

ID_COLUMNS = _CFG["id_columns"]
TARGET_COLUMN = _CFG["target_column"]
BINARY_COLUMN_MAPS = _CFG["binary_column_maps"]
KNOWN_CATEGORIES = _CFG["known_categories"]
NUMERIC_COLUMNS = _CFG["numeric_columns"]
ALREADY_BINARY_NUMERIC_COLUMNS = _CFG["already_binary_numeric_columns"]


def _clean_total_charges(data: pd.DataFrame) -> pd.DataFrame:
    """Fix the `TotalCharges` column: it is loaded as text and has 11 blank
    entries, all belonging to brand-new customers (tenure == 0). We convert
    it to numeric and set those blanks to 0 (no charges accrued yet), which
    reflects reality better than dropping rows or imputing a mean/median.
    """
    data = data.copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"] = data["TotalCharges"].fillna(0)
    return data


def _encode_binary_columns(data: pd.DataFrame, column_maps: dict[str, dict]) -> pd.DataFrame:
    """Map two-category Yes/No-style (and gender) columns to 0/1 in place,
    using fixed mappings so this works correctly even on a single row or a
    subset of data that doesn't contain both categories.
    """
    data = data.copy()
    for col, mapping in column_maps.items():
        if col not in data.columns:
            continue
        unexpected = set(data[col].dropna().unique()) - set(mapping.keys())
        if unexpected:
            raise ValueError(
                f"Column '{col}' has unexpected value(s) {unexpected}; "
                f"expected only {list(mapping.keys())}"
            )
        data[col] = data[col].map(mapping).astype(int)
    return data


def _one_hot_encode(data: pd.DataFrame, known_categories: dict[str, list[str]]) -> pd.DataFrame:
    """One-hot encode multi-category columns, dropping the first level of
    each to avoid the dummy-variable trap (needed for linear models; harmless
    for tree-based models).

    Each column is first cast to a pandas Categorical with a FIXED, known
    set of categories (rather than whatever happens to appear in `data`).
    This guarantees the same set of output dummy columns regardless of
    whether `data` is the full training set, a single row, or a small batch
    of new data that doesn't contain every category.
    """
    data = data.copy()
    columns_present = [c for c in known_categories if c in data.columns]
    for col in columns_present:
        unexpected = set(data[col].dropna().unique()) - set(known_categories[col])
        if unexpected:
            raise ValueError(
                f"Column '{col}' has unexpected value(s) {unexpected}; "
                f"expected only {known_categories[col]}"
            )
        data[col] = pd.Categorical(data[col], categories=known_categories[col])
    return pd.get_dummies(data, columns=columns_present, drop_first=True)


def preprocess_data(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True,
    return_dataframe: bool = False,
):
    """Turn the raw Telco Customer Churn dataframe into model-ready features.

    Parameters
    ----------
    data : pd.DataFrame
        The raw input dataframe (e.g. loaded straight from the CSV). Must
        contain the original Telco Customer Churn columns. A target column
        is optional — if absent, `y` is returned as None (useful for
        preprocessing genuinely unlabeled inference data).
    target_column : str, default "Churn"
        Name of the target column. Encoded to 0/1 ("Yes" -> 1, "No" -> 0).
    scaler : sklearn.preprocessing.StandardScaler, optional
        A previously-fitted scaler. Pass this (with fit_scaler=False) when
        preprocessing a validation/test/inference set, so it is scaled with
        the SAME statistics learned from the training set instead of leaking
        information from the new data.
    fit_scaler : bool, default True
        If True, fits a new StandardScaler on this data's numeric columns
        (use for training data). If False, `scaler` must be provided and is
        only used to `.transform()` (use for validation/test/inference data).
    return_dataframe : bool, default False
        If True, X is returned as a pandas DataFrame (with column names) —
        handy for feature-importance inspection with tree-based models.
        If False (default), X is returned as a numpy array, ready to feed
        directly into any scikit-learn-style estimator.

    Returns
    -------
    X : np.ndarray or pd.DataFrame
        Model-ready feature matrix (scaled numeric + binary-mapped + one-hot
        encoded categorical columns).
    y : np.ndarray or None
        Encoded target vector (1 = churned, 0 = retained), or None if
        `target_column` is not present in `data`.
    scaler : sklearn.preprocessing.StandardScaler
        The fitted (or reused) scaler — keep this to preprocess future data
        consistently (e.g. a held-out test set or new customers).
    feature_names : list[str]
        Column names of X, in order — useful for reading coefficients
        (linear models) or feature importances (tree-based models).
    """
    df = data.copy()

    # 1. Drop non-informative identifier columns.
    df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])

    # 2. Fix TotalCharges (text -> numeric, blanks -> 0 for tenure == 0 customers).
    df = _clean_total_charges(df)

    # 3. Encode the target, if present, then separate it from the features.
    y = None
    if target_column in df.columns:
        y = (df[target_column] == "Yes").astype(int).to_numpy()
        df = df.drop(columns=[target_column])

    # 4. Encode binary Yes/No-style categorical columns as 0/1.
    df = _encode_binary_columns(df, BINARY_COLUMN_MAPS)

    # 5. One-hot encode remaining multi-category columns.
    df = _one_hot_encode(df, KNOWN_CATEGORIES)

    # 6. Scale continuous numeric columns.
    numeric_cols_present = [c for c in NUMERIC_COLUMNS if c in df.columns]
    if fit_scaler:
        scaler = StandardScaler()
        df[numeric_cols_present] = scaler.fit_transform(df[numeric_cols_present])
    else:
        if scaler is None:
            raise ValueError("`scaler` must be provided when fit_scaler=False.")
        df[numeric_cols_present] = scaler.transform(df[numeric_cols_present])

    # 7. Ensure a consistent, all-numeric dtype (bool columns from get_dummies -> int).
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    feature_names = df.columns.tolist()
    X = df if return_dataframe else df.to_numpy(dtype=float)

    return X, y, scaler, feature_names