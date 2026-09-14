"""
tests/test_data.py

Pytest suite that validates the raw Telco Customer Churn CSV data quality
itself (independent of any preprocessing code) — i.e. it checks that the
data we're about to build a pipeline on top of actually looks the way we
expect. Run from the project root (or anywhere) with:

    pytest tests/test_data.py -v

Assumes the following project layout:
    project/
        data/Telco-Customer-Churn.csv
        tests/test_data.py   <- this file
"""

from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_PATH = Path("data") / "Telco-Customer-Churn.csv"

EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

# Expected category domains for each categorical column — used to catch
# unexpected/typo'd values (e.g. "no" vs "No", a new category appearing).
EXPECTED_CATEGORIES = {
    "gender": {"Female", "Male"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
    "Churn": {"Yes", "No"},
}


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    """Load the raw CSV once and share it across all tests in this module."""
    assert DATA_PATH.exists(), f"Data file not found at {DATA_PATH}"
    return pd.read_csv(DATA_PATH)


# ---------------------------------------------------------------------------
# Structural checks: file, shape, columns, duplicates
# ---------------------------------------------------------------------------


def test_data_file_exists():
    assert DATA_PATH.exists(), f"Expected data file at {DATA_PATH}"


def test_data_is_not_empty(raw_df):
    assert raw_df.shape[0] > 0, "Dataset has no rows"
    assert raw_df.shape[1] > 0, "Dataset has no columns"


def test_expected_columns_present(raw_df):
    missing = set(EXPECTED_COLUMNS) - set(raw_df.columns)
    assert not missing, f"Missing expected column(s): {missing}"


def test_no_unexpected_extra_columns(raw_df):
    extra = set(raw_df.columns) - set(EXPECTED_COLUMNS)
    assert not extra, f"Found unexpected column(s) not accounted for: {extra}"


def test_no_duplicate_customer_ids(raw_df):
    duplicate_count = raw_df["customerID"].duplicated().sum()
    assert duplicate_count == 0, f"Found {duplicate_count} duplicate customerID(s)"


def test_no_fully_duplicate_rows(raw_df):
    duplicate_count = raw_df.duplicated().sum()
    assert duplicate_count == 0, f"Found {duplicate_count} fully duplicated row(s)"


def test_customer_id_format(raw_df):
    """customerID is expected to follow the pattern NNNN-LLLLL (4 digits,
    a hyphen, 5 uppercase letters), e.g. '7590-VHVEG'."""
    pattern = r"^\d{4}-[A-Z]{5}$"
    invalid = raw_df.loc[~raw_df["customerID"].str.match(pattern), "customerID"]
    assert invalid.empty, (
        f"Found {len(invalid)} customerID(s) with an unexpected format: {invalid.tolist()[:5]}"
    )


# ---------------------------------------------------------------------------
# Missing-value checks
# ---------------------------------------------------------------------------


def test_no_missing_values_in_key_columns(raw_df):
    """Every column except TotalCharges (which has known blank strings for
    brand-new customers — see test_total_charges_* below) should have no
    missing values as loaded by pandas.
    """
    key_columns = [c for c in EXPECTED_COLUMNS if c != "TotalCharges"]
    na_counts = raw_df[key_columns].isnull().sum()
    offending = na_counts[na_counts > 0]
    assert offending.empty, f"Unexpected missing values found:\n{offending}"


# ---------------------------------------------------------------------------
# Categorical domain checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("column, expected_values", EXPECTED_CATEGORIES.items())
def test_categorical_values_within_expected_domain(raw_df, column, expected_values):
    actual_values = set(raw_df[column].dropna().unique())
    unexpected = actual_values - expected_values
    assert not unexpected, (
        f"Column '{column}' contains unexpected value(s) {unexpected}; "
        f"expected only {expected_values}"
    )


def test_senior_citizen_is_binary_flag(raw_df):
    assert set(raw_df["SeniorCitizen"].unique()) <= {0, 1}, (
        "SeniorCitizen should only contain 0 or 1"
    )


# ---------------------------------------------------------------------------
# Numeric range / sanity checks
# ---------------------------------------------------------------------------


def test_tenure_is_non_negative_and_reasonable(raw_df):
    assert (raw_df["tenure"] >= 0).all(), "Found negative tenure value(s)"
    # 10 years is a generous upper bound for a telecom subscription tenure
    # in months; anything beyond that likely signals a data issue.
    assert (raw_df["tenure"] <= 120).all(), (
        "Found implausibly large tenure value(s) (> 120 months)"
    )


def test_monthly_charges_is_positive_and_reasonable(raw_df):
    assert (raw_df["MonthlyCharges"] > 0).all(), (
        "Found non-positive MonthlyCharges value(s)"
    )
    assert (raw_df["MonthlyCharges"] <= 500).all(), (
        "Found implausibly large MonthlyCharges value(s)"
    )


def test_total_charges_parses_to_numeric_except_known_blanks(raw_df):
    """TotalCharges is stored as text and has a small number of blank
    entries. Every non-blank value must parse as a non-negative number.
    """
    parsed = pd.to_numeric(raw_df["TotalCharges"], errors="coerce")
    unparseable = raw_df.loc[parsed.isna(), "TotalCharges"]
    # Every unparseable entry should be blank/whitespace, not garbage text.
    assert (unparseable.str.strip() == "").all(), (
        f"Found non-numeric, non-blank TotalCharges value(s): {unparseable.tolist()}"
    )
    assert (parsed.dropna() >= 0).all(), "Found negative TotalCharges value(s)"


def test_total_charges_blanks_only_occur_for_brand_new_customers(raw_df):
    """The known blank TotalCharges entries should only occur for customers
    with tenure == 0 (i.e. they haven't been billed yet) — if a blank shows
    up for a customer with tenure > 0, that's a genuine data quality issue.
    """
    parsed = pd.to_numeric(raw_df["TotalCharges"], errors="coerce")
    blank_rows = raw_df.loc[parsed.isna()]
    assert (blank_rows["tenure"] == 0).all(), (
        "Found blank TotalCharges for customer(s) with tenure > 0 — "
        "unexpected data quality issue:\n"
        f"{blank_rows.loc[blank_rows['tenure'] != 0, ['customerID', 'tenure']]}"
    )


def test_total_charges_roughly_consistent_with_tenure_and_monthly_charges(raw_df):
    """Loose sanity check: TotalCharges should not be wildly larger than
    tenure * MonthlyCharges. Note MonthlyCharges reflects the customer's
    *current* rate, which can differ from past billing (plan changes,
    promotions), so exact equality does not hold — empirically the ratio
    tops out around 1.6x in this dataset, so we use a generous 3x bound
    purely to catch genuine corruption (e.g. a shifted decimal point),
    not to enforce an exact billing formula.
    """
    parsed_total = pd.to_numeric(raw_df["TotalCharges"], errors="coerce").fillna(0)
    upper_bound = raw_df["tenure"] * raw_df["MonthlyCharges"] * 3
    # Only applies where tenure > 0; a tenure-0 customer's bound would be 0
    # and is already covered by test_total_charges_blanks_only_occur_for_brand_new_customers.
    violations = raw_df.loc[(raw_df["tenure"] > 0) & (parsed_total > upper_bound)]
    assert violations.empty, (
        f"Found {len(violations)} row(s) where TotalCharges greatly exceeds "
        f"tenure * MonthlyCharges: {violations['customerID'].tolist()[:5]}"
    )


# ---------------------------------------------------------------------------
# Target variable sanity checks
# ---------------------------------------------------------------------------


def test_churn_is_binary_yes_no(raw_df):
    assert set(raw_df["Churn"].unique()) == {"Yes", "No"}, (
        f"Churn should only contain 'Yes'/'No', found: {raw_df['Churn'].unique()}"
    )


def test_churn_rate_within_plausible_bounds(raw_df):
    """Loose sanity bound (not a strict business rule) to catch a badly
    corrupted or mis-joined dataset — e.g. a target that's almost all one
    class, or a coin-flip 50/50 split that doesn't match the known problem.
    """
    churn_rate = (raw_df["Churn"] == "Yes").mean()
    assert 0.05 < churn_rate < 0.60, (
        f"Churn rate {churn_rate:.2%} is outside plausible bounds"
    )


# ---------------------------------------------------------------------------
# Cross-field consistency checks
# ---------------------------------------------------------------------------


def test_no_phone_service_implies_no_multiple_lines(raw_df):
    """A customer with no phone service can't have 'Yes'/'No' for
    MultipleLines — it must be the sentinel 'No phone service'.
    """
    no_phone = raw_df["PhoneService"] == "No"
    invalid = raw_df.loc[no_phone & (raw_df["MultipleLines"] != "No phone service")]
    assert invalid.empty, (
        f"Found {len(invalid)} customer(s) with PhoneService='No' but "
        f"MultipleLines is not 'No phone service'"
    )


def test_no_internet_service_implies_no_internet_addons(raw_df):
    """A customer with InternetService == 'No' can't have 'Yes'/'No' for
    any internet add-on — it must be the sentinel 'No internet service'.
    """
    internet_addons = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    no_internet = raw_df["InternetService"] == "No"
    for col in internet_addons:
        invalid = raw_df.loc[no_internet & (raw_df[col] != "No internet service")]
        assert invalid.empty, (
            f"Found {len(invalid)} customer(s) with InternetService='No' but "
            f"{col} is not 'No internet service'"
        )
