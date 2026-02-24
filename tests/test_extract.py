"""Tests for the data extraction pipeline."""

import numpy as np
import pandas as pd


def test_cohort_size(dataset):
    """Primary cohort should have ~53 patients (LOS >= 2d, first stay)."""
    X, A, Y, info = dataset
    assert 45 <= len(X) <= 60, f"Cohort size {len(X)} outside expected range"


def test_each_subject_appears_once(dataset, tables):
    """Each subject_id appears exactly once in the cohort."""
    _, _, _, info = dataset
    assert info["subject_id"].is_unique


def test_los_threshold(dataset, config):
    """All patients have LOS >= primary threshold."""
    _, _, _, info = dataset
    min_days = config["cohort"]["primary_min_los_days"]
    assert (info["los"] >= min_days).all()


def test_treatment_balance(dataset):
    """Treatment split should be approximately 50/50 for median split."""
    _, A, _, _ = dataset
    pct_liberal = A.mean()
    assert 0.35 <= pct_liberal <= 0.65, f"Treatment imbalance: {pct_liberal:.2%} liberal"


def test_treatment_binary(dataset):
    """Treatment should be 0/1 only."""
    _, A, _, _ = dataset
    assert set(A.unique()) == {0, 1}


def test_outcome_no_nan(dataset):
    """Outcome (log LOS) should have no NaN values."""
    _, _, Y, _ = dataset
    assert Y.notna().all()


def test_feature_matrix_no_nan(dataset):
    """After imputation, X should have zero NaN values."""
    X, _, _, _ = dataset
    assert X.isna().sum().sum() == 0, f"Found NaN values in X"


def test_feature_matrix_has_expected_columns(dataset):
    """X should contain key expected features from curated core set."""
    X, _, _, _ = dataset
    expected = ["age", "gender_male", "has_sepsis", "gcs_total", "sofa_total",
                "lactate_first_6h", "creatinine_first_6h", "map_first_6h"]
    for col in expected:
        assert col in X.columns, f"Missing expected column: {col}"
    # Feature curation drops redundant SOFA/GCS components and sparse labs
    assert X.shape[1] <= 50, f"Too many features ({X.shape[1]}); expected curated set"
    assert X.shape[1] >= 30, f"Too few features ({X.shape[1]}); check curation"


def test_sofa_total_range(dataset):
    """SOFA total score should be in [0, 24]."""
    X, _, _, _ = dataset
    assert "sofa_total" in X.columns, "Missing sofa_total"
    assert X["sofa_total"].min() >= 0
    assert X["sofa_total"].max() <= 24


def test_missingness_indicators_binary(dataset):
    """All missingness indicator columns are 0/1 only."""
    X, _, _, _ = dataset
    miss_cols = [c for c in X.columns if c.endswith("_missing")]
    for col in miss_cols:
        assert set(X[col].unique()).issubset({0, 1}), f"{col} not binary"


def test_gcs_total_present(dataset):
    """GCS total should be present (components dropped in curated set)."""
    X, _, _, _ = dataset
    assert "gcs_total" in X.columns, "Missing gcs_total"


def test_map_column_present(dataset):
    """MAP should have a combined arterial+NIBP column."""
    X, _, _, _ = dataset
    map_cols = [c for c in X.columns if "map" in c.lower() and "missing" not in c]
    assert len(map_cols) > 0, "No MAP column found"
