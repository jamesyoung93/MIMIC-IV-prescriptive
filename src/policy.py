"""Policy evaluation and decision rules for MIMIC-IV Prescriptive ICU.

Optimal action computation, subgroup analysis with Bonferroni correction,
interpretable policy tree, and value-of-personalization estimates.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score


# ---------------------------------------------------------------------------
# Optimal actions
# ---------------------------------------------------------------------------

def compute_optimal_actions(cate: np.ndarray) -> np.ndarray:
    """Determine optimal treatment for each patient based on CATE.

    CATE = E[Y|A=1] - E[Y|A=0] in log-LOS space.
    Since Y = log(LOS) and lower is better:
    - CATE < 0 → liberal (A=1) is optimal (shorter LOS under liberal)
    - CATE >= 0 → restrictive (A=0) is optimal

    Returns:
        Array of optimal actions (0 or 1).
    """
    return (cate < 0).astype(int)


# ---------------------------------------------------------------------------
# Subgroup analysis
# ---------------------------------------------------------------------------

def subgroup_analysis(
    X: pd.DataFrame,
    cate: np.ndarray,
    optimal: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Analyze CATE heterogeneity across clinically meaningful subgroups.

    Tests: sepsis vs no-sepsis, high vs low SOFA, surgical vs medical,
    elderly vs younger. Reports Bonferroni-corrected significance.
    """
    n_tests = config["evaluation"]["bonferroni_n_tests"]
    alpha = 0.05 / n_tests  # Bonferroni threshold

    results = []

    # 1. Sepsis
    if "has_sepsis" in X.columns:
        mask = X["has_sepsis"] == 1
        results.append(_subgroup_row(
            "Sepsis", mask, cate, optimal, alpha
        ))
        results.append(_subgroup_row(
            "No Sepsis", ~mask, cate, optimal, alpha
        ))

    # 2. SOFA severity (median split)
    if "sofa_total" in X.columns:
        median_sofa = X["sofa_total"].median()
        mask = X["sofa_total"] >= median_sofa
        results.append(_subgroup_row(
            f"High SOFA (>= {median_sofa:.0f})", mask, cate, optimal, alpha
        ))
        results.append(_subgroup_row(
            f"Low SOFA (< {median_sofa:.0f})", ~mask, cate, optimal, alpha
        ))

    # 3. ICU type
    if "icu_unit_surgical" in X.columns:
        mask = X["icu_unit_surgical"] == 1
        results.append(_subgroup_row(
            "Surgical ICU", mask, cate, optimal, alpha
        ))
        results.append(_subgroup_row(
            "Medical ICU", ~mask, cate, optimal, alpha
        ))

    # 4. Age (median split)
    if "age" in X.columns:
        median_age = X["age"].median()
        mask = X["age"] >= median_age
        results.append(_subgroup_row(
            f"Elderly (>= {median_age:.0f}y)", mask, cate, optimal, alpha
        ))
        results.append(_subgroup_row(
            f"Younger (< {median_age:.0f}y)", ~mask, cate, optimal, alpha
        ))

    return pd.DataFrame(results)


def _subgroup_row(
    name: str,
    mask: pd.Series,
    cate: np.ndarray,
    optimal: np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    """Compute summary statistics for one subgroup."""
    from scipy.stats import ttest_1samp

    sub_cate = cate[mask.values]
    sub_optimal = optimal[mask.values]
    n = int(mask.sum())

    if n < 2:
        return {
            "subgroup": name,
            "n": n,
            "mean_cate": np.nan,
            "std_cate": np.nan,
            "pct_liberal_optimal": np.nan,
            "p_value": np.nan,
            "significant": False,
        }

    stat, p = ttest_1samp(sub_cate, 0)

    return {
        "subgroup": name,
        "n": n,
        "mean_cate": float(np.mean(sub_cate)),
        "std_cate": float(np.std(sub_cate)),
        "pct_liberal_optimal": float(sub_optimal.mean()),
        "p_value": float(p),
        "significant": p < alpha,
    }


# ---------------------------------------------------------------------------
# Policy tree
# ---------------------------------------------------------------------------

def fit_policy_tree(
    X: pd.DataFrame,
    optimal: np.ndarray,
    config: dict[str, Any],
) -> tuple[DecisionTreeClassifier, float]:
    """Fit an interpretable depth-limited decision tree on optimal actions.

    Uses LOOCV accuracy to evaluate the tree's ability to recover
    the personalized policy.

    Returns:
        (fitted_tree, loocv_accuracy)
    """
    max_depth = config["evaluation"]["policy_tree_max_depth"]

    # Full model for interpretation
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree.fit(X, optimal)

    # LOOCV accuracy
    loo = LeaveOneOut()
    scores = cross_val_score(
        DecisionTreeClassifier(max_depth=max_depth, random_state=42),
        X, optimal, cv=loo, scoring="accuracy",
    )
    accuracy = scores.mean()

    return tree, accuracy


# ---------------------------------------------------------------------------
# Value of personalization
# ---------------------------------------------------------------------------

def compute_personalization_value(
    mu0: np.ndarray,
    mu1: np.ndarray,
    A: pd.Series | np.ndarray,
    Y: pd.Series | np.ndarray,
    optimal: np.ndarray,
) -> dict[str, Any]:
    """Estimate the value of personalized treatment vs. uniform policies.

    Computes mean predicted LOS (exponentiated from log scale) under:
    1. Observed treatment
    2. Always liberal
    3. Always restrictive
    4. Personalized (CATE-optimal)

    Returns dict with mean LOS in days under each policy.
    """
    Y = np.asarray(Y)
    A = np.asarray(A)

    # Predicted outcomes under each policy
    predicted_observed = np.where(A == 1, mu1, mu0)
    predicted_liberal = mu1
    predicted_restrictive = mu0
    predicted_personalized = np.where(optimal == 1, mu1, mu0)

    # Convert from log-LOS to LOS in days
    return {
        "observed_mean_los": float(np.exp(predicted_observed).mean()),
        "always_liberal_mean_los": float(np.exp(predicted_liberal).mean()),
        "always_restrictive_mean_los": float(np.exp(predicted_restrictive).mean()),
        "personalized_mean_los": float(np.exp(predicted_personalized).mean()),
        "observed_median_los": float(np.exp(predicted_observed.mean())),
        "always_liberal_median_los": float(np.exp(predicted_liberal.mean())),
        "always_restrictive_median_los": float(np.exp(predicted_restrictive.mean())),
        "personalized_median_los": float(np.exp(predicted_personalized.mean())),
        "pct_would_switch": float((optimal != A).mean()),
        "n_liberal_optimal": int(optimal.sum()),
        "n_restrictive_optimal": int((optimal == 0).sum()),
    }
