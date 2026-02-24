"""Meta-learner models and evaluation for MIMIC-IV Prescriptive ICU.

Implements T-Learner and S-Learner with XGBoost for CATE estimation,
Leave-One-Out Cross-Validation (LOOCV) for nuisance model evaluation,
permutation testing for CATE heterogeneity, and AUTOC ranking metric.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


def regression_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Compute standard regression metrics as a dictionary."""
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "pearson_r": pearsonr(y_true, y_pred).statistic,
        "spearman_rho": spearmanr(y_true, y_pred).statistic,
        "n_samples": len(y_true),
    }


# ---------------------------------------------------------------------------
# Meta-learners
# ---------------------------------------------------------------------------

class TLearner:
    """T-Learner: separate XGBoost models for each treatment arm."""

    def __init__(self, xgb_params: dict[str, Any] | None = None) -> None:
        self.xgb_params = xgb_params or {}
        self.model_0: XGBRegressor | None = None
        self.model_1: XGBRegressor | None = None

    def fit(self, X: pd.DataFrame, A: pd.Series, Y: pd.Series) -> "TLearner":
        """Fit separate models on control (A=0) and treated (A=1) subsets."""
        mask_0 = A == 0
        mask_1 = A == 1

        self.model_0 = XGBRegressor(**self.xgb_params)
        self.model_1 = XGBRegressor(**self.xgb_params)

        self.model_0.fit(X[mask_0], Y[mask_0])
        self.model_1.fit(X[mask_1], Y[mask_1])
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        """CATE = E[Y|X, A=1] - E[Y|X, A=0] (liberal minus restrictive)."""
        mu0 = self.model_0.predict(X)
        mu1 = self.model_1.predict(X)
        return mu1 - mu0

    def predict_outcome(self, X: pd.DataFrame, A: pd.Series) -> np.ndarray:
        """Predict outcome routing each patient to the correct arm model."""
        mu0 = self.model_0.predict(X)
        mu1 = self.model_1.predict(X)
        return np.where(A == 1, mu1, mu0)

    def predict_mu(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (mu0, mu1) for all patients."""
        return self.model_0.predict(X), self.model_1.predict(X)

    def get_models(self) -> dict[str, XGBRegressor]:
        """Return underlying XGBoost models."""
        return {"model_0": self.model_0, "model_1": self.model_1}


class SLearner:
    """S-Learner: single XGBoost model with treatment as a feature."""

    def __init__(self, xgb_params: dict[str, Any] | None = None) -> None:
        self.xgb_params = xgb_params or {}
        self.model: XGBRegressor | None = None

    def _augment(self, X: pd.DataFrame, A: pd.Series | int) -> pd.DataFrame:
        """Append treatment column to feature matrix."""
        Xa = X.copy()
        Xa["treatment"] = A
        return Xa

    def fit(self, X: pd.DataFrame, A: pd.Series, Y: pd.Series) -> "SLearner":
        """Fit a single model on X augmented with treatment indicator."""
        Xa = self._augment(X, A)
        self.model = XGBRegressor(**self.xgb_params)
        self.model.fit(Xa, Y)
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        """CATE = E[Y|X, A=1] - E[Y|X, A=0]."""
        mu0 = self.model.predict(self._augment(X, 0))
        mu1 = self.model.predict(self._augment(X, 1))
        return mu1 - mu0

    def predict_outcome(self, X: pd.DataFrame, A: pd.Series) -> np.ndarray:
        """Predict outcome using actual treatment assignment."""
        return self.model.predict(self._augment(X, A))

    def predict_mu(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (mu0, mu1) for all patients."""
        mu0 = self.model.predict(self._augment(X, 0))
        mu1 = self.model.predict(self._augment(X, 1))
        return mu0, mu1

    def get_models(self) -> dict[str, XGBRegressor]:
        """Return underlying XGBoost model."""
        return {"model": self.model}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class LOOCVResult:
    """Container for LOOCV results."""

    y_true: np.ndarray
    y_pred: np.ndarray
    cate_estimates: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray
    metrics: dict[str, Any] = field(default_factory=dict)


def run_loocv(
    X: pd.DataFrame,
    A: pd.Series,
    Y: pd.Series,
    learner_class: type,
    config: dict[str, Any],
) -> LOOCVResult:
    """Leave-One-Out Cross-Validation for nuisance model evaluation.

    For each patient i, trains on all other patients and predicts for i.
    Returns observed-arm predictions (for calibration) and full CATE estimates.
    """
    xgb_params = config["model"]["xgboost"].copy()
    xgb_params["verbosity"] = 0
    n = len(X)

    y_pred = np.zeros(n)
    cate_all = np.zeros(n)
    mu0_all = np.zeros(n)
    mu1_all = np.zeros(n)
    y_true = Y.values

    idx = X.index

    for i in range(n):
        # Leave one out
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False

        X_train = X.iloc[train_mask]
        A_train = A.iloc[train_mask]
        Y_train = Y.iloc[train_mask]
        X_test = X.iloc[[i]]
        A_test = A.iloc[[i]]

        learner = learner_class(xgb_params)
        learner.fit(X_train, A_train, Y_train)

        y_pred[i] = learner.predict_outcome(X_test, A_test)[0]
        cate_all[i] = learner.predict_cate(X_test)[0]
        mu0, mu1 = learner.predict_mu(X_test)
        mu0_all[i] = mu0[0]
        mu1_all[i] = mu1[0]

    metrics = regression_report_dict(y_true, y_pred)

    return LOOCVResult(
        y_true=y_true,
        y_pred=y_pred,
        cate_estimates=cate_all,
        mu0=mu0_all,
        mu1=mu1_all,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Permutation test for CATE heterogeneity
# ---------------------------------------------------------------------------

def permutation_test(
    X: pd.DataFrame,
    A: pd.Series,
    Y: pd.Series,
    learner_class: type,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Permutation test for heterogeneous treatment effects.

    Null hypothesis: treatment effects are homogeneous (no CATE variation).
    Test statistic: variance of CATE estimates across patients.

    Under the null, we permute treatment labels, re-run the LOOCV CATE
    estimation, and compare observed Var(CATE) to the null distribution.
    """
    n_perms = config["evaluation"]["n_permutations"]
    seed = config["evaluation"]["permutation_seed"]
    rng = np.random.default_rng(seed)

    # Observed statistic
    observed_result = run_loocv(X, A, Y, learner_class, config)
    observed_stat = np.var(observed_result.cate_estimates)

    # Null distribution
    null_stats = np.zeros(n_perms)
    for p in range(n_perms):
        A_perm = A.sample(frac=1, random_state=rng.integers(1e9)).values
        A_perm = pd.Series(A_perm, index=A.index, name=A.name)

        perm_result = run_loocv(X, A_perm, Y, learner_class, config)
        null_stats[p] = np.var(perm_result.cate_estimates)

    p_value = (null_stats >= observed_stat).mean()

    return {
        "observed_stat": observed_stat,
        "null_distribution": null_stats,
        "p_value": p_value,
        "n_permutations": n_perms,
    }


# ---------------------------------------------------------------------------
# AUTOC: Area Under the TOC Curve
# ---------------------------------------------------------------------------

def compute_autoc(
    Y: pd.Series | np.ndarray,
    A: pd.Series | np.ndarray,
    cate_estimates: np.ndarray,
) -> float:
    """Compute Area Under the TOC (Targeting Operator Characteristic) curve.

    Ranks patients by predicted CATE (descending benefit from treatment=1),
    computes cumulative average treatment effect as we target more patients.
    A positive AUTOC means ranking by CATE successfully identifies patients
    who benefit most from treatment=1.

    Reference: Yadlowsky et al. (2021).
    """
    Y = np.asarray(Y)
    A = np.asarray(A)

    n = len(Y)
    # Sort by CATE descending (most benefit from liberal first)
    order = np.argsort(-cate_estimates)
    Y_sorted = Y[order]
    A_sorted = A[order]

    # Cumulative average treatment effect by fraction targeted
    # Using IPW-style estimate: E[Y*A/e - Y*(1-A)/(1-e)] per stratum
    # For median-split binary treatment, e=0.5 (approx)
    e = A.mean()
    ipw = Y * A / e - Y * (1 - A) / (1 - e)
    ipw_sorted = ipw[order]

    # Cumulative mean IPW estimate
    cum_ate = np.cumsum(ipw_sorted) / np.arange(1, n + 1)

    # Overall ATE
    overall_ate = ipw.mean()

    # AUTOC = integral of (cumulative ATE - overall ATE) over fraction targeted
    fractions = np.arange(1, n + 1) / n
    autoc = np.trapezoid(cum_ate - overall_ate, fractions)

    return float(autoc)
