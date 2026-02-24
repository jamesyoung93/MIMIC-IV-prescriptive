"""Interpretability module for MIMIC-IV Prescriptive ICU.

SHAP-based explanations for observed outcomes and CATE estimates,
KNN clinical comparators, and representative patient selection.
"""

from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# SHAP: Observed outcome explanations
# ---------------------------------------------------------------------------

def compute_shap_observed(
    learner: Any,
    X: pd.DataFrame,
    A: pd.Series,
) -> shap.Explanation:
    """Compute SHAP values routing each patient to the correct arm model.

    For T-Learner: routes each patient to model_0 or model_1 based on A.
    For S-Learner: uses the single model with treatment appended.
    """
    models = learner.get_models()

    if "model_0" in models and "model_1" in models:
        # T-Learner: route to correct arm
        explainer_0 = shap.TreeExplainer(models["model_0"])
        explainer_1 = shap.TreeExplainer(models["model_1"])

        shap_0 = explainer_0(X[A == 0])
        shap_1 = explainer_1(X[A == 1])

        # Combine into single explanation aligned with original index
        values = np.zeros((len(X), X.shape[1]))
        base_values = np.zeros(len(X))

        mask_0 = (A == 0).values
        mask_1 = (A == 1).values
        values[mask_0] = shap_0.values
        values[mask_1] = shap_1.values
        base_values[mask_0] = shap_0.base_values
        base_values[mask_1] = shap_1.base_values

        return shap.Explanation(
            values=values,
            base_values=base_values,
            data=X.values,
            feature_names=list(X.columns),
        )
    else:
        # S-Learner: single model
        Xa = X.copy()
        Xa["treatment"] = A.values
        explainer = shap.TreeExplainer(models["model"])
        return explainer(Xa)


# ---------------------------------------------------------------------------
# SHAP: CATE surrogate model (Guo et al. 2025)
# ---------------------------------------------------------------------------

def compute_shap_cate_surrogate(
    X: pd.DataFrame,
    cate: np.ndarray,
    config: dict[str, Any],
) -> tuple[XGBRegressor, shap.Explanation, float]:
    """Fit a shallow surrogate to CATE estimates and compute SHAP on it.

    Instead of naively differencing SHAP_mu1 - SHAP_mu0 (which violates
    Shapley additivity axioms), we fit a shallow XGBRegressor to the CATE
    estimates and compute SHAP on the surrogate.

    Returns:
        (surrogate_model, shap_explanation, r2_score)
    """
    from sklearn.metrics import r2_score

    surrogate_params = config["model"]["cate_surrogate"].copy()
    surrogate_params["verbosity"] = 0

    surrogate = XGBRegressor(**surrogate_params)
    surrogate.fit(X, cate)

    pred = surrogate.predict(X)
    r2 = r2_score(cate, pred)

    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer(X)

    return surrogate, shap_values, r2


# ---------------------------------------------------------------------------
# KNN clinical comparators
# ---------------------------------------------------------------------------

def find_knn_comparators(
    X: pd.DataFrame,
    A: pd.Series,
    Y: pd.Series,
    cate: np.ndarray,
    config: dict[str, Any],
    patient_info: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    """Find K nearest neighbors for each patient in standardized space.

    Returns a dict mapping stay_id → DataFrame of K nearest neighbors
    with their covariates, treatment, outcome, and CATE.
    """
    k = config["evaluation"]["knn_k"]
    metric = config["evaluation"]["knn_metric"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(X_scaled)

    result = {}
    stay_ids = X.index.values

    for i, sid in enumerate(stay_ids):
        dists, indices = nn.kneighbors(X_scaled[[i]])
        # Exclude self (first neighbor)
        neighbor_idx = indices[0][1:]
        neighbor_dists = dists[0][1:]
        neighbor_sids = stay_ids[neighbor_idx]

        rows = []
        for j, (nsid, dist) in enumerate(zip(neighbor_sids, neighbor_dists)):
            info = patient_info.loc[nsid]
            row = {
                "stay_id": nsid,
                "distance": dist,
                "treatment": "Liberal" if A.loc[nsid] == 1 else "Restrictive",
                "log_los": Y.loc[nsid],
                "los_days": info["los"],
                "cate": cate[np.where(stay_ids == nsid)[0][0]],
                "age": info["age"],
                "gender": info["gender"],
            }
            rows.append(row)

        result[sid] = pd.DataFrame(rows)

    return result


# ---------------------------------------------------------------------------
# SHAP waterfall figure generation
# ---------------------------------------------------------------------------

def generate_shap_waterfall_figure(
    shap_values: shap.Explanation,
    idx: int,
    title: str = "",
    max_display: int = 15,
):
    """Generate a SHAP waterfall plot for a single patient.

    Returns a matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[idx], max_display=max_display, show=False)
    if title:
        plt.title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    return plt.gcf()


# ---------------------------------------------------------------------------
# Representative patient selection
# ---------------------------------------------------------------------------

def select_representative_patients(
    cate: np.ndarray,
    stay_ids: np.ndarray,
) -> dict[str, int]:
    """Select representative patients for detailed display.

    Returns dict with stay_ids for:
    - most_benefit_liberal: lowest CATE (most benefit from switching to liberal)
      Note: CATE = E[Y|A=1] - E[Y|A=0] in log-LOS space.
      Negative CATE → liberal leads to shorter LOS.
    - most_benefit_restrictive: highest CATE (liberal increases LOS most)
    - median_effect: patient closest to median CATE
    """
    median_cate = np.median(cate)
    median_idx = np.argmin(np.abs(cate - median_cate))

    return {
        "most_benefit_liberal": stay_ids[np.argmin(cate)],
        "most_benefit_restrictive": stay_ids[np.argmax(cate)],
        "median_effect": stay_ids[median_idx],
    }
