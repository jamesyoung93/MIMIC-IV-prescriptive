"""Pipeline runner for MIMIC-IV Prescriptive ICU.

Orchestrates extract → model → interpret → policy, generates figures,
saves results for Streamlit consumption, and logs experiment to LOGBOOK.
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Path setup
_project_root = Path(__file__).parent.parent
_src = str(_project_root / "src")
_research_root = str(_project_root.parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)
if _research_root not in sys.path:
    sys.path.insert(0, _research_root)

from extract import load_config, build_dataset
from models import TLearner, SLearner, run_loocv, permutation_test, compute_autoc
from interpret import (
    compute_shap_observed,
    compute_shap_cate_surrogate,
    find_knn_comparators,
    select_representative_patients,
)
from policy import (
    compute_optimal_actions,
    subgroup_analysis,
    fit_policy_tree,
    compute_personalization_value,
)
from commons.utils.evaluation import log_experiment
from commons.utils.plotting import apply_pub_style, save_figure


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def _fig_cate_distribution(cate_t: np.ndarray, cate_s: np.ndarray, path: Path) -> None:
    """CATE distribution for T-Learner and S-Learner."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, cate, name in zip(axes, [cate_t, cate_s], ["T-Learner", "S-Learner"]):
        ax.hist(cate, bins=15, edgecolor="black", alpha=0.7, color="#4C72B0")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="No effect")
        ax.axvline(np.median(cate), color="orange", linestyle="-", linewidth=1.5,
                   label=f"Median = {np.median(cate):.3f}")
        ax.set_xlabel("CATE (log-LOS: liberal - restrictive)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name} CATE Distribution")
        ax.legend(fontsize=9)

    save_figure(fig, path)
    plt.close(fig)


def _fig_loocv_calibration(result_t, result_s, path: Path) -> None:
    """Observed vs predicted scatter for LOOCV."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, result, name in zip(axes, [result_t, result_s], ["T-Learner", "S-Learner"]):
        ax.scatter(result.y_true, result.y_pred, alpha=0.7, edgecolors="black", linewidth=0.5)
        lims = [min(result.y_true.min(), result.y_pred.min()) - 0.1,
                max(result.y_true.max(), result.y_pred.max()) + 0.1]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect calibration")
        ax.set_xlabel("Observed log(LOS)")
        ax.set_ylabel("Predicted log(LOS)")
        ax.set_title(f"{name} LOOCV (R² = {result.metrics['r2']:.3f})")
        ax.legend(fontsize=9)

    save_figure(fig, path)
    plt.close(fig)


def _fig_treatment_vs_outcome(A, Y, path: Path) -> None:
    """Box plot: treatment group vs log-LOS."""
    fig, ax = plt.subplots(figsize=(8, 5))
    data = {"Treatment": ["Liberal" if a == 1 else "Restrictive" for a in A],
            "log(ICU LOS)": Y.values}
    import pandas as pd
    df = pd.DataFrame(data)
    sns.boxplot(data=df, x="Treatment", y="log(ICU LOS)", hue="Treatment",
                ax=ax, palette=["#DD8452", "#4C72B0"], legend=False)
    sns.stripplot(data=df, x="Treatment", y="log(ICU LOS)", ax=ax,
                  color="black", alpha=0.4, size=5)
    ax.set_title("ICU Length of Stay by Treatment Group")
    save_figure(fig, path)
    plt.close(fig)


def _fig_personalization_value(pv: dict, path: Path) -> None:
    """Bar chart comparing mean LOS under different policies."""
    fig, ax = plt.subplots(figsize=(8, 5))
    policies = ["Observed", "Always Liberal", "Always Restrictive", "Personalized"]
    values = [
        pv["observed_mean_los"],
        pv["always_liberal_mean_los"],
        pv["always_restrictive_mean_los"],
        pv["personalized_mean_los"],
    ]
    colors = ["#7f7f7f", "#DD8452", "#4C72B0", "#55A868"]
    bars = ax.bar(policies, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Mean Predicted ICU LOS (days)")
    ax.set_title("Value of Personalized Fluid Management")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}d", ha="center", va="bottom", fontsize=10)

    save_figure(fig, path)
    plt.close(fig)


def _fig_subgroup_forest(subgroups, path: Path) -> None:
    """Forest plot of subgroup mean CATE with CIs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    names = subgroups["subgroup"].values[::-1]
    means = subgroups["mean_cate"].values[::-1]
    stds = subgroups["std_cate"].values[::-1]
    ns = subgroups["n"].values[::-1]
    sigs = subgroups["significant"].values[::-1]

    # 95% CI = mean +/- 1.96*std/sqrt(n)
    ci = 1.96 * stds / np.sqrt(ns)

    y_pos = np.arange(len(names))
    colors = ["#E24A33" if s else "#348ABD" for s in sigs]

    ax.errorbar(means, y_pos, xerr=ci, fmt="o", color="black",
                ecolor="gray", elinewidth=1.5, capsize=4, markersize=6)
    for i, (m, c) in enumerate(zip(means, colors)):
        ax.scatter(m, y_pos[i], color=c, s=80, zorder=5)

    ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean CATE (log-LOS)")
    ax.set_title("Subgroup Treatment Effect Heterogeneity")

    save_figure(fig, path)
    plt.close(fig)


def _fig_policy_tree(tree, feature_names, path: Path) -> None:
    """Visualize the depth-2 policy tree."""
    from sklearn.tree import plot_tree

    fig, ax = plt.subplots(figsize=(14, 8))
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=["Restrictive", "Liberal"],
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax,
    )
    ax.set_title("Interpretable Policy Tree (Depth-2)")
    save_figure(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    config_path: str | Path = "configs/default.yaml",
    save_results: bool = True,
    generate_figures: bool = True,
    run_perm_test: bool = True,
) -> dict[str, Any]:
    """Run the full prescriptive ML pipeline.

    Args:
        config_path: Path to YAML configuration.
        save_results: Whether to pickle results for Streamlit.
        generate_figures: Whether to generate publication figures.
        run_perm_test: Whether to run permutation test (~17 min).

    Returns:
        Dictionary of all results.
    """
    apply_pub_style()

    # 1. Extract
    print("=" * 60)
    print("Phase 1: Data Extraction")
    print("=" * 60)
    config = load_config(config_path)
    X, A, Y, patient_info = build_dataset(config, cohort_type="primary")
    print(f"  Cohort: n={len(X)}, features={X.shape[1]}")
    print(f"  Treatment: {(A==1).sum()} liberal / {(A==0).sum()} restrictive")
    print(f"  Outcome: log(LOS), range [{Y.min():.2f}, {Y.max():.2f}]")

    results: dict[str, Any] = {
        "config": config,
        "X": X,
        "A": A,
        "Y": Y,
        "patient_info": patient_info,
    }

    # 2. Model: T-Learner
    print("\n" + "=" * 60)
    print("Phase 2: T-Learner LOOCV")
    print("=" * 60)
    result_t = run_loocv(X, A, Y, TLearner, config)
    print(f"  R² = {result_t.metrics['r2']:.4f}")
    print(f"  RMSE = {result_t.metrics['rmse']:.4f}")
    print(f"  Pearson r = {result_t.metrics['pearson_r']:.4f}")
    results["loocv_t"] = result_t

    # 3. Model: S-Learner
    print("\n" + "=" * 60)
    print("Phase 3: S-Learner LOOCV")
    print("=" * 60)
    result_s = run_loocv(X, A, Y, SLearner, config)
    print(f"  R² = {result_s.metrics['r2']:.4f}")
    print(f"  RMSE = {result_s.metrics['rmse']:.4f}")
    print(f"  Pearson r = {result_s.metrics['pearson_r']:.4f}")
    results["loocv_s"] = result_s

    # AUTOC
    autoc_t = compute_autoc(Y, A, result_t.cate_estimates)
    autoc_s = compute_autoc(Y, A, result_s.cate_estimates)
    print(f"\n  AUTOC: T-Learner = {autoc_t:.4f}, S-Learner = {autoc_s:.4f}")
    results["autoc_t"] = autoc_t
    results["autoc_s"] = autoc_s

    # Select primary learner (higher AUTOC)
    if autoc_t >= autoc_s:
        primary_result = result_t
        primary_name = "T-Learner"
        primary_class = TLearner
    else:
        primary_result = result_s
        primary_name = "S-Learner"
        primary_class = SLearner
    print(f"\n  Primary learner: {primary_name}")
    results["primary_name"] = primary_name

    # Fit full model on all data for SHAP
    xgb_params = config["model"]["xgboost"].copy()
    xgb_params["verbosity"] = 0
    full_learner = primary_class(xgb_params)
    full_learner.fit(X, A, Y)
    results["full_learner"] = full_learner

    # 4. Interpretability
    print("\n" + "=" * 60)
    print("Phase 4: Interpretability")
    print("=" * 60)

    # SHAP observed
    print("  Computing SHAP values (observed outcomes)...")
    shap_obs = compute_shap_observed(full_learner, X, A)
    results["shap_observed"] = shap_obs

    # SHAP CATE surrogate
    print("  Fitting CATE surrogate model...")
    cate = primary_result.cate_estimates
    surrogate, shap_cate, surrogate_r2 = compute_shap_cate_surrogate(X, cate, config)
    print(f"  Surrogate R² = {surrogate_r2:.4f}")
    results["shap_cate"] = shap_cate
    results["surrogate"] = surrogate
    results["surrogate_r2"] = surrogate_r2

    # KNN comparators
    print("  Finding KNN clinical comparators...")
    knn = find_knn_comparators(X, A, Y, cate, config, patient_info)
    results["knn"] = knn

    # Representative patients
    reps = select_representative_patients(cate, X.index.values)
    results["representative_patients"] = reps
    print(f"  Representatives: {reps}")

    # 5. Policy
    print("\n" + "=" * 60)
    print("Phase 5: Policy Analysis")
    print("=" * 60)

    optimal = compute_optimal_actions(cate)
    results["optimal_actions"] = optimal
    print(f"  Optimal: {optimal.sum()} liberal / {(optimal==0).sum()} restrictive")

    # Subgroup analysis
    subgroups = subgroup_analysis(X, cate, optimal, config)
    results["subgroups"] = subgroups
    print("\n  Subgroup analysis:")
    print(subgroups.to_string(index=False))

    # Policy tree
    tree, tree_acc = fit_policy_tree(X, optimal, config)
    results["policy_tree"] = tree
    results["policy_tree_accuracy"] = tree_acc
    print(f"\n  Policy tree LOOCV accuracy: {tree_acc:.4f}")

    # Value of personalization
    pv = compute_personalization_value(
        primary_result.mu0, primary_result.mu1,
        A, Y, optimal,
    )
    results["personalization_value"] = pv
    print(f"\n  Mean predicted LOS:")
    print(f"    Observed:      {pv['observed_mean_los']:.2f} days")
    print(f"    Always Liberal: {pv['always_liberal_mean_los']:.2f} days")
    print(f"    Always Restr:  {pv['always_restrictive_mean_los']:.2f} days")
    print(f"    Personalized:  {pv['personalized_mean_los']:.2f} days")
    print(f"    % would switch: {pv['pct_would_switch']:.1%}")

    # 6. Permutation test (optional — slow)
    if run_perm_test:
        print("\n" + "=" * 60)
        print("Phase 6: Permutation Test (this may take several minutes)")
        print("=" * 60)
        perm = permutation_test(X, A, Y, primary_class, config)
        results["permutation_test"] = perm
        print(f"  Observed Var(CATE) = {perm['observed_stat']:.6f}")
        print(f"  Permutation p-value = {perm['p_value']:.4f}")
    else:
        print("\n  [Skipping permutation test (--no-permutation)]")

    # 7. Figures
    if generate_figures:
        print("\n" + "=" * 60)
        print("Phase 7: Generating Figures")
        print("=" * 60)
        fig_dir = Path(config["paths"]["figures_dir"])
        fig_dir.mkdir(parents=True, exist_ok=True)

        _fig_cate_distribution(
            result_t.cate_estimates, result_s.cate_estimates,
            fig_dir / "cate_distribution",
        )
        print("  [1/6] CATE distribution")

        _fig_loocv_calibration(result_t, result_s, fig_dir / "loocv_calibration")
        print("  [2/6] LOOCV calibration")

        _fig_treatment_vs_outcome(A, Y, fig_dir / "treatment_vs_outcome")
        print("  [3/6] Treatment vs outcome")

        _fig_personalization_value(pv, fig_dir / "personalization_value")
        print("  [4/6] Personalization value")

        _fig_subgroup_forest(subgroups, fig_dir / "subgroup_forest")
        print("  [5/6] Subgroup forest plot")

        _fig_policy_tree(tree, list(X.columns), fig_dir / "policy_tree")
        print("  [6/6] Policy tree")

    # 8. Save results
    if save_results:
        results_dir = Path(config["paths"]["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        cache_path = Path(config["paths"]["app_cache"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(results, f)
        print(f"\n  Results saved to {cache_path}")

    # 9. Log experiment
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    metrics = {
        "n_patients": len(X),
        "n_features": X.shape[1],
        "t_learner_r2": result_t.metrics["r2"],
        "s_learner_r2": result_s.metrics["r2"],
        "autoc_t": autoc_t,
        "autoc_s": autoc_s,
        "primary_learner": primary_name,
        "surrogate_r2": surrogate_r2,
        "policy_tree_accuracy": tree_acc,
    }
    if run_perm_test:
        metrics["permutation_p_value"] = perm["p_value"]

    log_experiment(
        logbook_path=str(_project_root / config["paths"]["logbook"]),
        title="Full pipeline run (primary cohort, LOS >= 2d)",
        hypothesis="T-learner/S-learner meta-learners can detect heterogeneous "
                   "treatment effects of liberal vs. restrictive fluid management "
                   "on ICU length of stay.",
        result=f"Primary learner: {primary_name}. "
               f"LOOCV R²={primary_result.metrics['r2']:.4f}. "
               f"AUTOC={max(autoc_t, autoc_s):.4f}. "
               f"Surrogate R²={surrogate_r2:.4f}. "
               f"{pv['pct_would_switch']:.0%} patients would switch treatment.",
        interpretation="Results from 100-patient demo cohort. Proof-of-concept "
                       "demonstrating feasibility of personalized fluid management. "
                       "Statistical power limited by sample size; federation with "
                       "partner sites is required for clinical conclusions.",
        next_steps="Partner recruitment via Streamlit app. Validate on full "
                   "MIMIC-IV (>50k patients). Multi-site federated analysis.",
        metrics=metrics,
    )

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line interface for the pipeline."""
    parser = argparse.ArgumentParser(
        description="MIMIC-IV Prescriptive ICU Pipeline",
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--no-permutation", action="store_true",
        help="Skip the permutation test (saves ~17 minutes)",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Skip figure generation",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save results to disk",
    )
    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        save_results=not args.no_save,
        generate_figures=not args.no_figures,
        run_perm_test=not args.no_permutation,
    )


if __name__ == "__main__":
    main()
