"""Streamlit proof-of-concept app for MIMIC-IV Prescriptive ICU.

4-page dashboard:
  1. About — Disclaimer, methodology, cohort summary, metric guide
  2. Patient Explorer — Individual SHAP explanations, KNN comparators
  3. Population Insights — CATE distribution, policy tree, subgroups
  4. Partner With Us — Partnership brief, federated learning, roadmap

Loads pre-computed results from results/pipeline_results.pkl.
All computation is local — no internet required.
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

# Path setup
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
RESULTS_PATH = PROJECT_ROOT / "results" / "pipeline_results.pkl"
ASSETS_DIR = APP_DIR / "assets"
FIGURES_DIR = PROJECT_ROOT / "figures"

_src = str(PROJECT_ROOT / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_results() -> dict:
    """Load pre-computed pipeline results."""
    if not RESULTS_PATH.exists():
        st.error(
            f"Results file not found at `{RESULTS_PATH}`.\n\n"
            "Run the pipeline first:\n"
            "```\npython -m src.main --no-permutation\n```"
        )
        st.stop()
    with open(RESULTS_PATH, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cate_to_hours(cate_log: float, baseline_los_days: float = 5.0) -> float:
    """Convert CATE in log-LOS space to approximate hours difference."""
    los_liberal = baseline_los_days * np.exp(cate_log)
    diff_days = los_liberal - baseline_los_days
    return diff_days * 24


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------

def page_about(results: dict) -> None:
    """About page with disclaimer, methodology, and cohort summary."""
    st.title("Prescriptive ICU: Exploring Personalized Fluid Management")

    st.warning(
        "**Research Proof-of-Concept** — This application demonstrates a "
        "prescriptive ML framework on the MIMIC-IV Clinical Database Demo "
        "(100 patients, MIT open data). It is NOT a medical device and should "
        "NOT be used for clinical decision-making. All results are model "
        "estimates from a small retrospective cohort and require independent "
        "validation before any clinical application."
    )

    st.markdown("---")

    st.header("What This Does")
    st.markdown("""
    This tool explores whether **different ICU patients might respond differently**
    to liberal vs. restrictive fluid management, as measured by ICU length of stay.

    Rather than asking "which fluid strategy is better on average?", it asks:

    > **Does the model detect patterns suggesting some patients respond differently
    > to fluid management strategies?**

    The framework uses **meta-learner** algorithms (T-Learner and S-Learner with
    XGBoost) to estimate the **Conditional Average Treatment Effect (CATE)** —
    a model-based estimate of how each patient's outcome might differ between
    the two strategies, given their baseline characteristics.

    These are **model estimates, not clinical facts**. They represent patterns
    the algorithm detects in this small dataset and serve as hypotheses for
    investigation with larger cohorts.
    """)

    st.header("Data Source")
    X = results["X"]
    A = results["A"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Patients in Cohort", len(X),
                help="First ICU stay per patient with LOS >= 2 days")
    col2.metric("Clinical Features", X.shape[1],
                help="Curated non-redundant features (demographics, labs, vitals, scores)")
    col3.metric("Liberal Group", int((A == 1).sum()),
                help="Patients above median 48h fluid balance")
    col4.metric("Restrictive Group", int((A == 0).sum()),
                help="Patients below median 48h fluid balance")

    st.markdown(f"""
    - **Source**: MIMIC-IV Clinical Database Demo v2.2 (PhysioNet, MIT License)
    - **Cohort**: First ICU stay per patient, LOS >= 2 days (immortal time bias control)
    - **Treatment definition**: 48-hour net fluid balance, split at median
      (above median = "liberal", below = "restrictive")
    - **Outcome**: log(ICU Length of Stay in days)
    - **Confounder window**: 0-6 hours post-admission (pre-treatment)
    """)

    st.info(
        "**Why so few patients?** The MIMIC-IV Demo contains 100 patients total. "
        "After filtering for first ICU stay and LOS >= 2 days (to avoid immortal "
        f"time bias), {len(X)} patients remain. This is deliberately small — "
        "the demo exists to prove the pipeline works, not to draw clinical conclusions."
    )

    with st.expander("Methodology Details"):
        st.markdown("""
        **Causal Framework**:
        - Explicit causal DAG with conditional ignorability assumption
        - Confounders measured in pre-treatment window (0-6h)
        - Mediators (e.g., total urine output, vasopressor dose) excluded
        - Primary analysis restricted to LOS >= 2 days to avoid immortal time bias

        **Models**:
        - **T-Learner**: Two separate XGBoost models (one per treatment arm).
          Each arm has only ~27 patients, so individual arm predictions have
          wide uncertainty.
        - **S-Learner**: Single XGBoost with treatment as a feature.
          More stable but may underestimate heterogeneity.
        - Evaluated via Leave-One-Out Cross-Validation (LOOCV)
        - Model selection via AUTOC (see metric guide below)

        **Interpretability** (per Guo et al. 2025):
        - SHAP waterfall plots for individual outcome predictions
        - CATE explained via surrogate model SHAP (not naive differencing)
        - K=5 nearest neighbor clinical comparators (cosine similarity)
        - Depth-2 policy tree for interpretable treatment rules

        **Key References**:
        - Zhang et al. (2012) — Meta-learners for treatment effect estimation
        - Guo et al. (2025) — SHAP for CATE via surrogate models
        - Yadlowsky et al. (2021) — AUTOC evaluation metric
        """)

    with st.expander("How to Interpret the Metrics"):
        primary_result = results["loocv_s"] if results["primary_name"] == "S-Learner" \
            else results["loocv_t"]
        cate = primary_result.cate_estimates

        st.markdown(f"""
        ### Model Performance Metrics

        **R-squared (Coefficient of Determination)**
        - *What it measures*: What fraction of outcome variation the model explains
        - *Range*: 0 (no signal) to 1 (perfect prediction); can be negative if
          the model is worse than predicting the mean
        - *Context*: Individual ICU LOS depends on hundreds of unmeasured factors.
          R-squared of 0.1-0.3 is typical for ICU outcome prediction with limited
          features. This does NOT mean the model is useless — it means LOS is
          inherently hard to predict for individuals.

        **RMSE (Root Mean Squared Error)**
        - *What it measures*: Average prediction error in log-LOS units
        - *Interpretation*: RMSE = 0.5 means the model's typical prediction is
          off by a factor of exp(0.5) = 1.65x. If true LOS is 5 days, the model
          might predict anywhere from ~3 to ~8 days.
        - *Why this matters*: Large RMSE means individual predictions are noisy.
          Population-level patterns may still be reliable even when individual
          predictions are imprecise.

        **AUTOC (Area Under the Targeting Operator Characteristic)**
        - *What it measures*: How well the model *ranks* patients by estimated
          treatment effect — NOT overall prediction accuracy
        - *Interpretation*: Higher AUTOC means the model more consistently
          identifies which patients show larger vs. smaller estimated effects.
          Used to select between T-Learner and S-Learner.

        ### Treatment Effect Metrics

        **CATE (Conditional Average Treatment Effect)**
        - *What it measures*: Model's estimate of how much a patient's log-LOS
          would differ between liberal and restrictive fluid management
        - *Scale*: Log-LOS units. CATE = -0.2 means the model estimates ~18%
          shorter LOS under liberal fluids. CATE = +0.3 means ~35% shorter
          under restrictive.
        - *Current range*: [{cate.min():.3f}, {cate.max():.3f}] — this reflects
          the model's detected heterogeneity across {len(cate)} patients
        - *Confidence*: These are point estimates with NO confidence intervals.
          With ~27 patients per arm, true uncertainty is substantial.

        **SHAP Values**
        - *What they measure*: Each feature's contribution to a specific prediction
        - *Interpretation*: Positive SHAP = pushes prediction higher (longer LOS);
          negative = pushes lower (shorter LOS)
        - *Limitation*: SHAP explains *model behavior*, not necessarily *causal
          mechanisms*. A feature having high SHAP importance does not mean
          intervening on that feature would change outcomes.

        ### Important Caveats
        - All metrics come from a **{len(cate)}-patient retrospective cohort**
        - No external validation, no prospective testing
        - Treatment assignment is observational, not randomized
        - Results should be interpreted as **hypothesis-generating**, not
          **hypothesis-confirming**
        """)

    with st.expander("Feature List"):
        feature_groups = {
            "Demographics": [c for c in X.columns if c in
                           ["age", "gender_male", "race_white",
                            "insurance_medicare", "admission_type_emergency"]],
            "Diagnoses": [c for c in X.columns if c.startswith("has_") or
                         c in ["icu_unit_surgical"]],
            "Labs (first 6h)": [c for c in X.columns if "_first_6h" in c and
                               "missing" not in c and "hr_" not in c and
                               "rr_" not in c and "spo2" not in c and
                               "map_" not in c],
            "Vitals (first 6h)": [c for c in X.columns if c in
                                 [f"{v}_first_6h" for v in
                                  ["hr", "rr", "spo2", "map"]]],
            "Severity Scores": [c for c in X.columns if c in
                               ["sofa_total", "gcs_total"]],
            "Interventions": [c for c in X.columns if c in
                            ["vasopressor_at_admission", "mechanical_vent_at_admission"]],
            "Missingness Indicators": [c for c in X.columns if c.endswith("_missing")],
        }
        for group, cols in feature_groups.items():
            if cols:
                st.markdown(f"**{group}** ({len(cols)}): {', '.join(cols)}")


# ---------------------------------------------------------------------------
# Page: Patient Explorer
# ---------------------------------------------------------------------------

def page_patient_explorer(results: dict) -> None:
    """Individual patient exploration with SHAP and KNN comparators."""
    st.title("Patient Explorer")

    st.caption(
        "Explore model estimates for individual patients. These are algorithmic "
        "predictions, not clinical recommendations. See the About page for "
        "metric definitions."
    )

    X = results["X"]
    A = results["A"]
    Y = results["Y"]
    info = results["patient_info"]
    primary_result = results["loocv_s"] if results["primary_name"] == "S-Learner" \
        else results["loocv_t"]
    cate = primary_result.cate_estimates
    knn = results["knn"]
    shap_obs = results["shap_observed"]
    shap_cate = results["shap_cate"]
    reps = results["representative_patients"]

    # Patient selector
    stay_ids = X.index.values
    labels = []
    for sid in stay_ids:
        p = info.loc[sid]
        label = f"Stay {sid} — {p['gender']}, {p['age']:.0f}y, {p['treatment_label']}, LOS={p['los']:.1f}d"
        labels.append(label)

    # Quick-select representative patients
    st.sidebar.markdown("### Quick Select")
    if st.sidebar.button("Largest estimated liberal effect"):
        default_idx = int(np.where(stay_ids == reps["most_benefit_liberal"])[0][0])
    elif st.sidebar.button("Largest estimated restrictive effect"):
        default_idx = int(np.where(stay_ids == reps["most_benefit_restrictive"])[0][0])
    elif st.sidebar.button("Near-median estimated effect"):
        default_idx = int(np.where(stay_ids == reps["median_effect"])[0][0])
    else:
        default_idx = 0

    selected = st.selectbox("Select a patient:", labels, index=default_idx)
    idx = labels.index(selected)
    sid = stay_ids[idx]

    # Patient summary card
    p = info.loc[sid]
    patient_cate = cate[idx]

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Age", f"{p['age']:.0f}")
    col2.metric("Gender", p["gender"])
    col3.metric("Actual Treatment", p["treatment_label"],
                help="What this patient actually received (based on 48h fluid balance)")
    col4.metric("ICU LOS", f"{p['los']:.1f} days",
                help="Actual observed ICU length of stay")

    col5, col6, _ = st.columns(3)
    col5.metric("Fluid Balance", f"{p['fluid_balance_ml']:.0f} mL",
                help="Net 48-hour fluid balance (inputs - outputs, mL only)")
    col6.metric("ICU Unit", p["first_careunit"][:25])

    # CATE interpretation — neutral framing
    st.markdown("### Model Estimate: Treatment Effect")

    cate_magnitude = abs(patient_cate)
    los_ratio = np.exp(cate_magnitude)
    direction = "liberal" if patient_cate < 0 else "restrictive"
    approx_hours = abs(_cate_to_hours(patient_cate, baseline_los_days=p["los"]))

    st.info(
        f"**CATE = {patient_cate:.4f}** (log-LOS scale)\n\n"
        f"The model estimates this patient's LOS would be approximately "
        f"**{los_ratio:.2f}x** {'shorter' if patient_cate < 0 else 'longer'} "
        f"under liberal vs. restrictive fluids "
        f"(~{approx_hours:.1f} hours difference at this patient's LOS).\n\n"
        f"Model-suggested direction: **{direction}** fluid management.\n\n"
        f"*This estimate is based on {len(X)} patients (~{int((A==1).sum())} per arm) "
        f"and has not been externally validated. Treat as a hypothesis, not a recommendation.*"
    )

    # Counterfactual comparison
    st.markdown("### Counterfactual Comparison (Model-Based)")
    mu0_days = np.exp(primary_result.mu0[idx])
    mu1_days = np.exp(primary_result.mu1[idx])

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Predicted LOS (Restrictive)", f"{mu0_days:.1f} days",
               help="Model's prediction if this patient received restrictive fluids")
    cc2.metric("Predicted LOS (Liberal)", f"{mu1_days:.1f} days",
               help="Model's prediction if this patient received liberal fluids")
    cc3.metric("Actual LOS", f"{p['los']:.1f} days",
               help="What actually happened")

    r2 = primary_result.metrics["r2"]
    st.caption(
        f"These predictions come from the {results['primary_name']} model "
        f"(LOOCV R-squared = {r2:.3f}). Individual predictions have substantial "
        f"uncertainty — compare with actual LOS to gauge reliability."
    )

    # SHAP waterfall: observed outcome
    st.markdown("### Feature Contributions: Observed Outcome Prediction")
    st.caption(
        "SHAP values show how each feature pushed this patient's *predicted* "
        "LOS up (red/right) or down (blue/left) relative to the average. "
        "This explains model behavior, not necessarily causal mechanisms."
    )
    fig_obs, ax_obs = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_obs[idx], max_display=15, show=False)
    st.pyplot(plt.gcf())
    plt.close("all")

    # SHAP waterfall: CATE
    st.markdown("### Feature Contributions: Estimated Treatment Effect")
    st.caption(
        "Which features drive this patient's estimated CATE? Computed via a "
        "surrogate model (Guo et al. 2025) — a shallow model trained to "
        f"approximate the CATE estimates (surrogate R-squared = {results['surrogate_r2']:.3f})."
    )
    fig_cate, ax_cate = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_cate[idx], max_display=15, show=False)
    st.pyplot(plt.gcf())
    plt.close("all")

    # KNN comparators
    st.markdown("### Similar Patients (K=5 Nearest Neighbors)")
    st.caption(
        "Patients with the most similar baseline characteristics (cosine "
        "similarity in standardized feature space). Compare their treatments "
        "and outcomes for clinical context."
    )
    if sid in knn:
        knn_df = knn[sid].copy()
        knn_df = knn_df.rename(columns={
            "stay_id": "Stay ID",
            "distance": "Distance",
            "treatment": "Treatment",
            "los_days": "LOS (days)",
            "cate": "CATE",
            "age": "Age",
            "gender": "Gender",
        })
        knn_df["CATE"] = knn_df["CATE"].map("{:.4f}".format)
        knn_df["LOS (days)"] = knn_df["LOS (days)"].map("{:.1f}".format)
        knn_df["Distance"] = knn_df["Distance"].map("{:.3f}".format)
        st.dataframe(knn_df[["Stay ID", "Age", "Gender", "Treatment",
                            "LOS (days)", "CATE", "Distance"]],
                     use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: Population Insights
# ---------------------------------------------------------------------------

def page_population_insights(results: dict) -> None:
    """Population-level results: CATE distribution, subgroups, policy."""
    st.title("Population Insights")

    st.caption(
        "Population-level patterns detected by the model. All results are "
        "estimates from a small retrospective cohort and should be treated "
        "as exploratory findings."
    )

    X = results["X"]
    A = results["A"]
    primary_result = results["loocv_s"] if results["primary_name"] == "S-Learner" \
        else results["loocv_t"]
    cate = primary_result.cate_estimates

    # Model performance
    st.header("Model Performance (LOOCV)")
    st.markdown(
        "Leave-One-Out Cross-Validation: each patient is predicted by a model "
        "trained on all other patients. This prevents data leakage but means "
        f"each model is trained on only **{len(X)-1} patients**."
    )

    result_t = results["loocv_t"]
    result_s = results["loocv_s"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("T-Learner")
        st.metric("R-squared", f"{result_t.metrics['r2']:.4f}",
                  help="Fraction of outcome variance explained. Negative means "
                       "worse than predicting the mean — common when each arm "
                       f"has only ~{int((A==0).sum())} patients.")
        st.metric("RMSE", f"{result_t.metrics['rmse']:.4f}",
                  help="Average prediction error in log-LOS units. "
                       f"Translates to ~{np.exp(result_t.metrics['rmse']):.1f}x "
                       "multiplicative error in LOS.")
        st.metric("AUTOC", f"{results['autoc_t']:.4f}",
                  help="Area Under TOC — measures quality of patient ranking "
                       "by estimated treatment effect, not overall accuracy.")
    with col2:
        st.subheader("S-Learner")
        st.metric("R-squared", f"{result_s.metrics['r2']:.4f}",
                  help="Fraction of outcome variance explained. S-Learner "
                       "pools both arms, so it has more training data per model.")
        st.metric("RMSE", f"{result_s.metrics['rmse']:.4f}",
                  help="Average prediction error in log-LOS units. "
                       f"Translates to ~{np.exp(result_s.metrics['rmse']):.1f}x "
                       "multiplicative error in LOS.")
        st.metric("AUTOC", f"{results['autoc_s']:.4f}",
                  help="Area Under TOC — measures quality of patient ranking "
                       "by estimated treatment effect.")

    st.info(
        f"**Selected model**: {results['primary_name']} (higher AUTOC, meaning "
        f"it ranks patients by estimated treatment effect more consistently). "
        f"Note: AUTOC measures *ranking quality*, not prediction accuracy."
    )

    # CATE distribution
    st.markdown("---")
    st.header("Distribution of Estimated Treatment Effects")

    cate_hours = [_cate_to_hours(c) for c in cate]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=cate, nbinsx=20, name="CATE",
        marker_color="#4C72B0", opacity=0.8,
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="red",
                  annotation_text="No difference")
    fig.add_vline(x=np.median(cate), line_dash="solid", line_color="orange",
                  annotation_text=f"Median = {np.median(cate):.3f}")
    fig.update_layout(
        xaxis_title="CATE (log-LOS scale: liberal minus restrictive)",
        yaxis_title="Number of patients",
        title="How does estimated treatment effect vary across patients?",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Context on effect sizes
    median_los = np.exp(results["Y"].median())
    st.markdown(
        f"**Reading this chart:**\n"
        f"- CATE < 0 (left of red line): model estimates shorter LOS under liberal fluids\n"
        f"- CATE > 0 (right of red line): model estimates shorter LOS under restrictive fluids\n"
        f"- Median CATE = {np.median(cate):.3f} in log-LOS = roughly "
        f"{abs(_cate_to_hours(np.median(cate), median_los)):.0f} hours difference "
        f"at the median patient's LOS ({median_los:.1f} days)\n"
        f"- CATE range: [{cate.min():.3f}, {cate.max():.3f}] — "
        f"the model detects {'substantial' if np.std(cate) > 0.1 else 'limited'} "
        f"heterogeneity (std = {np.std(cate):.3f})\n\n"
        f"These are model estimates from {len(X)} patients. Individual CATEs "
        f"have no confidence intervals and should not be over-interpreted."
    )

    # Subgroup analysis
    st.markdown("---")
    st.header("Subgroup Analysis")
    st.markdown(
        "Does the model estimate different effects for clinically defined "
        "subgroups? A significant p-value means the mean estimated CATE "
        "differs from zero in that subgroup. **Significance here reflects "
        "model behavior, not necessarily a true clinical effect** — with small "
        "subgroups, even minor systematic patterns can reach significance."
    )

    subgroups = results["subgroups"]
    st.dataframe(
        subgroups.style.format({
            "mean_cate": "{:.4f}",
            "std_cate": "{:.4f}",
            "pct_liberal_optimal": "{:.1%}",
            "p_value": "{:.2e}",
        }).map(
            lambda v: "background-color: #d4edda" if v else "",
            subset=["significant"],
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Bonferroni-corrected alpha = 0.05/4 = 0.0125. Green highlighting "
        "indicates the model estimates a consistent directional effect for "
        "that subgroup. This is a pattern in the model, not proof of a "
        "clinical effect."
    )

    # Figures from disk
    st.markdown("---")
    st.header("LOOCV Calibration")
    st.caption(
        "Observed vs. predicted log-LOS. Points near the diagonal mean "
        "good calibration. Scatter around it reflects prediction uncertainty."
    )
    cal_path = FIGURES_DIR / "loocv_calibration.png"
    if cal_path.exists():
        st.image(str(cal_path), use_container_width=True)

    st.header("Subgroup Forest Plot")
    st.caption(
        "Mean estimated CATE per subgroup with 95% confidence intervals. "
        "Intervals that don't cross zero (red line) match the 'significant' "
        "column in the table above."
    )
    forest_path = FIGURES_DIR / "subgroup_forest.png"
    if forest_path.exists():
        st.image(str(forest_path), use_container_width=True)

    # Policy tree
    st.markdown("---")
    st.header("Interpretable Decision Rules")
    st.markdown(
        f"A depth-2 decision tree trained to approximate the model's treatment "
        f"recommendations. LOOCV accuracy: **{results['policy_tree_accuracy']:.1%}** — "
        f"this measures how consistently the simple tree matches the full model's "
        f"suggestions, not clinical accuracy."
    )
    tree_path = FIGURES_DIR / "policy_tree.png"
    if tree_path.exists():
        st.image(str(tree_path), use_container_width=True)
    st.caption(
        "This tree summarizes patterns in the model's CATE estimates. It shows "
        "which patient characteristics the model uses to differentiate treatment "
        "suggestions — useful for hypothesis generation."
    )

    # Value of personalization
    st.markdown("---")
    st.header("Estimated Policy Comparison")
    st.markdown(
        "**If the model's estimates were perfectly accurate**, what would "
        "mean ICU LOS look like under different treatment policies? This is "
        "a thought experiment, not a validated prediction."
    )

    pv = results["personalization_value"]

    policies = ["Observed\n(actual)", "Always\nLiberal", "Always\nRestrictive",
                "Model-Guided\nPersonalized"]
    values = [
        pv["observed_mean_los"],
        pv["always_liberal_mean_los"],
        pv["always_restrictive_mean_los"],
        pv["personalized_mean_los"],
    ]
    colors = ["#7f7f7f", "#DD8452", "#4C72B0", "#55A868"]

    fig = go.Figure(data=[
        go.Bar(x=policies, y=values, marker_color=colors,
               text=[f"{v:.2f}d" for v in values], textposition="outside")
    ])
    fig.update_layout(
        yaxis_title="Mean Predicted ICU LOS (days)",
        title="Estimated ICU LOS Under Different Policies (Model-Based)",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    r2 = primary_result.metrics["r2"]
    st.markdown(
        f"- The model estimates **{pv['pct_would_switch']:.0%}** of patients "
        f"received a treatment different from what the model would suggest\n"
        f"- Model-guided split: **{pv['n_liberal_optimal']}** patients toward "
        f"liberal, **{pv['n_restrictive_optimal']}** toward restrictive\n"
        f"- **Caveat**: This assumes perfect model accuracy. The model explains "
        f"only R-squared = {r2:.3f} of outcome variance in LOOCV, so the actual "
        f"achievable benefit is likely smaller than shown."
    )

    # Permutation test
    if "permutation_test" in results:
        st.markdown("---")
        st.header("Permutation Test: Is Treatment Effect Heterogeneity Real?")
        st.markdown(
            "Tests whether the variation in CATE estimates exceeds what you'd "
            "expect by chance. Treatment labels are randomly shuffled 1000 times "
            "and the model is re-run each time. If the observed CATE variation "
            "is larger than most shuffled versions, it suggests the model is "
            "detecting something beyond noise."
        )

        perm = results["permutation_test"]
        p1, p2 = st.columns(2)
        p1.metric("p-value", f"{perm['p_value']:.4f}",
                  help="Fraction of permutations with Var(CATE) >= observed. "
                       "Low p-value suggests genuine heterogeneity.")
        p2.metric("Observed Var(CATE)", f"{perm['observed_stat']:.6f}",
                  help="Variance of CATE estimates from the real data.")

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=perm["null_distribution"], nbinsx=50,
            name="Null distribution (shuffled)", marker_color="#CCCCCC",
        ))
        fig.add_vline(x=perm["observed_stat"], line_color="red", line_dash="solid",
                      annotation_text=f"Observed = {perm['observed_stat']:.4f}")
        fig.update_layout(
            xaxis_title="Var(CATE) under shuffled treatment labels",
            yaxis_title="Count",
            title="Is the detected heterogeneity greater than chance?",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Partner With Us
# ---------------------------------------------------------------------------

def page_partner(results: dict) -> None:
    """Partnership recruitment page."""
    st.title("Partner With Us")

    st.markdown("""
    ### From Proof-of-Concept to Clinical Impact

    This framework demonstrates that **exploring personalized fluid management**
    is technically feasible using routinely collected ICU data. But with only
    ~50 patients meeting inclusion criteria, these results are hypothesis-generating.

    **We need partners to test these hypotheses at scale.**
    """)

    st.markdown("---")

    # Privacy-first callout
    st.header("Privacy-First Design")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Local Execution")
        st.markdown("""
        - All computation runs on **your infrastructure**
        - No patient data leaves your institution
        - No internet connection required
        - Full source code available for audit
        """)
    with col2:
        st.markdown("### Federated Learning")
        st.markdown("""
        **How it works:**

        1. Each hospital trains the model **locally** on their own data
        2. Only **model weights** (not patient data) are shared
        3. A central server aggregates weights into a **global model**
        4. The updated global model is sent back to each site
        5. Repeat until convergence

        *Result*: A model trained on data from multiple hospitals, without
        any patient data ever crossing institutional boundaries.
        """)

    st.markdown("---")

    # Roadmap
    st.header("Development Roadmap")

    phases = [
        ("Phase 1", "Proof-of-Concept", "MIMIC-IV Demo (100 patients)", "Complete"),
        ("Phase 2", "Full Validation", "MIMIC-IV (~50,000 ICU stays)", "Next"),
        ("Phase 3", "Multi-Site Federation", "2-3 partner hospitals", "Seeking partners"),
        ("Phase 4", "Grant Submission", "NIH R01/R21 application", "Planning"),
        ("Phase 5", "Prospective Pilot", "Clinical decision support integration", "Future"),
    ]

    for phase, title, desc, status in phases:
        if status == "Complete":
            st.success(f"**{phase}: {title}** — {desc} ({status})")
        elif status == "Next":
            st.info(f"**{phase}: {title}** — {desc} ({status})")
        elif status == "Seeking partners":
            st.warning(f"**{phase}: {title}** — {desc} ({status})")
        else:
            st.markdown(f"**{phase}: {title}** — {desc} ({status})")

    st.markdown("---")

    # Partnership brief download
    st.header("Download Partnership Brief")
    brief_path = ASSETS_DIR / "partnership_brief.md"
    if brief_path.exists():
        brief_content = brief_path.read_text()
        st.download_button(
            label="Download Partnership Brief (Markdown)",
            data=brief_content,
            file_name="MIMIC_IV_Prescriptive_Partnership_Brief.md",
            mime="text/markdown",
        )
    st.caption(
        "Share this brief with your research team, IRB, or department head."
    )

    st.markdown("---")

    # Ideal partner
    st.header("Ideal Partner Profile")
    st.markdown("""
    - ICU research group with access to structured EHR data
    - Interest in **precision medicine** and **causal ML** for critical care
    - IRB infrastructure for retrospective EHR studies
    - Willingness to collaborate on a **federated analysis protocol**
    - Ambitious and seeking **NIH/AHRQ funding** partnerships
    """)

    st.markdown("---")
    st.markdown("""
    ### Contact

    **James Young, PhD**
    *Complexity science bridging biology, ML, and strategic foresight*

    View the full source code and methodology on GitHub.
    """)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    """Streamlit app entry point."""
    st.set_page_config(
        page_title="Prescriptive ICU",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("Prescriptive ICU")
    st.sidebar.markdown("*Exploring Personalized Fluid Management*")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        ["About", "Patient Explorer", "Population Insights", "Partner With Us"],
    )

    results = load_results()

    if page == "About":
        page_about(results)
    elif page == "Patient Explorer":
        page_patient_explorer(results)
    elif page == "Population Insights":
        page_population_insights(results)
    elif page == "Partner With Us":
        page_partner(results)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Built on MIMIC-IV Clinical Database Demo v2.2\n\n"
        "Research proof-of-concept only.\n"
        "Not for clinical use."
    )


if __name__ == "__main__":
    main()
