# REVIEW: MIMIC-IV Prescriptive ICU -- Counterfactual Fluid Management Optimization

**Reviewer**: Research Review Agent (Claude Opus 4.6)
**Date**: 2026-02-23
**Proposal reviewed**: `C:/Users/Admin/research/_incubator/MIMIC-IV-prescriptive/PROPOSAL.md`

---

## Overall Recommendation: **APPROVE WITH CONDITIONS**

This is a well-structured, honest proposal that correctly frames itself as a methods demonstration rather than a clinical contribution. The hypothesis is falsifiable, the risk assessment is unusually candid, and the interpretability stack (SHAP waterfalls + KNN comparators + policy trees) is a genuine value-add. However, there are several methodological issues that, if left unaddressed, would undermine the credibility of even a "methods demo" paper. The conditions below are organized as **blocking** (must fix before build) and **non-blocking** (fix during implementation or write-up).

---

## 1. Scientific Rigor

### 1.1 Hypothesis and Falsifiability -- GOOD

The hypothesis (Section "Hypothesis") is clearly stated and explicitly falsifiable via two mechanisms: (a) CATE indistinguishable from zero, and (b) >90% of patients assigned the same action. This meets the lab standard in `CLAUDE.md`. The 90% threshold is reasonable, though a formal power calculation for the permutation test would strengthen the claim (see Section 3.3 below).

### 1.2 Causal Identification Strategy -- BLOCKING CONCERN

The proposal relies on **conditional ignorability** (no unmeasured confounders given X) but never explicitly names this assumption. This is the single most important assumption in the entire project and it must be stated formally.

**What is missing**: A directed acyclic graph (DAG). Without a DAG, there is no principled way to determine which variables are confounders (should be adjusted for), which are mediators (should NOT be adjusted for), and which are colliders (adjusting would introduce bias). The proposal lists covariates in the "Feature Engineering Plan" section, but the rationale for inclusion is "they are available," not "they satisfy the backdoor criterion."

**Specific problem -- mediator adjustment**: The proposal includes `total_urine_output_24h` as a covariate. But urine output in the first 24 hours is a direct consequence of fluid administration during that same window (more fluids --> more urine output). If fluid balance is measured over 0-48h and urine output over 0-24h, then early urine output is a mediator on the causal path from early fluid decisions to the 48h cumulative balance. Adjusting for a mediator blocks part of the causal effect and biases the CATE estimate.

Similarly, `vasopressor_dose_norepi_equiv_24h` could be a time-varying confounder affected by prior treatment -- clinicians titrate vasopressors in response to the fluid resuscitation they have already started. Adjusting for post-treatment variables is a textbook source of bias in causal inference (Hernan & Robins, *Causal Inference: What If*, Chapter 9).

**Required fix**:
1. Draw a DAG explicitly showing the assumed causal structure among demographics, severity at admission, early interventions, fluid balance (treatment), and LOS (outcome).
2. Use the DAG to justify which variables satisfy the backdoor criterion.
3. Either remove mediators (urine output, vasopressor dose) from the confounder set, or move to a time-zero confounder set that is measured strictly before any treatment decisions are made.

### 1.3 Methods Appropriateness for n=100 -- ADEQUATE WITH CAVEATS

The proposal correctly identifies that n=100 is at the lower boundary of feasibility for meta-learners. The choice of T-learner with aggressive regularization (max_depth=3, min_child_weight=10) is defensible. The exclusion of causal forests, X-learner, and DR-learner at this sample size is well-reasoned and correctly argued in the "Why NOT" table.

**However**: The proposal claims ~55 features with missingness indicators. With ~50 patients per arm and 55 features, the effective degrees of freedom ratio is approximately 1:1, which is dangerously low even with regularized tree models. XGBoost will fit *something*, but the signal-to-noise ratio will be poor.

**Suggestion (non-blocking)**: Reduce the feature set to ~15-20 clinically motivated variables before entering the meta-learner. Use the DAG (once drawn) to select the minimal sufficient adjustment set. Report results with both the full (~55) and reduced (~15-20) feature sets as a sensitivity analysis.

---

## 2. Data Feasibility

### 2.1 Table Inventory and Row Counts -- PARTIALLY VERIFIED

I was unable to run direct verification queries against the demo `.csv.gz` files in this review session. The proposal's claimed row counts (e.g., inputevents: 20,404 rows, 138 stays; outputevents: 9,362 rows, 137 stays) appear consistent with the known size of MIMIC-IV Clinical Database Demo v2.2 (100 patients, ~140 ICU stays). These should be independently verified during the build phase by running `wc -l` or pandas `.shape` on each table and comparing against the proposal's table in the "Key table inventory" section.

### 2.2 Fluid Balance Extraction -- BLOCKING CONCERN

The extraction logic in the "Action Variable" section naively sums ALL `inputevents.amount`:

```python
inputs_48h = inputevents[...].groupby('stay_id')['amount'].sum()
```

**This is incorrect.** The `inputevents` table in MIMIC-IV contains mixed units in the `amount` column. Crystalloid fluids are measured in mL, but vasopressor infusions are measured in mcg, mg, or units. Blindly summing the `amount` column across all itemids would add milliliters of normal saline to micrograms of norepinephrine, producing a nonsensical number.

**Required fix**: Filter `inputevents` to fluid-type itemids only (e.g., Normal Saline 225158, Lactated Ringers 225828, D5W 225823, Albumin 25% 220862, etc.) or use the `amountuom` column to restrict to rows where `amountuom == 'ml'`. Better yet, use the `ingredientevents` table (which the proposal already notes is available) with the water content ingredient to calculate true fluid volume, as the MIMIC documentation recommends.

This is a critical data engineering bug that would silently corrupt the treatment variable if not caught.

### 2.3 Output Events Completeness -- MINOR CONCERN

The proposal lists only 4 urine output itemids (226559, 226560, 226631, 226627). MIMIC-IV has additional output categories that contribute to fluid balance: chest tube drainage, nasogastric output, wound drainage, etc. If the goal is net fluid balance, ALL output categories should be summed, not just urine. Alternatively, the proposal should explicitly state that the measure is "net fluid balance excluding insensible losses and non-urine outputs" and discuss how this systematically biases the treatment variable (it will overestimate positive fluid balance for all patients).

### 2.4 Lab Coverage After Restricting to First ICU Stays -- NEEDS VERIFICATION

The proposal reports lab coverage per `hadm_id` (e.g., Lactate: 120/275 hadm). But the analytic cohort is first ICU stay per patient (n=100), not all hospital admissions (n=275). Coverage rates could be materially different -- and likely lower -- when restricted to the 100 first-ICU-stay hadm_ids. For instance, if lactate is available in 120/275 admissions overall, it might be available in only 40-50/100 first-stay admissions. This needs to be checked before committing to the feature set.

### 2.5 Time-Zero Definition and Feature Windows -- ADEQUATE

The choice to use 0-24h for confounders and 0-48h for the treatment variable is reasonable and partially addresses time-zero bias. However, the proposal acknowledges this overlap in the Limitations section (item 5) but underplays it. A cleaner design would measure confounders at ICU admission (0-6h window) and treatment over 6-48h. This would reduce the temporal overlap between confounder measurement and treatment, at the cost of fewer lab values in the confounder window. This should be explored as a sensitivity analysis.

---

## 3. Statistical Concerns

### 3.1 LOOCV for CATE Evaluation -- PROBLEMATIC

LOOCV is proposed for evaluating the *outcome model* (R-squared, MAE, RMSE of predicted vs. actual log-LOS). This is appropriate at n=100 -- it is nearly unbiased and maximizes training set size. However, the proposal conflates two distinct evaluation targets:

1. **Outcome model accuracy**: How well does the model predict log(LOS)? LOOCV is fine for this.
2. **CATE estimation quality**: How accurate are the individual treatment effect estimates? LOOCV R-squared on observed outcomes tells you almost nothing about this. You can have a perfect outcome model and terrible CATE estimates if both arms are predicted accurately but their *difference* is dominated by noise.

**The fundamental problem**: There is no ground-truth CATE in observational data. You cannot evaluate CATE accuracy with LOOCV (or any cross-validation) because you never observe the counterfactual outcome for any patient. The proposal's "Metrics reported" section lists outcome-model metrics but presents them as if they validate the CATE estimates.

**Required fix**: Be explicit in the write-up that LOOCV validates the *nuisance models* (outcome prediction), not the *causal estimates*. The permutation test (Section 3.3 below) is the only direct test of CATE heterogeneity. Consider adding the AUTOC (Area Under the TOC curve) or RATE metric from Yadlowsky et al. (2021) as a more principled evaluation of treatment effect heterogeneity ranking.

### 3.2 T-Learner Variance at n=50 Per Arm -- NON-BLOCKING CONCERN

The Kunzel et al. (2019) PNAS paper that the proposal cites actually warns that T-learners have the highest variance among meta-learners in small samples because they estimate two separate models. Simulation studies consistently show that at n=50 per arm, T-learner CATE estimates are often no better than predicting the ATE for everyone.

The proposal mitigates this with aggressive regularization, which is correct. But the S-learner comparison is essential here -- if the S-learner produces near-zero CATE heterogeneity (due to its known shrinkage bias) while the T-learner produces large heterogeneity, the honest interpretation is "we cannot distinguish signal from noise at this sample size," not "the T-learner found real heterogeneity."

**Suggestion**: Add an explicit comparison criterion: if the T-learner and S-learner CATE distributions disagree qualitatively (e.g., T-learner Var(CATE) is 5x larger than S-learner), flag this as evidence of T-learner overfitting rather than real heterogeneity.

### 3.3 Permutation Test Design -- PARTIALLY VALID

The permutation test permutes treatment labels and re-runs the full LOOCV pipeline. The test statistic is Var(CATE_LOOCV). This is a reasonable approach, but there are two issues.

**Issue 1: Computational cost underestimated.** Each permutation requires 100 LOOCV iterations (one per patient), and each LOOCV iteration trains two XGBoost models (T-learner). That is 1000 permutations x 100 LOOCV folds x 2 models = 200,000 XGBoost fits. At ~55 features and n~50, each fit is fast (~0.01s), so the total is ~30 minutes as claimed. This is correct.

**Issue 2: Nuisance parameter problem.** Recent work by Olivares (2021, *Journal of Econometrics*) and Chung (2025, *Journal of Applied Econometrics*) shows that permutation tests for treatment effect heterogeneity can have inflated Type I error when the ATE is non-zero and estimated from the data. The proposal's test permutes treatment labels, which destroys both the ATE and any heterogeneity. Under the null of no heterogeneity (but potentially non-zero ATE), the permuted datasets have ATE=0, which is a different null than "constant non-zero ATE." This can inflate Type I error.

**Suggestion (non-blocking)**: Consider a residual-based permutation test. First estimate the ATE under the null of constant treatment effect. Then compute residuals under this null model. Permute the residuals rather than the treatment labels. This preserves the ATE under the null and tests only for heterogeneity.

### 3.4 Multiple Comparisons in Subgroup Analysis -- MISSING

The "Policy Summary" section proposes stratifying optimal action by ICU type, sepsis, vasopressor use, and AKI. That is 4 subgroup tests. With n=100 and binary subgroups, some cells will have <15 patients. No multiple comparison correction is mentioned.

**Required fix**: Apply Bonferroni or Holm correction to subgroup p-values, or present subgroup analyses as purely exploratory/descriptive with no inferential claims.

---

## 4. Clinical Validity

### 4.1 Fluid Balance as Action Variable -- DEFENSIBLE BUT IMPERFECT

The proposal correctly identifies clinical equipoise on fluid management (post-CLASSIC/CLOVERS) and notes that fluid balance is a discretionary clinical decision. This is well-argued.

**Concern**: Cumulative fluid balance over 48 hours is a *composite summary* of many sequential clinical decisions (individual boluses, rate changes, diuretic administration). Treating it as a single binary action implicitly assumes that clinicians make one decision at ICU admission and stick with it for 48 hours. In reality, fluid management is a dynamic process with feedback loops (patient responds to fluids --> clinician adjusts --> patient changes). This dynamic treatment regime is better modeled with marginal structural models or dynamic treatment regimes (DTR), not a single-timepoint binary treatment.

The proposal acknowledges the median split limitation (Risk Assessment, row 5) but does not acknowledge the dynamic treatment problem. This should be added to the Limitations section.

### 4.2 Median Split Binarization -- ACCEPTABLE FOR DEMO

Median splitting a continuous treatment variable is a well-known source of information loss, but it is standard practice in the meta-learner literature for pedagogical demonstrations. The proposed sensitivity analysis with tercile splits and GAM dose-response is good. The key risk is that the median is estimated from the sample and is therefore a function of the data -- this creates a subtle form of data-dependent treatment definition. At n=100 this is minor, but should be noted.

### 4.3 Confounder Sufficiency -- BLOCKING CONCERN (see 1.2)

Beyond the mediator adjustment issue raised in Section 1.2, the confounder set is missing several variables that are routinely available in MIMIC-IV and are strong predictors of both fluid management and LOS:

- **SOFA score** (or its components): The proposal uses individual lab values as proxies but never computes a severity score. SOFA is the standard severity metric in ICU research and is constructable from the available data (labs + vitals + vasopressor dose + ventilation + GCS). Its omission would be questioned by any critical care reviewer.
- **GCS (Glasgow Coma Scale)**: Available in `chartevents` (itemid 220739 for GCS-Eye, 223900 for GCS-Verbal, 223901 for GCS-Motor). Not mentioned in the feature set.
- **Body weight**: Available in `chartevents` (itemid 224639 or 226512). Clinicians dose fluids by weight. Its absence as a confounder is a potential source of unmeasured confounding.
- **Time of day / day of week of admission**: Known to affect staffing and practice patterns in ICUs.

**Required fix**: Add SOFA score (or at minimum its components) and GCS to the confounder set. Consider body weight if available in the demo.

### 4.4 Outcome Choice -- GOOD

The choice of log(ICU LOS) over mortality is well-justified given 11 deaths in the cohort. The proposal correctly identifies that a binary outcome with 11 events and 55 covariates would be hopeless. Log-transformation of LOS is standard.

**Minor note**: The epsilon adjustment `log(los + 0.01)` handles near-zero stays, but `los` in `icustays` is in fractional days. A patient with LOS = 0.02 days (~29 minutes) likely represents a transfer or data artifact. Consider excluding stays with LOS < 0.05 days (roughly 1 hour) rather than epsilon-adjusting them, as these ultra-short stays have very different data profiles and could distort the model.

---

## 5. Interpretability Stack

### 5.1 SHAP Waterfalls -- GOOD BUT NEEDS CLARIFICATION

The per-patient SHAP waterfall for the outcome model under observed treatment is straightforward and well-specified. `shap.TreeExplainer` with XGBoost is a standard, fast approach.

### 5.2 SHAP-Based CATE Decomposition -- BLOCKING CONCERN

The proposal defines:

```
SHAP_CATE_feature_j = SHAP_mu1_j(x) - SHAP_mu0_j(x)
```

This is presented as if it decomposes the CATE into feature contributions. **This is not theoretically justified.** SHAP values are additive decompositions of a *single model's* prediction, satisfying the efficiency property (SHAP values sum to prediction minus expected value). Taking the difference of SHAP values from two separate models fitted on different subsets of the data does NOT produce valid SHAP values for the difference function. The background distributions (E[f(X)]) differ between the two models, the feature interactions differ, and the resulting "CATE SHAP" values do not satisfy any of the Shapley axioms for the CATE function.

A recent preprint (Guo et al., 2025, arXiv:2505.01145) specifically addresses this problem and recommends a surrogate model approach: fit a single model to predict the CATE estimates themselves, then compute SHAP on that surrogate.

**Required fix**: Either (a) fit a surrogate model `g(X) -> CATE_hat` and compute SHAP on `g`, or (b) use the S-learner (where CATE is a function of a single model and SHAP decomposition is principled via intervention on the treatment feature), or (c) keep the naive difference but clearly label it as a heuristic that does not satisfy Shapley axioms.

### 5.3 KNN Clinical Comparator -- GOOD

The KNN approach (k=5, cosine distance in standardized covariate space) is a practical and clinically intuitive audit layer. The display format (neighbor treatment, LOS, CATE, distance) is well-designed.

**Minor suggestion**: With ~55 features and n=100, cosine distance in the full feature space will suffer from the curse of dimensionality. Consider using the first 10-15 principal components instead, or using the propensity score as a 1-dimensional summary for neighbor matching.

### 5.4 Policy Tree -- ADEQUATE

A depth-2 decision tree on `X -> optimal_action` is a clean visualization. However, at n=100, the "optimal action" labels are themselves noisy estimates (they depend on the CATE estimates, which are uncertain). Fitting a tree to noisy labels amplifies the noise. The proposal should report the tree's cross-validated accuracy on these labels, and if it is near 50% (random chance for binary action), acknowledge that the tree is not capturing a real decision boundary.

---

## 6. Novelty Assessment

### 6.1 Competing Work -- SIGNIFICANT OVERLAP

The proposal claims: "No published study on MIMIC-IV has combined [meta-learner CATE, per-patient counterfactual simulation, SHAP interpretability, KNN audit layer] for fluid management."

This is technically true as a conjunction of all four elements, but the individual components have substantial precedent:

**Directly competing paper**: Zhang et al. (2024/2025), ["Personalized Fluid Management in Patients with Sepsis and AKI: A Causal Machine Learning Approach"](https://pmc.ncbi.nlm.nih.gov/articles/PMC12677861/), published in *Critical Care Explorations*. This paper uses MIMIC-IV, causal forest for heterogeneous treatment effects of restrictive fluids, policy trees for subgroup identification, and validates externally on SICdb. It uses a much larger cohort (thousands of patients), achieves AUTOC of 0.73 in development and 0.15 in external validation, and directly addresses fluid management personalization.

This paper substantially narrows the novelty claim. The proposal must:
1. Cite this paper explicitly.
2. Differentiate clearly: the Zhang et al. paper focuses on sepsis+AKI patients, uses causal forest (not meta-learners), and does not include per-patient SHAP waterfalls or KNN comparators. The proposal's contribution is the *interpretability stack*, not the causal question itself.
3. Acknowledge that at n=100, this project cannot compete with Zhang et al. on clinical validity, and the contribution is purely methodological/pedagogical.

### 6.2 Broader Literature Context

The proposal cites appropriate foundational references (Kunzel et al. 2019, Feuerriegel et al. 2024, Sanchez-Pinto et al. 2024). However, it should also cite:

- **Zhang et al. (2024/2025)** as noted above -- the most directly competing work.
- **Komorowski et al. (2018), Nature Medicine** -- "The Artificial Clinician learns optimal treatment strategies for sepsis in intensive care." Pioneered RL-based prescriptive analytics in MIMIC.
- **Yadlowsky et al. (2021)** -- "Estimation and validation of CATE" for the AUTOC/RATE evaluation metrics that would strengthen the evaluation plan.

---

## 7. Risks and Limitations

### 7.1 Stated Risks -- HONEST AND COMPREHENSIVE

The risk table is unusually good for an early proposal. The high-likelihood, high-impact risks (n=100 instability, confounding by indication) are correctly flagged. The mitigations are reasonable.

### 7.2 Missing Risks

The following risks are absent from the assessment:

| Risk | Likelihood | Impact | Why it matters |
|------|-----------|--------|----------------|
| **Treatment-confounder feedback** (dynamic confounding) | H | H | Fluid administration in hours 0-24 affects labs/vitals in hours 0-24, which are then used as confounders. Standard conditional ignorability does not handle this. |
| **Informative censoring** | M | M | Patients who die in the ICU have their LOS truncated. The 11 deaths in the cohort have ICU LOS determined by death, not recovery. Log(LOS) for these patients has a qualitatively different meaning. |
| **Selection bias in the demo cohort** | M | M | The 100 patients in the MIMIC-IV demo are not a random sample of all MIMIC-IV patients. They were selected by the PhysioNet team. If selection was non-random (e.g., interesting cases, complete data), the demo cohort may not represent the distribution of the full database, limiting the "path to scale" story. |
| **Unit-mixing in inputevents** | H | H | As discussed in Section 2.2 -- naively summing amounts across all itemids produces nonsensical fluid balance values. |
| **Immortal time bias** | L | M | If treatment is defined over 0-48h but some patients leave the ICU before 48h, those patients are guaranteed to have lower fluid balance AND shorter LOS, creating a spurious association between restrictive fluids and short LOS. |

**Required fix**: Add treatment-confounder feedback, informative censoring, and immortal time bias to the risk table. The immortal time bias for short-stay patients (LOS < 48h) is particularly insidious and must be addressed -- either by restricting the cohort to patients with LOS >= 48h (which will dramatically reduce n) or by prorating the fluid balance to the actual observation period.

---

## 8. Scope and Feasibility

### 8.1 Time Estimate -- OPTIMISTIC BUT PLAUSIBLE

The proposal estimates 4-6 agent-hours for the full pipeline. This is achievable if the feature engineering is straightforward (it will not be -- the inputevents unit-mixing problem alone could consume 1-2 hours of debugging). A more realistic estimate is 6-10 agent-hours including the data engineering fixes identified in this review.

### 8.2 Dependencies -- REASONABLE

All dependencies (pandas, numpy, scikit-learn, xgboost, shap, matplotlib, seaborn) are standard and well-maintained. No exotic packages.

### 8.3 Deliverables -- WELL-SCOPED

The deliverable list is comprehensive and realistic. The manuscript-ready methods section is a good inclusion for portfolio purposes.

---

## 9. Cross-Pollination Value

The cross-pollination story with CBRE and ECOCHEMICAL is genuinely compelling. The LOOCV + permutation test harness, SHAP waterfall generator, and KNN comparator retriever are all reusable components. If built cleanly (type hints, docstrings, modular design per `CLAUDE.md`), these tools will earn their keep across multiple projects.

---

## Summary of Required Changes (Blocking)

These must be addressed before proceeding to the build phase:

1. **Draw a DAG** specifying the causal structure. Use it to justify the confounder set. Remove mediators (urine output, vasopressor dose) or justify their inclusion causally. (Section 1.2)

2. **Fix the inputevents extraction logic** to filter by fluid-type itemids or restrict to `amountuom == 'ml'`. Do not sum mixed units. (Section 2.2)

3. **Add SOFA score components and GCS** to the confounder set. (Section 4.3)

4. **Fix the SHAP-CATE decomposition** to either use a surrogate model or clearly label the naive difference as a heuristic. (Section 5.2)

5. **Address immortal time bias** for patients with LOS < 48h. Either restrict to LOS >= 48h, prorate fluid balance, or use the full-stay fluid balance (and rename accordingly). (Section 7.2)

6. **Cite Zhang et al. (2024/2025)** and differentiate the contribution. (Section 6.1)

7. **Clarify that LOOCV validates nuisance models, not CATE estimates.** Add AUTOC or RATE as a CATE-specific evaluation metric. (Section 3.1)

## Summary of Suggestions (Non-Blocking)

These would strengthen the work but are not required before building:

- Reduce feature set to ~15-20 via DAG-based minimal sufficient adjustment set; report full set as sensitivity analysis. (Section 1.3)
- Add T-learner vs. S-learner CATE variance comparison as an overfitting diagnostic. (Section 3.2)
- Consider residual-based permutation test to avoid nuisance parameter inflation. (Section 3.3)
- Apply multiple comparison correction to subgroup analyses. (Section 3.4)
- Use PCA-reduced features for KNN neighbor retrieval. (Section 5.3)
- Exclude ultra-short stays (LOS < 1 hour) rather than epsilon-adjusting. (Section 4.4)
- Explore 0-6h confounder window as sensitivity analysis. (Section 2.5)
- Report policy tree cross-validated accuracy. (Section 5.4)
- Add informative censoring and selection bias to the risk table. (Section 7.2)
- Revise time estimate upward to 6-10 agent-hours. (Section 8.1)

---

## Scoring

| Dimension | Score (1-5) | Rationale |
|-----------|-------------|-----------|
| **Novelty** | 2.5 | Zhang et al. (2024/2025) substantially preempts the causal question. The interpretability stack adds value, but the core analysis is not new. |
| **Feasibility** | 4.0 | Buildable in a weekend. Data is free, dependencies are standard, scope is well-contained. Blocking issues are fixable. |
| **Impact** | 3.0 | As a methods demo and reusable toolkit, moderate impact across the portfolio. Not publishable as a standalone clinical paper at n=100. |
| **Fit with portfolio** | 4.5 | Excellent fit. Direct cross-pollination with CBRE (causal ML) and ECOCHEMICAL (prescriptive optimization). Builds reusable commons infrastructure. |

**Weighted average**: 3.5 / 5.0

---

## Final Verdict

**APPROVE WITH CONDITIONS.** The blocking issues enumerated above are all fixable within the proposed timeline. The project's value lies in the reusable causal ML infrastructure and the interpretability stack, not in clinical novelty. If the blocking conditions are met, this is a worthwhile weekend build that pays dividends across the portfolio. If they are not met, the project risks producing silently incorrect results (especially the inputevents unit-mixing bug and the mediator adjustment problem) that would not survive even casual peer scrutiny.

---

## References Identified During Review

- Zhang et al. (2024/2025). [Personalized Fluid Management in Patients with Sepsis and AKI: A Causal Machine Learning Approach](https://pmc.ncbi.nlm.nih.gov/articles/PMC12677861/). *Critical Care Explorations*.
- Kunzel et al. (2019). [Metalearners for estimating heterogeneous treatment effects](https://www.pnas.org/doi/10.1073/pnas.1804597116). *PNAS*.
- Guo et al. (2025). [Overview and practical recommendations on using Shapley Values for identifying predictive biomarkers via CATE modeling](https://arxiv.org/html/2505.01145v1). *arXiv preprint*.
- Olivares (2021). [Permutation test for heterogeneous treatment effects with a nuisance parameter](https://www.sciencedirect.com/science/article/abs/pii/S0304407621001561). *Journal of Econometrics*.
- Chung (2025). [Quantile-based test for heterogeneous treatment effects](https://onlinelibrary.wiley.com/doi/full/10.1002/jae.3093). *Journal of Applied Econometrics*.
- Komorowski et al. (2018). The Artificial Clinician learns optimal treatment strategies for sepsis in intensive care. *Nature Medicine*.
- Hernan & Robins. *Causal Inference: What If*. Chapman & Hall/CRC. (For mediator adjustment and time-varying confounding.)
- VanderWeele & Ding (2017). [Sensitivity Analysis in Observational Research: Introducing the E-Value](https://pubmed.ncbi.nlm.nih.gov/28693043/). *Annals of Internal Medicine*.
