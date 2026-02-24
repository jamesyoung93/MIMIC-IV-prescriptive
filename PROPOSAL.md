# MIMIC-IV Prescriptive ICU: Counterfactual Fluid Management Optimization

**Proposed**: 2026-02-23
**Status**: Under Review
**Domain**: Clinical ML / Causal Inference / Prescriptive Analytics

---

## Hypothesis

**Among ICU patients, individualized fluid management strategy (liberal vs. restrictive cumulative balance in the first 48 hours) causally affects ICU length of stay, and a meta-learner framework can identify patient subgroups for whom the optimal strategy differs from the observed clinical decision.**

This would be disproved if: (a) the estimated conditional average treatment effect (CATE) is indistinguishable from zero across all patient strata, or (b) the recommended policy assigns the same fluid strategy to >90% of patients regardless of covariates (no meaningful heterogeneity).

## Background & Gap

### What exists today

Fluid management is one of the most contested decisions in critical care. The CLASSIC trial (2022), CLOVERS (2023), and CONSERVE (2024) showed that restrictive vs. liberal fluid strategies yield mixed results *at the population level* — suggesting the answer is not "one size fits all" but rather patient-dependent. Multiple MIMIC-IV analyses have examined early enteral nutrition timing (Wang et al. 2024, *Frontiers in Nutrition*; Chen et al. 2025, *BMC Infectious Diseases*) and vasopressor initiation, but almost exclusively using propensity score matching or IPTW for *average* treatment effects — not *heterogeneous* treatment effects that would allow individualized prescription.

### Specific gap

Zhang et al. (2024/2025, *Critical Care Explorations*) recently applied causal forests and policy trees to personalized fluid management in MIMIC-IV for sepsis+AKI patients, achieving AUTOC of 0.73 in development. However, their work focuses on a specific subpopulation and does not provide:
1. **Per-patient counterfactual simulation with full interpretability** — SHAP waterfall decomposition of *why* a specific patient is predicted to benefit from one strategy over another
2. **KNN clinical audit layer** — showing real comparable patients who received each strategy, enabling clinician verification of model reasoning
3. **A reusable, locally-deployable pipeline** — designed for federated retraining on a partner institution's own data without centralizing sensitive records

Our contribution is the *interpretability and deployment stack* around the causal question, not the causal question itself.

### Why now

- The `ingredientevents`, `inputevents`, and `outputevents` tables in MIMIC-IV v2.2 provide granular, timestamped fluid I/O data absent in MIMIC-III
- Meta-learner implementations (EconML, CausalML) are now mature and SHAP-compatible
- Clinical equipoise on fluid management (post-CLASSIC/CLOVERS) means prescriptive analytics here address a genuine decision problem, not an already-settled question

### Key references

- **Zhang et al. (2024/2025)**. [Personalized Fluid Management in Sepsis and AKI: A Causal ML Approach](https://pmc.ncbi.nlm.nih.gov/articles/PMC12677861/). *Critical Care Explorations*. — Most directly related prior work.
- Sanchez-Pinto et al. (2024). [Causal inference using observational ICU data: scoping review](https://www.nature.com/articles/s41746-023-00961-1). *npj Digital Medicine*.
- Feuerriegel et al. (2024). [Causal machine learning for predicting treatment outcomes](https://pubmed.ncbi.nlm.nih.gov/38641741/). *Nature Medicine*.
- Kunzel et al. (2019). [Metalearners for estimating heterogeneous treatment effects](https://www.pnas.org/doi/10.1073/pnas.1804597116). *PNAS*.
- Komorowski et al. (2018). The Artificial Clinician learns optimal treatment strategies for sepsis in intensive care. *Nature Medicine*. — Pioneered RL-based prescriptive analytics in MIMIC.
- Yadlowsky et al. (2021). Estimation and validation of ratio-weighted average treatment effects. — AUTOC/RATE evaluation metrics for CATE.
- Bertsimas & Dunn (2019). [Optimal policy trees](https://dl.acm.org/doi/abs/10.1007/s10994-022-06128-5). *Machine Learning*.
- Guo et al. (2025). [SHAP values for CATE modeling: recommendations](https://arxiv.org/html/2505.01145v1). *arXiv preprint*. — Guides our surrogate-model SHAP approach.
- Hernan & Robins. *Causal Inference: What If*. Chapman & Hall/CRC. — Mediator adjustment, time-varying confounding.
- Wang et al. (2024). [Early vs. delayed EN in sepsis — MIMIC-IV](https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2024.1370472/full). *Frontiers in Nutrition*.

---

## Approach

### Causal Identification Strategy

#### Assumed Causal Structure (DAG)

```
                    ┌──────────────────────────────┐
                    │    Pre-Treatment Confounders  │
                    │  (measured at ICU admission)  │
                    │                               │
                    │  W = {age, gender, race,      │
                    │       admission type,          │
                    │       diagnoses (sepsis, AKI,  │
                    │       HF, resp failure),       │
                    │       SOFA components at 0-6h, │
                    │       GCS at admission,        │
                    │       first labs (0-6h),        │
                    │       first vitals (0-6h)}      │
                    └──────────┬───────┬────────────┘
                               │       │
                               ▼       ▼
                    ┌──────────┐   ┌──────────┐
                    │ Treatment│   │  Outcome  │
                    │ A: Fluid │──▶│ Y: log    │
                    │ balance  │   │ (ICU LOS) │
                    │ (0-48h)  │   │           │
                    └──────────┘   └──────────┘
                               ▲       ▲
                               │       │
                    ┌──────────┴───────┴────────────┐
                    │    Unmeasured Confounders (U)  │
                    │  clinician gestalt, family     │
                    │  preferences, pre-ICU trajectory│
                    └──────────────────────────────┘
```

#### Key assumption: Conditional Ignorability

`Y(a) ⊥ A | W` — treatment assignment is independent of potential outcomes, conditional on the measured pre-treatment confounders W.

This is **not testable** and is almost certainly violated to some degree. We proceed under this assumption while reporting E-value sensitivity analysis to quantify how strong an unmeasured confounder would need to be to explain away the estimated effects.

#### DAG-justified covariate selection rules

1. **Include**: All common causes of treatment (A) and outcome (Y) that are measured **before** treatment begins — these satisfy the backdoor criterion.
2. **Exclude mediators**: Variables on the causal path A → M → Y. Adjusting for mediators blocks part of the causal effect.
   - `total_urine_output_24h` — **REMOVED**. Urine output in 0–24h is a direct consequence of fluid administration. Including it would block a causal pathway (more fluids → more urine → masks the true fluid balance effect).
   - `vasopressor_dose_norepi_equiv_24h` — **REMOVED as continuous dose**. Vasopressor dose is titrated in response to ongoing fluid resuscitation, making it a time-varying confounder affected by prior treatment. We retain only `vasopressor_any_at_admission` (binary: was a vasopressor running at ICU admission?), which is a pre-treatment severity indicator.
3. **Exclude colliders**: Variables caused by both treatment and outcome (e.g., discharge disposition). None identified in our feature set.
4. **Confounder window**: All confounders measured in the **0–6 hour** window after ICU admission (before the bulk of the 48h treatment unfolds), with a sensitivity analysis using the full 0–24h window.

### Data

**Source**: MIMIC-IV Clinical Database Demo v2.2 (PhysioNet, open access)
- 100 patients, 140 ICU stays → **analytic cohort: first ICU stay per patient (n=100)**
- All structured tables; no clinical notes

**Key table inventory** (verified row counts from demo):

| Table | Rows | Stays | Role |
|-------|------|-------|------|
| `icu/icustays.csv.gz` | 140 | 140 | Cohort definition, LOS outcome |
| `hosp/admissions.csv.gz` | 275 | — | Mortality flag, demographics |
| `hosp/patients.csv.gz` | 100 | — | Age, gender, death date |
| `icu/inputevents.csv.gz` | 20,404 | 138 | Fluid inputs, vasopressors |
| `icu/outputevents.csv.gz` | 9,362 | 137 | Urine, drain outputs |
| `icu/ingredientevents.csv.gz` | 25,728 | 138 | Caloric delivery (itemid 226060) |
| `hosp/labevents.csv.gz` | 107,727 | — | Lab values |
| `icu/chartevents.csv.gz` | 668,862 | 140 | Vitals (HR, MAP, RR) |
| `hosp/diagnoses_icd.csv.gz` | 4,506 | — | ICD-10 diagnoses |
| `icu/procedureevents.csv.gz` | 1,468 | — | Ventilation status |

### Outcome Definition

**Primary outcome: log(ICU LOS in days)**

Rationale for LOS over mortality:
- Only **11 deaths** in the first-ICU-stay cohort (11%) — a binary outcome with 11 events and ~25 covariates would produce wildly unstable estimates
- ICU LOS is continuous (median 2.2d, mean 3.7d, range 0.02–20.5d), giving 100 real-valued observations — far more statistical power
- Log-transform normalizes the right-skewed distribution and makes effects interpretable as percent changes
- LOS is clinically meaningful and economically relevant (~$5,000/ICU day)

**Cohort restriction for immortal time bias**: Patients with ICU LOS < 48 hours are mechanically guaranteed to have lower cumulative fluid balance AND shorter LOS, creating a spurious association between restrictive fluids and short stays. To address this:
- **Primary analysis**: Restrict to patients with LOS >= 2 days (n ≈ 55–65, estimated from median LOS of 2.2d). Fluid balance is measured over their full first 48h.
- **Sensitivity analysis**: Include all 100 patients but **prorate** fluid balance to the actual observation period (mL/hour × hours observed) to remove the mechanical correlation.
- Patients with LOS < 1 hour (likely transfers or data artifacts) are excluded in all analyses.

**Extraction logic**:
```python
# Exclude ultra-short stays (< 1 hour = 0.042 days)
cohort = icustays[icustays.los >= 0.042]

# Primary cohort: LOS >= 48h to avoid immortal time bias
cohort_primary = cohort[cohort.los >= 2.0]

outcome = np.log(cohort_primary.los)  # no epsilon needed after exclusions
```

**Secondary outcome** (reported but not used for meta-learner): `admissions.hospital_expire_flag` as a descriptive sanity check. Note: among the 11 deaths in the first-ICU cohort, LOS is truncated by death, not recovery — this is **informative censoring** and is acknowledged as a limitation.

### Action Variable: 48-Hour Cumulative Fluid Balance

**Definition**: Net fluid balance (mL) = total IV inputs − total measured outputs during hours 0–48 from ICU admission.

**Binarization**:
- `A=1` (Liberal): cumulative balance ≥ cohort median
- `A=0` (Restrictive): cumulative balance < cohort median

**Why this action variable**:
1. **Coverage**: 137–138/140 ICU stays have both input and output data — near-complete
2. **Malleability**: Fluid management is a discretionary clinical decision, not a fixed patient characteristic
3. **Clinical equipoise**: Post-CLASSIC/CLOVERS, optimal fluid strategy is genuinely unresolved
4. **Variation**: ICU fluid balance exhibits high inter-patient variability

**Extraction logic**:
```python
# CRITICAL: Filter inputevents to fluid-volume rows only (amountuom == 'ml').
# The inputevents table mixes fluid volumes (mL) with drug doses (mcg, mg, units)
# in the same 'amount' column. Summing without filtering would add mL of saline
# to micrograms of norepinephrine — a silent data corruption bug.
inputs_48h = inputevents[
    (inputevents.starttime >= icu_intime) &
    (inputevents.starttime < icu_intime + 48h) &
    (inputevents.amountuom == 'ml')  # restrict to fluid volumes only
].groupby('stay_id')['amount'].sum()

# Outputs: sum ALL output categories (not just urine) for true fluid balance.
# Includes: Foley (226559), Void (226560), chest tube, NG drainage, wound drains, etc.
outputs_48h = outputevents[
    (outputevents.charttime >= icu_intime) &
    (outputevents.charttime < icu_intime + 48h)
].groupby('stay_id')['value'].sum()

fluid_balance_48h = inputs_48h - outputs_48h
treatment = (fluid_balance_48h >= fluid_balance_48h.median()).astype(int)
```

**Note**: This measures charted fluid balance only. Insensible losses (~500–1000 mL/day) and non-charted oral intake are not captured, resulting in a systematic overestimate of positive fluid balance for all patients. This is a known limitation of all MIMIC-IV fluid balance analyses.

**Secondary action variable** (exploratory): Early vasopressor initiation (norepinephrine started within 6h of ICU admission vs. later/never). Available in 39/100 first ICU stays with vasopressor data. This will be analyzed descriptively but not as the primary meta-learner treatment due to the confounding-by-indication problem (sicker patients get vasopressors earlier).

### Feature Engineering Plan

All confounders measured in the **0–6 hour** window after ICU admission (pre-treatment), per the DAG above. Sensitivity analysis with a 0–24h window is reported separately.

#### Demographics (from `patients`, `admissions`)
| Feature | Source | Logic |
|---------|--------|-------|
| `age` | `patients.anchor_age` | Direct |
| `gender` | `patients.gender` | Binary encode (M=1) |
| `race_white` | `admissions.race` | Binary: White vs. other |
| `insurance_medicare` | `admissions.insurance` | Binary: Medicare vs. other |
| `admission_type_emergency` | `admissions.admission_type` | Binary: emergency vs. other |

#### Diagnosis severity proxy (from `diagnoses_icd`)
| Feature | Source | Logic |
|---------|--------|-------|
| `n_diagnoses` | `diagnoses_icd` | Count of ICD codes per `hadm_id` |
| `has_sepsis` | `diagnoses_icd.icd_code` | ICD-10 A40*, A41*, R65.2* |
| `has_aki` | `diagnoses_icd.icd_code` | ICD-10 N17* |
| `has_heart_failure` | `diagnoses_icd.icd_code` | ICD-10 I50* |
| `has_respiratory_failure` | `diagnoses_icd.icd_code` | ICD-10 J96* |
| `icu_unit_surgical` | `icustays.first_careunit` | Binary: SICU/TSICU vs. medical |

#### Key labs (from `labevents`) — FIRST value in 0–6h window (pre-treatment)
| Lab | `itemid` | Coverage | Feature |
|-----|----------|----------|---------|
| Lactate | 50813 | 120/275 hadm | `lactate_first_6h` |
| Creatinine | 50912 | 250/275 | `creatinine_first_6h` |
| Albumin | 50862 | 129/275 | `albumin_first_6h` |
| BUN | 51006 | 250/275 | `bun_first_6h` |
| WBC | 51301 | 245/275 | `wbc_first_6h` |
| Glucose | 50931 | 246/275 | `glucose_first_6h` |
| Platelets | 51265 | 248/275 | `platelets_first_6h` |
| Hemoglobin | 51222 | 246/275 | `hemoglobin_first_6h` |
| Bicarbonate | 50882 | 246/275 | `bicarb_first_6h` |
| Sodium | 50983 | 250/275 | `sodium_first_6h` |
| Potassium | 50971 | 250/275 | `potassium_first_6h` |

Each lab gets a **missingness indicator** column (`lactate_missing_6h`, etc.) since missingness is informative in ICU data (sicker patients get more labs drawn). Coverage rates above are per all admissions; coverage restricted to the first-ICU-stay cohort will be verified during build and may be lower.

**Note**: Using only the first value in 0–6h (rather than worst-in-24h) prevents contamination from post-treatment lab values that reflect the fluid management decisions we are trying to evaluate.

#### Vitals (from `chartevents`) — first values in 0–6h window
| Vital | `itemid` | Coverage | Features |
|-------|----------|----------|----------|
| Heart rate | 220045 | 140/140 stays | `hr_first_6h`, `hr_max_6h` |
| MAP | 220052 | 65/140 stays | `map_first_6h`, `map_missing` |
| Respiratory rate | 220210 | 140/140 stays | `rr_first_6h` |

#### SOFA Score Components and GCS (from `chartevents`, `labevents`) — at ICU admission (0–6h)

SOFA and GCS are standard ICU severity metrics. Their omission would be immediately challenged by any critical care reviewer.

| Feature | Source | Logic |
|---------|--------|-------|
| `gcs_eye` | `chartevents` itemid 220739 | First value in 0–6h |
| `gcs_verbal` | `chartevents` itemid 223900 | First value in 0–6h |
| `gcs_motor` | `chartevents` itemid 223901 | First value in 0–6h |
| `gcs_total` | Derived | Sum of eye + verbal + motor |
| `sofa_renal` | Derived from creatinine | Creatinine-based SOFA renal component (0–4) |
| `sofa_hepatic` | Derived from bilirubin (50885) | Bilirubin-based SOFA hepatic component (0–4) |
| `sofa_coagulation` | Derived from platelets | Platelet-based SOFA coagulation component (0–4) |
| `sofa_respiratory` | Derived from PaO2/FiO2 if available, else from SpO2 | Oxygenation-based component (0–4); high missingness expected |
| `sofa_cardiovascular` | Derived from MAP + vasopressor at admission | MAP and vasopressor-at-admission based (0–4) |
| `sofa_neurological` | Derived from GCS | GCS-based component (0–4) |
| `sofa_total` | Sum of components | Composite SOFA score (0–24) |

#### Pre-treatment intervention status — DAG-justified confounders measured BEFORE treatment window
| Feature | Source | Logic |
|---------|--------|-------|
| `vasopressor_at_admission` | `inputevents` itemids [221906, 221289, 221749, 222315, 221662] | Binary: vasopressor running at ICU admission (within first 1h). This is a severity indicator measured before the treatment window, NOT a mediator. |
| `mechanical_vent_at_admission` | `procedureevents` itemid 225792 | Binary: invasive ventilation initiated within first 6h |

**Variables explicitly EXCLUDED per DAG**:
- ~~`vasopressor_dose_norepi_equiv_24h`~~ — Removed. Vasopressor dose over 0–24h is titrated in response to fluid resuscitation, making it a time-varying confounder affected by prior treatment (Hernan & Robins, Ch. 9). Adjusting would bias CATE.
- ~~`total_urine_output_24h`~~ — Removed. Urine output is a direct consequence of fluid administration (more fluids → more urine), making it a mediator on the causal path A → M → Y. Adjusting blocks part of the causal effect.
- ~~`total_calories_48h`~~ — Removed. Caloric delivery over 0–48h overlaps with the treatment window and may be influenced by fluid management decisions.

**Imputation strategy**: Median imputation with binary missingness indicators for all features. No iterative/model-based imputation — at n=100, the added complexity introduces more noise than it resolves.

**Total feature count**: ~30 features (reduced from ~55) before missingness indicators, ~42 with indicators. The reduced set follows from the DAG — a tighter, principled covariate set with ~30 features for ~50 patients per arm yields a healthier ~1.7:1 sample-to-feature ratio.

### Meta-Learner Architecture

#### Primary: T-Learner with XGBoost

```
T-Learner:
  μ₁(x) = E[Y | X=x, A=1]    # XGBoost trained on liberal-fluid patients
  μ₀(x) = E[Y | X=x, A=0]    # XGBoost trained on restrictive-fluid patients
  CATE(x) = μ₁(x) - μ₀(x)    # Individual treatment effect
```

**Why T-Learner for n=100**:
- Each arm gets ~50 patients — tight, but XGBoost with shallow trees (max_depth=3, n_estimators=100) can learn in this regime
- T-Learner is the most transparent meta-learner: two separate models, easy to audit
- Known limitation: T-Learner can overfit treatment effect heterogeneity. Mitigated by aggressive regularization (min_child_weight=10, subsample=0.8)

**XGBoost hyperparameters** (conservative for n=50 per arm):
```python
params = {
    'max_depth': 3,
    'n_estimators': 100,
    'learning_rate': 0.05,
    'min_child_weight': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'random_state': 42,
}
```

#### Comparison: S-Learner with XGBoost

```
S-Learner:
  μ(x, a) = E[Y | X=x, A=a]    # Single XGBoost with treatment as a feature
  CATE(x) = μ(x, 1) - μ(x, 0)
```

The S-Learner uses all 100 observations in one model (better for small n) but tends to shrink treatment effects toward zero due to regularization. We report both and compare CATE distributions.

#### Why NOT X-Learner, DR-Learner, or Causal Forest

| Method | Reason to exclude |
|--------|-------------------|
| X-Learner | Requires cross-fitting with pseudo-outcomes; at n=50 per arm, the imputed treatment effects are extremely noisy |
| DR-Learner | Needs propensity score estimation, which is unstable at n ≈ 55–65 with ~42 features — doubly robust becomes "doubly overfit" |
| Causal Forest | GRF's `causal_forest` partitions the sample recursively; with n=100, leaves contain <5 observations, producing meaningless local estimates |
| Any deep-learning method | Catastrophically overfit at n=100; not considered |

### Cross-Validation Plan

#### Primary: Leave-One-Out Cross-Validation (LOOCV)

```
For i in 1..100:
    Train T-Learner on 99 patients (excluding patient i)
    Predict μ₁(xᵢ) and μ₀(xᵢ) for held-out patient i
    Record: ŷᵢ (predicted outcome under observed treatment)
           CATEᵢ (predicted effect of switching treatment)
```

**Important distinction**: LOOCV validates the **nuisance models** (outcome prediction under each treatment arm), NOT the CATE estimates themselves. There is no ground-truth CATE in observational data — you never observe both potential outcomes for any patient. We are explicit about this.

**Nuisance model metrics** (on LOOCV out-of-fold predictions of observed outcomes):
- **R²** of predicted vs. actual log(LOS) (pooling predictions from the observed-treatment model)
- **MAE** and **RMSE** on log(LOS)
- **Spearman rank correlation** between predicted and actual LOS
- **Calibration plot**: predicted vs. actual LOS in quintiles

**CATE-specific evaluation** (since we cannot observe counterfactuals):
- **AUTOC** (Area Under the TOC curve, Yadlowsky et al. 2021): Ranks patients by estimated CATE, then measures whether patients predicted to benefit most from treatment actually show larger outcome differences. This is the principled metric for evaluating CATE *ranking* quality.
- **T-learner vs. S-learner CATE agreement**: If T-learner Var(CATE) is >5x S-learner Var(CATE), this flags T-learner overfitting rather than real heterogeneity. We report both distributions and their ratio.
- **Permutation test** (see below): The only direct statistical test for CATE heterogeneity.

#### Sensitivity: 10-Fold Stratified CV
Stratified on treatment assignment (liberal vs. restrictive). Used only if LOOCV proves computationally prohibitive (unlikely at n=100).

#### Permutation test for CATE heterogeneity
To test whether the meta-learner is finding real heterogeneity (vs. overfitting noise):
```
1. Compute variance of CATE estimates: Var(CATE_LOOCV)
2. Permute treatment labels 1000 times
3. Re-run T-Learner LOOCV each time
4. p-value = fraction of permuted Var(CATE) >= observed Var(CATE)
```
If p > 0.10, we conclude that the CATE heterogeneity is not distinguishable from noise — an honest negative result.

### Interpretability Stack

#### 1. SHAP Waterfall Plots (per patient)

For each patient, generate a SHAP waterfall plot showing which features drive the **predicted outcome under observed treatment**. Uses `shap.TreeExplainer` on the XGBoost model from the T-Learner arm that matches the patient's observed treatment.

```python
import shap
explainer = shap.TreeExplainer(model_treated)  # or model_control
shap_values = explainer(X_patient)
shap.waterfall_plot(shap_values[0])
```

Additionally: a **SHAP-based CATE decomposition** — which features drive the *difference* between μ₁(x) and μ₀(x) for each patient.

**Note on methodology** (per Guo et al. 2025): Taking the naive difference of SHAP values from two independently fitted models (`SHAP_μ₁_j - SHAP_μ₀_j`) does NOT produce valid Shapley values for the CATE function, because the two models have different background distributions and feature interactions. Instead, we use a **surrogate model approach**:

```python
# 1. Compute CATE estimates for all patients from the T-learner
cate_hat = mu1_predictions - mu0_predictions

# 2. Fit a lightweight surrogate model: X → CATE_hat
surrogate = XGBRegressor(max_depth=2, n_estimators=50)
surrogate.fit(X_covariates, cate_hat)

# 3. Compute SHAP on the surrogate — this is a valid Shapley decomposition
#    of the CATE function approximation
explainer_cate = shap.TreeExplainer(surrogate)
shap_cate_values = explainer_cate(X_covariates)
```

This produces a principled decomposition: "which features drive the predicted *treatment effect* for this patient?" The surrogate is shallow (depth-2) to avoid overfitting the already-noisy CATE estimates.

#### 2. KNN Clinical Comparator Retrieval

For each index patient, retrieve k=5 nearest neighbors in standardized covariate space (excluding treatment and outcome):

```python
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_covariates)
nn = NearestNeighbors(n_neighbors=5, metric='cosine')
nn.fit(X_scaled)
distances, indices = nn.kneighbors(X_scaled[patient_idx].reshape(1, -1))
```

**Display per patient**: A table showing for each of the 5 nearest neighbors:
- Their actual treatment received (liberal/restrictive)
- Their actual ICU LOS
- Their CATE estimate
- Cosine distance from the index patient

This serves as a **clinical audit layer** — a clinician can see: "Among patients most similar to this one, those who received restrictive fluids had shorter LOS."

#### 3. Policy Summary

Across all 100 patients, report:
- **Optimal action distribution**: What % of patients have CATE < 0 (benefit from liberal) vs. CATE > 0 (benefit from restrictive)?
- **Subgroup analysis** (exploratory, Bonferroni-corrected for 4 comparisons): Stratify optimal action by:
  - ICU type (surgical vs. medical)
  - Sepsis (yes/no)
  - Vasopressor at admission (yes/no)
  - AKI diagnosis (yes/no)
  - Note: With n ≈ 55–65 and binary subgroups, some cells may have <15 patients. All subgroup results are presented as hypothesis-generating, not confirmatory.
- **Policy tree visualization**: Fit a shallow (depth-2) decision tree on `X → optimal_action` to produce a human-readable rule
- **Value of personalization**: Compare mean predicted LOS under (a) observed actions, (b) always-liberal, (c) always-restrictive, (d) meta-learner-recommended action

---

## Expected Deliverables

- [ ] **`src/extract.py`** — Cohort extraction and feature engineering pipeline (all table joins, time-windowing, imputation)
- [ ] **`src/models.py`** — T-Learner and S-Learner implementations with XGBoost, LOOCV harness
- [ ] **`src/interpret.py`** — SHAP waterfall generator, KNN comparator retriever, CATE decomposition
- [ ] **`src/policy.py`** — Policy summary statistics, policy tree, value-of-personalization calculator
- [ ] **`notebooks/01_eda.ipynb`** — Exploratory data analysis: cohort description, feature distributions, treatment-outcome relationship
- [ ] **`notebooks/02_results.ipynb`** — Full results notebook: LOOCV performance, CATE distribution, SHAP waterfalls, KNN audit tables, policy summary
- [ ] **`figures/`** — Publication-quality figures (300 DPI):
  - Fig 1: Cohort flow diagram
  - Fig 2: CATE distribution (histogram + density)
  - Fig 3: SHAP waterfall for 3 representative patients (best/worst/median CATE)
  - Fig 4: KNN comparator table for same 3 patients
  - Fig 5: Policy tree (depth-2 decision boundaries)
  - Fig 6: Value-of-personalization bar chart
- [ ] **`LOGBOOK.md`** — Timestamped experiment log per lab conventions
- [ ] **Manuscript-ready methods section** — 800-word description suitable for supplementary material

### Streamlit Proof-of-Concept Application

- [ ] **`app/streamlit_app.py`** — Interactive Streamlit dashboard with the following pages/sections:

#### Page 1: "About This Project"
- Clear header: **"Prescriptive ICU: A Proof of Concept"**
- Plain-language explanation of what prescriptive ML is and why it matters for ICU fluid management
- Dataset disclosure: "Built on MIMIC-IV Clinical Database Demo (100 de-identified ICU patients from Beth Israel Deaconess Medical Center). This is an open-access demonstration dataset — not a clinical tool."
- Visual cohort summary (patient count, ICU types, outcome distribution)
- Methodology overview with expandable sections for technical readers (meta-learner architecture, LOOCV, SHAP)

#### Page 2: "Patient Explorer"
- Dropdown or slider to select a patient from the cohort
- For the selected patient, display:
  - **SHAP waterfall plot** — what drove this patient's predicted outcome
  - **Counterfactual comparison** — predicted LOS under liberal vs. restrictive fluid strategy, with the estimated benefit highlighted
  - **KNN comparator table** — 5 most similar patients, their actual treatments and outcomes
- Explanatory text framing each visualization for a clinical audience

#### Page 3: "Population-Level Insights"
- CATE distribution plot (who benefits from which strategy?)
- Policy tree visualization (simple decision rules)
- Subgroup breakdown (surgical vs. medical, sepsis, vasopressor use)
- Value-of-personalization summary

#### Page 4: "Partner With Us" (Call to Action)
- **Framing**: "This proof of concept demonstrates a prescriptive ML framework on 100 patients. With credentialed access to the full MIMIC-IV database (200,000+ admissions), external validation on eICU-CRD, and clinical domain expertise, this becomes a fundable research program."
- **What we're looking for**: ICU researchers, clinical informaticists, critical care physicians interested in co-developing this into a grant-funded study (R21/R01-scale)
- **What we bring**: The complete ML pipeline (feature engineering, meta-learner architecture, interpretability stack), cross-domain expertise in causal ML, and a working codebase ready to scale
- **What a partner brings**: Clinical domain authority, IRB access, credentialed MIMIC-IV/eICU access, patient-facing validation expertise, co-PI status on grant applications
- **Privacy-first design callout** (prominent, with lock icon):
  - "This tool runs **100% locally** — no internet connection required, no data leaves your machine."
  - "Designed for air-gapped clinical workstations behind your institutional firewall."
  - "Retrain on **your own patients** with a single command. The model adapts to your case mix, your protocols, your documentation patterns."
- **Federated learning vision** (expandable section with the diagram from the proposal):
  - Plain-language explanation: "Multiple hospitals each train the model on their own data. Only learned patterns — not patient records — are shared and combined. The result is a multi-center model built without any institution ever seeing another's patients."
  - "This turns a single-site proof of concept into a multi-center research program, with a dramatically simplified IRB pathway."
- **Scaling roadmap**: Visual timeline showing:
  - Phase 1: Demo (current, 100 patients) — proof of concept
  - Phase 2: Partner site retrain (your institution's data) — local validation
  - Phase 3: Full MIMIC-IV (200K+) — credentialed scale-up
  - Phase 4: Federated multi-site consortium — external validation
  - Phase 5: Prospective pilot — clinical integration
- **Contact/next steps**: Link to GitHub repository + downloadable 1-pager PDF
- `st.download_button` for the partnership brief PDF

- [ ] **`app/assets/partnership_brief.pdf`** — A single-page PDF (generated from `app/assets/partnership_brief.md` via a build step) containing:
  - Project title and 2-sentence summary
  - Key figure: CATE distribution or policy tree (one compelling visual)
  - "The Opportunity" — 3-bullet summary of what full-scale development would enable
  - "Privacy-First" callout — runs locally, no data leaves your institution, retrain on your own patients
  - "Federated Path" — one sentence on multi-site consortium without centralizing data
  - "What We're Seeking" — co-PI, clinical site, credentialed data access
  - "Funding Pathway" — target mechanisms (NIH R21 for pilot, R01 for full study; AHRQ PCORI for comparative effectiveness)
  - Contact information and GitHub link
  - QR code linking to the Streamlit app URL

- [ ] **`app/requirements.txt`** — Streamlit app dependencies (streamlit, plotly, shap, xgboost, pandas, etc.)
- [ ] **`app/README.md`** — Deployment instructions (local run + Streamlit Cloud)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **n=100 too small for stable CATE estimates** | **H** | **H** | Aggressive regularization (max_depth=3, min_child_weight=10); permutation test for CATE heterogeneity; report confidence intervals via LOOCV; present as methods demonstration, not clinical recommendation |
| **Confounding by indication** — sicker patients get more fluids AND have longer LOS | **H** | **H** | Include severity proxies (lactate, vasopressor use, ventilation status) as covariates; acknowledge unobserved confounding explicitly; run sensitivity analysis (E-value or partial R² bounds) |
| **Positivity violation** — some covariate strata may have only one treatment level | **M** | **H** | Check propensity score overlap; trim patients with extreme propensity (<0.1 or >0.9); report effective sample size after trimming |
| **Fluid balance measurement error** — some inputs/outputs may not be charted | **M** | **M** | Use `inputevents` + `outputevents` only (not estimating insensible losses); acknowledge as systematic undercount of true fluid balance |
| **Median split creates artificial groups** — treatment is really continuous | **M** | **M** | Report sensitivity analysis with tercile split (3 groups) and continuous treatment dose-response via GAM; median split is a simplification for the meta-learner framework |
| **MAP available in only 65/140 stays** (arterial line dependent) | **H** | **L** | Use missingness indicator; MAP is a covariate not the treatment; impute with median |
| **Albumin and lactate have ~50% missingness** | **M** | **L** | Missingness indicators; these are informative covariates but not essential for CATE estimation |
| **Immortal time bias** — patients with LOS < 48h mechanically have lower fluid balance AND shorter LOS | **H** | **H** | Primary analysis restricts to LOS >= 48h; sensitivity analysis prorates fluid balance to actual observation period. Explicitly reported as a design choice. |
| **Treatment-confounder feedback** (dynamic confounding) — fluids in hours 0–24 affect labs/vitals in hours 0–24, which are then used as confounders | **H** | **H** | Mitigated by restricting confounder window to 0–6h (pre-treatment). Standard conditional ignorability cannot fully handle this; acknowledged as limitation. |
| **Informative censoring** — 11 deaths truncate LOS by mortality, not recovery | **M** | **M** | Log(LOS) for deceased patients has qualitatively different meaning. Report sensitivity analysis excluding deaths (n ≈ 50–55 after all exclusions). |
| **Demo cohort selection bias** — 100 patients may not be random sample of full MIMIC-IV | **M** | **M** | PhysioNet selection criteria not fully documented. Limits "path to scale" claims. Acknowledged in limitations. |
| **Streamlit app misinterpreted as clinical tool** | **M** | **H** | Prominent disclaimers on every page ("Proof of concept — not for clinical use"); no patient-identifiable data (MIMIC demo is de-identified); frame as research partnership recruitment, not decision support |

---

## Resource Estimate

- **Compute**: Laptop-scale. XGBoost LOOCV on n ≈ 55–65 with ~42 features runs in <30 seconds. Permutation test (1000 iterations) ~30 minutes. Entire pipeline runs offline with no internet required.
- **Data acquisition**: Zero — demo is open access, already downloaded and extracted.
- **Estimated agent-hours**: ~6–8 hours of Claude Code time for full pipeline build (extract → model → interpret → notebook → Streamlit app → partnership brief).
- **Human review time**: ~3 hours for James to review proposal, audit feature engineering logic, verify clinical reasonableness of CATE interpretations, and refine partnership brief messaging for target audience.
- **Dependencies**: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`, `streamlit`, `plotly`. No exotic packages. Partnership brief PDF generation via `weasyprint` or `markdown-pdf` (one-time build step).

---

## Cross-Pollination

| Connection | Detail |
|------------|--------|
| **CBRE project** (Causal ML / Systems Biology) | Shares the causal inference methodological core. Meta-learner infrastructure built here can be templated into CBRE's biological intervention prediction. The SHAP-CATE decomposition technique is directly transferable. |
| **ECOCHEMICAL** (NP Discovery) | The "prescriptive" framing — simulating counterfactual interventions to find optimal actions — mirrors ECOCHEMICAL's goal of finding optimal co-culture conditions. The policy tree output format generalizes to any action-optimization setting. |
| **commons/ infrastructure** | The LOOCV + permutation test harness, KNN comparator retriever, and SHAP waterfall generator are all reusable components that should migrate to `commons/` after validation. |

---

## Success Criteria

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Nuisance model R² (LOOCV) | > 0.15 | Modest bar; ICU LOS is inherently noisy. R² > 0.15 means the outcome model captures more signal than random noise. Note: this validates the nuisance model, not the CATE. |
| CATE heterogeneity permutation test | p < 0.10 | Evidence that the meta-learner is detecting real treatment effect heterogeneity, not overfitting |
| AUTOC (CATE ranking quality) | > 0 with 95% CI excluding 0 | Evidence that patients ranked as high-CATE actually show larger outcome differences |
| T-learner / S-learner agreement | Var(CATE_T) / Var(CATE_S) < 5 | If ratio > 5, the T-learner is likely overfitting heterogeneity |
| Propensity score overlap | All patients have P(A=1\|X) ∈ [0.1, 0.9] | Ensures no extreme positivity violations that would invalidate causal estimates |
| Clinical face validity | Reviewed by James | CATE direction and magnitude make clinical sense (e.g., restrictive fluids benefit patients with AKI/heart failure; liberal fluids benefit hypovolemic/septic patients) |
| Reproducibility | Single `make run` or `python -m src.main` | Full pipeline runs end-to-end from raw `.csv.gz` files to all figures and results |
| Streamlit app runs | `streamlit run app/streamlit_app.py` launches without errors | All 4 pages render, patient explorer is interactive, PDF downloads correctly |
| Partnership brief | 1-page PDF passes "30-second scan" test | An ICU researcher can understand the opportunity, the ask, and how to respond within 30 seconds of looking at it |
| Code quality | Passes `ruff check`, all functions typed and docstringed | Per lab standards |

---

## Limitations (Stated Upfront)

This is a **methods demonstration**, not a clinical recommendation. The following limitations are fundamental and must be prominently stated in any writeup:

1. **n=100 is insufficient for reliable causal effect estimation**. After restricting to LOS >= 48h and excluding ultra-short stays, the analytic sample drops to ~55–65 patients. The CATE estimates will have wide confidence intervals. This project demonstrates the *pipeline and interpretability framework*, not clinically actionable findings.

2. **Single-center data** (Beth Israel Deaconess Medical Center). The 100-patient demo may not be a random sample of the full MIMIC-IV database. Findings do not generalize.

3. **Observational confounding** is not fully addressed. Conditional ignorability (Y(a) ⊥ A | W) is assumed but not testable. Unobserved confounders (clinician gestalt, family preferences, pre-ICU trajectory) may bias CATE estimates. E-value sensitivity analysis quantifies the required confounder strength.

4. **Dynamic treatment regime simplification**: Cumulative 48h fluid balance is a composite summary of many sequential decisions (boluses, rate changes, diuretics). Treating it as a single binary action ignores feedback loops (patient responds to fluids → clinician adjusts). Marginal structural models or dynamic treatment regimes would be more appropriate but require larger samples.

5. **Median split on fluid balance** is a convenience binarization. The median is data-dependent, and the binary split loses dose-response information. Sensitivity analyses with tercile split and GAM dose-response are reported.

6. **Time-zero bias mitigated but not eliminated**: Confounders are measured in the 0–6h window; treatment is measured over 0–48h. Residual overlap exists in the first 6 hours.

7. **Informative censoring**: The 11 deaths in the cohort have ICU LOS truncated by mortality, not recovery. Log(LOS) has qualitatively different meaning for these patients.

8. **Immortal time bias**: Addressed by restricting primary analysis to LOS >= 48h, but this reduces sample size and may introduce selection on the outcome.

9. **No external validation cohort**. Internal LOOCV only. External validation requires credentialed access to full MIMIC-IV or eICU-CRD.

---

## Path to Scale: Full MIMIC-IV (200K+ Admissions)

After demonstrating the pipeline on the 100-patient demo, the same codebase extends to full MIMIC-IV with these changes:

| Aspect | Demo (current) | Full MIMIC-IV |
|--------|---------------|---------------|
| Access | Open | Requires PhysioNet credentialing + CITI training |
| n | 100 patients | ~50,000 first ICU stays |
| Outcome model | XGBoost T-Learner | Same, but can also run Causal Forest (GRF) and DR-Learner with stable estimates |
| CV strategy | LOOCV | 5-fold or 10-fold (LOOCV unnecessary at large n) |
| Treatment definition | Median split | Can model continuous treatment via generalized propensity score or dose-response curves |
| Additional actions | Fluid balance only | Add early EN timing, vasopressor initiation, ventilation weaning strategy |
| Validation | Internal only | Temporal split (train on 2008–2016, test on 2017–2022) or external validation on eICU-CRD |
| Publication potential | Methods paper / technical report | Full clinical research paper with policy implications |

The code architecture (`src/extract.py`, `src/models.py`, etc.) is designed to swap in the full dataset with only changes to file paths and cohort size parameters.

### Local-Only Execution: No Internet Required

The finalized application is designed to run **entirely offline on a local machine** with no internet connectivity. This is a deliberate architectural decision for clinical data security:

- **All computation is local**: XGBoost training, SHAP computation, KNN retrieval, and Streamlit rendering all run on the user's hardware. No data leaves the machine.
- **No cloud dependencies**: No API calls, no telemetry, no external model hosting. The Streamlit app serves from `localhost` only.
- **Air-gapped compatible**: A partner institution can install the pipeline on an isolated clinical workstation behind their firewall, load their own data, train, and analyze — with zero network exposure.
- **Dependency bundle**: All Python packages can be pre-installed via `pip install --no-index` from a vendored wheel directory for fully offline deployment.

This matters because clinical data (even de-identified data under HIPAA Safe Harbor) is subject to institutional data governance policies that often prohibit cloud processing. An air-gapped, local-only tool removes that barrier entirely.

### Federated Learning: Better Models Without Sharing Patient Data

**What is federated learning?** In plain terms: instead of collecting patient data from multiple hospitals into one central database (which raises enormous privacy, legal, and logistical barriers), federated learning lets each hospital train the model *on their own data, on their own servers*. Only the learned model parameters (numbers, not patient records) are shared and combined. No individual patient record ever leaves its home institution.

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Hospital A   │    │  Hospital B   │    │  Hospital C   │
│  (200 pts)    │    │  (500 pts)    │    │  (1000 pts)   │
│               │    │               │    │               │
│ Train locally │    │ Train locally │    │ Train locally │
│ on own data   │    │ on own data   │    │ on own data   │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                    │                    │
       │   Model weights    │   Model weights    │
       │   (no patient      │   (no patient      │
       │    data)           │    data)           │
       ▼                    ▼                    ▼
    ┌──────────────────────────────────────────────┐
    │           Central Aggregation Server          │
    │   Combines model updates → improved model     │
    │   Sends updated model back to each site       │
    └──────────────────────────────────────────────┘
```

**Why this matters for our pipeline**:
- The T-learner architecture (two XGBoost models) is naturally suited to federated aggregation — each hospital trains its own local T-learner, and model parameters are averaged across sites using established federated XGBoost protocols (e.g., via NVIDIA FLARE or Flower framework).
- Multi-site data would dramatically improve CATE estimation: n=100 at one site becomes n=5,000+ across a consortium, enabling causal forests and DR-learners that are unstable at small n.
- Each site's patients reflect local clinical practices, case mix, and protocols — federated learning preserves this diversity rather than homogenizing it, producing models that generalize better.
- **IRB pathway**: Federated learning simplifies IRB review because no protected health information crosses institutional boundaries. Each site operates under its own IRB with a shared protocol.

### Retraining on Partner Data: The Model Gets Better on YOUR Patients

The pipeline is designed for **local retraining** — a partner institution can plug in their own ICU data and immediately get a model tuned to their patient population:

1. **Data format**: The partner exports their EHR data into the same table schema as MIMIC-IV (a common format that many institutions already use or can map to). A data dictionary and mapping guide will be provided.
2. **One-command retrain**: `python -m src.main --data-dir /path/to/your/data --retrain`. The pipeline re-runs feature extraction, re-fits the T-learner, recomputes SHAP values and KNN comparators, and regenerates the Streamlit dashboard — all reflecting the partner's own patients.
3. **Transfer learning benefit**: The initial model (trained on MIMIC-IV demo or full MIMIC-IV) provides sensible defaults for hyperparameters and feature importance rankings. Retraining on local data then adapts these to the partner's case mix, local protocols, and documentation patterns.
4. **Continuous improvement**: As the partner accumulates more patients, periodic retraining incorporates new data. The LOOCV and permutation test harnesses automatically re-evaluate whether the CATE heterogeneity signal is strengthening with larger n.

**The pitch to a partner**: "You don't need to send us your data. We send you the tool. You run it on your patients, behind your firewall. The model learns *your* institution's patterns. If the results are promising, we write the grant together."

---

## Go/No-Go Recommendation

**Go — with explicit framing as a methods demonstration and partnership recruitment tool.**

**Arguments for**:
- The pipeline is buildable in a single weekend session (~6–8 agent-hours including Streamlit app)
- It produces a genuinely useful, reusable causal ML toolkit that transfers to CBRE and ECOCHEMICAL
- The interpretability stack (SHAP waterfalls + KNN comparators + policy trees) differentiates from Zhang et al. (2024/2025) and makes a compelling demonstration piece
- Fluid management is in genuine clinical equipoise — this isn't a solved problem
- The MIMIC-IV demo is free, no access barriers
- All dependencies are standard (xgboost, shap, sklearn, streamlit) — no infrastructure risk
- The local-only, retrain-on-your-data architecture removes the biggest barrier to clinical adoption (data governance)
- The federated learning roadmap gives potential partners a clear, fundable multi-phase vision

**Arguments against**:
- n ≈ 55–65 (after LOS >= 48h restriction) will produce noisy CATE estimates — the permutation test may well return p > 0.10 (honest null result)
- Zhang et al. (2024/2025) substantially preempts the causal question; our contribution is the *deployment and interpretability stack*, not clinical novelty
- Observational confounding is real and unresolvable without stronger designs
- Risk of over-interpreting results from a demo dataset

**Recommendation**: Proceed, but commit upfront that a null result (no detectable heterogeneity) is a publishable and honest outcome. The value is threefold: (1) the *reusable pipeline* — causal ML infrastructure that transfers across the portfolio, (2) the *interpretability stack* — per-patient SHAP waterfalls + KNN audit + policy trees, and (3) the *partnership recruitment tool* — a Streamlit app that shows an ICU researcher exactly what this framework does, then asks them to run it on their own data. Frame as: "Here is how you would do prescriptive ML on ICU data, demonstrated at small scale, designed to run on your patients behind your firewall, ready for federated multi-site validation."

---

## Appendix: MIMIC-IV Item ID Reference

### Vasopressors (`inputevents`)
| itemid | Label |
|--------|-------|
| 221906 | Norepinephrine |
| 221289 | Epinephrine |
| 221749 | Phenylephrine |
| 222315 | Vasopressin |
| 221662 | Dopamine |
| 221653 | Dobutamine |
| 221986 | Milrinone |

### Enteral Nutrition (`inputevents`)
| itemid | Label |
|--------|-------|
| 225937 | Ensure (Full) |
| 226877 | Ensure Plus (Full) |
| 229013 | Glucerna 1.2 (Full) |
| 229295 | Glucerna 1.5 (Full) |
| 229011 | Jevity 1.5 (Full) |
| 229010 | Jevity 1.2 (Full) |

### Parenteral Nutrition (`inputevents`)
| itemid | Label |
|--------|-------|
| 225916 | TPN w/ Lipids |
| 225917 | TPN without Lipids |
| 225920 | Peripheral Parenteral Nutrition |

### Ingredient Events
| itemid | Label |
|--------|-------|
| 226060 | Calories |
| 226221 | Enteral Nutrition Ingredient |
| 227079 | Parenteral Nutrition Ingredient |

### Key Labs (`labevents`)
| itemid | Label | Coverage |
|--------|-------|----------|
| 50813 | Lactate (blood) | 120/275 |
| 50912 | Creatinine | 250/275 |
| 50862 | Albumin | 129/275 |
| 51006 | Urea Nitrogen (BUN) | 250/275 |
| 51301 | WBC Count | 245/275 |
| 50931 | Glucose | 246/275 |
| 51265 | Platelet Count | 248/275 |
| 50885 | Bilirubin Total | 148/275 |
| 51222 | Hemoglobin | 246/275 |
| 50882 | Bicarbonate | 246/275 |
| 50983 | Sodium | 250/275 |
| 50971 | Potassium | 250/275 |

### Key Vitals (`chartevents`)
| itemid | Label | Coverage |
|--------|-------|----------|
| 220045 | Heart Rate | 140/140 |
| 220052 | Arterial Blood Pressure mean (MAP) | 65/140 |
| 220210 | Respiratory Rate | 140/140 |

### Urine Output (`outputevents`)
| itemid | Label |
|--------|-------|
| 226559 | Foley |
| 226560 | Void |
| 226631 | PACU Urine |
| 226627 | OR Urine |

### Ventilation (`procedureevents`)
| itemid | Label |
|--------|-------|
| 225792 | Invasive Ventilation |
| 225794 | Non-invasive Ventilation |
| 227194 | Extubation |
