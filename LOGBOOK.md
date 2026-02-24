# Experiment Logbook

<!-- Append new entries below -->

## 2026-02-23 — Full pipeline run (primary cohort, LOS >= 2d)
**Hypothesis**: T-learner/S-learner meta-learners can detect heterogeneous treatment effects of liberal vs. restrictive fluid management on ICU length of stay.
**Result**: Primary learner: S-Learner. LOOCV R²=0.2806. AUTOC=1.3065. Surrogate R²=0.9883. 68% patients would switch treatment.
**Interpretation**: Results from 100-patient demo cohort. Proof-of-concept demonstrating feasibility of personalized fluid management. Statistical power limited by sample size; federation with partner sites is required for clinical conclusions.
**Metrics**:
- n_patients: 53
- n_features: 61
- t_learner_r2: 0.3799
- s_learner_r2: 0.2806
- autoc_t: 1.0194
- autoc_s: 1.3065
- primary_learner: S-Learner
- surrogate_r2: 0.9883
- policy_tree_accuracy: 0.8491
**Next steps**: Partner recruitment via Streamlit app. Validate on full MIMIC-IV (>50k patients). Multi-site federated analysis.

## 2026-02-23 — Full pipeline run (primary cohort, LOS >= 2d)
**Hypothesis**: T-learner/S-learner meta-learners can detect heterogeneous treatment effects of liberal vs. restrictive fluid management on ICU length of stay.
**Result**: Primary learner: T-Learner. LOOCV R²=-0.1121. AUTOC=0.3149. Surrogate R²=0.9875. 53% patients would switch treatment.
**Interpretation**: Results from 100-patient demo cohort. Proof-of-concept demonstrating feasibility of personalized fluid management. Statistical power limited by sample size; federation with partner sites is required for clinical conclusions.
**Metrics**:
- n_patients: 53
- n_features: 27
- t_learner_r2: -0.1121
- s_learner_r2: 0.2112
- autoc_t: 0.3149
- autoc_s: 0.1826
- primary_learner: T-Learner
- surrogate_r2: 0.9875
- policy_tree_accuracy: 0.6226
**Next steps**: Partner recruitment via Streamlit app. Validate on full MIMIC-IV (>50k patients). Multi-site federated analysis.

## 2026-02-23 — Full pipeline run (primary cohort, LOS >= 2d)
**Hypothesis**: T-learner/S-learner meta-learners can detect heterogeneous treatment effects of liberal vs. restrictive fluid management on ICU length of stay.
**Result**: Primary learner: T-Learner. LOOCV R²=0.4185. AUTOC=1.1466. Surrogate R²=0.9927. 58% patients would switch treatment.
**Interpretation**: Results from 100-patient demo cohort. Proof-of-concept demonstrating feasibility of personalized fluid management. Statistical power limited by sample size; federation with partner sites is required for clinical conclusions.
**Metrics**:
- n_patients: 53
- n_features: 45
- t_learner_r2: 0.4185
- s_learner_r2: 0.2717
- autoc_t: 1.1466
- autoc_s: 0.2392
- primary_learner: T-Learner
- surrogate_r2: 0.9927
- policy_tree_accuracy: 0.4906
**Next steps**: Partner recruitment via Streamlit app. Validate on full MIMIC-IV (>50k patients). Multi-site federated analysis.
