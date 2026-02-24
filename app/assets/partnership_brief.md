# Partnership Brief: Personalized ICU Fluid Management

## The Opportunity

Fluid management is one of the most common and consequential decisions in intensive care. Yet there is no consensus on optimal fluid strategy for individual patients — the CLASSIC, CLOVERS, and PROCESS trials show population-level equipoise between liberal and restrictive approaches.

**What if the answer is not one-size-fits-all, but patient-specific?**

## What We've Built

Using the MIMIC-IV Clinical Database Demo (100 patients, MIT open data), we developed a **proof-of-concept prescriptive ML framework** that:

- Estimates **individualized treatment effects** (CATE) of liberal vs. restrictive fluid management on ICU length of stay
- Uses **T-Learner and S-Learner** meta-learners with XGBoost, evaluated via Leave-One-Out Cross-Validation
- Provides **fully interpretable outputs**: SHAP waterfall plots, clinical comparator patients, and a depth-2 decision tree policy
- Identifies which patients would benefit from each strategy based on their admission characteristics

## Why Partner With Us

- **Privacy-first design**: All computation runs locally on your institution's infrastructure. No patient data leaves your site. We support a **federated learning** approach where only model weights are shared.
- **Retrain on your data**: The pipeline is designed to retrain on your own patient population, capturing site-specific patterns in fluid management and outcomes.
- **Scalable**: The demo uses 53 patients meeting inclusion criteria. With a full MIMIC-IV deployment (~50,000+ ICU stays) or multi-site federation, we can achieve the statistical power needed for clinical translation.
- **Publication-ready**: Causal identification strategy with explicit DAG, Bonferroni-corrected subgroup analyses, and permutation testing for heterogeneity.

## What We're Looking For

We are seeking **1-2 academic medical centers** interested in:

1. **Validating** the framework on their own de-identified ICU data
2. **Co-authoring** a multi-site study on personalized fluid management
3. **Applying** for NIH/AHRQ funding (R01/R21) for a prospective validation trial

## Ideal Partner Profile

- ICU research group with access to structured EHR data (MIMIC-IV format or compatible)
- Interest in precision medicine / causal ML for critical care
- IRB infrastructure for retrospective EHR studies
- Willingness to collaborate on a federated analysis protocol

## Funding Pathway

1. **Phase 1** (Current): Proof-of-concept on MIMIC-IV Demo — complete
2. **Phase 2**: Full MIMIC-IV validation (~50k ICU stays)
3. **Phase 3**: Multi-site retrospective validation (federated)
4. **Phase 4**: NIH R01 submission for prospective pilot
5. **Phase 5**: Prospective clinical decision support integration

## Contact

**James Young, PhD**
Research: Complexity science bridging biology, ML, and strategic foresight
GitHub: [View the code and interactive demo]

*This framework is a research proof-of-concept. It is not a medical device and should not be used for clinical decision-making without prospective validation.*
