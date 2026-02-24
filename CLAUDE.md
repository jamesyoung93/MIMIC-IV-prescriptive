# MIMIC-IV Prescriptive ICU — Project Context

**Status**: Approved with conditions (all blocking conditions addressed)
**Domain**: Clinical ML / Causal Inference / Prescriptive Analytics

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.main                    # Run full pipeline
streamlit run app/streamlit_app.py    # Launch dashboard
```

## Verification

```bash
ruff check src/ app/ tests/
pytest tests/ -v
```

## Key Constraint

This is a METHODS DEMONSTRATION on 100 patients (53 in primary cohort).
Never claim clinical validity. Always include disclaimers.

## Data Location

`data/` contains symlinks to `mimic-iv-clinical-database-demo-2.2/`.
All config in `configs/default.yaml` — no magic numbers in code.

## Module Dependency Order

`extract.py` → `models.py` → `interpret.py` / `policy.py` → `main.py` → `app/streamlit_app.py`
