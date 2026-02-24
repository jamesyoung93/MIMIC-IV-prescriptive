"""Cohort extraction and feature engineering for MIMIC-IV Prescriptive ICU.

Loads MIMIC-IV demo tables, builds the analytic cohort (first ICU stay per
patient, LOS >= 2 days), computes fluid balance treatment variable, extracts
pre-treatment confounders in the 0-6h window per the causal DAG, and returns
a fully imputed feature matrix ready for meta-learner modeling.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_DATETIME_COLS = {
    "icustays": ["intime", "outtime"],
    "admissions": ["admittime", "dischtime", "deathtime", "edregtime", "edouttime"],
    "inputevents": ["starttime", "endtime", "storetime"],
    "outputevents": ["charttime", "storetime"],
    "labevents": ["charttime", "storetime"],
    "chartevents": ["charttime", "storetime"],
    "procedureevents": ["starttime", "endtime", "storetime"],
}


def load_tables(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Load all required MIMIC-IV tables from csv.gz files."""
    base = Path(config["paths"]["mimic_dir"])
    tables: dict[str, pd.DataFrame] = {}

    file_map = {
        "icustays": base / "icu" / "icustays.csv.gz",
        "patients": base / "hosp" / "patients.csv.gz",
        "admissions": base / "hosp" / "admissions.csv.gz",
        "inputevents": base / "icu" / "inputevents.csv.gz",
        "outputevents": base / "icu" / "outputevents.csv.gz",
        "labevents": base / "hosp" / "labevents.csv.gz",
        "chartevents": base / "icu" / "chartevents.csv.gz",
        "diagnoses_icd": base / "hosp" / "diagnoses_icd.csv.gz",
        "procedureevents": base / "icu" / "procedureevents.csv.gz",
    }

    for name, path in file_map.items():
        parse_dates = _DATETIME_COLS.get(name, [])
        tables[name] = pd.read_csv(path, parse_dates=parse_dates)

    return tables


# ---------------------------------------------------------------------------
# Cohort construction
# ---------------------------------------------------------------------------

def build_cohort(
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build analytic cohort: first ICU stay per patient, LOS >= threshold."""
    icu = tables["icustays"].copy()

    # First ICU stay per patient
    icu = icu.sort_values("intime")
    cohort = icu.groupby("subject_id", as_index=False).first()

    # Exclude ultra-short stays (< min_los_hours)
    min_hours = config["cohort"]["min_los_hours"]
    cohort = cohort[cohort["los"] >= min_hours / 24.0].copy()

    # Primary cohort: LOS >= 2 days
    min_days = config["cohort"]["primary_min_los_days"]
    cohort = cohort[cohort["los"] >= min_days].copy()

    cohort = cohort.reset_index(drop=True)
    return cohort


# ---------------------------------------------------------------------------
# Treatment variable: fluid balance
# ---------------------------------------------------------------------------

def compute_fluid_balance(
    cohort: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> pd.Series:
    """Compute net fluid balance (mL) over 0-48h from ICU admission.

    CRITICAL: Filters inputevents to amountuom == 'ml' to avoid summing
    drug doses (mcg, mg, units) with fluid volumes.
    """
    window_h = config["treatment"]["window_hours"]
    unit_filter = config["treatment"]["input_unit_filter"]
    inp = tables["inputevents"]
    out = tables["outputevents"]

    balances = {}
    for _, row in cohort.iterrows():
        sid = row["stay_id"]
        t0 = row["intime"]
        t1 = t0 + pd.Timedelta(hours=window_h)

        # Inputs: only ml-unit rows
        mask_in = (
            (inp["stay_id"] == sid)
            & (inp["starttime"] >= t0)
            & (inp["starttime"] < t1)
            & (inp["amountuom"] == unit_filter)
        )
        total_in = inp.loc[mask_in, "amount"].sum()

        # Outputs: all categories (all are in ml)
        mask_out = (
            (out["stay_id"] == sid)
            & (out["charttime"] >= t0)
            & (out["charttime"] < t1)
        )
        total_out = out.loc[mask_out, "value"].sum()

        balances[sid] = total_in - total_out

    return pd.Series(balances, name="fluid_balance_48h")


def binarize_treatment(
    fluid_balance: pd.Series,
    config: dict[str, Any],
) -> pd.Series:
    """Binarize fluid balance: liberal (1) >= median, restrictive (0) < median."""
    method = config["treatment"]["binarize_method"]
    if method == "median":
        threshold = fluid_balance.median()
    else:
        raise ValueError(f"Unsupported binarize_method: {method}")
    return (fluid_balance >= threshold).astype(int).rename("treatment")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_demographics(
    cohort: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Extract demographic features from patients and admissions."""
    pts = tables["patients"]
    adm = tables["admissions"]

    merged = cohort[["stay_id", "subject_id", "hadm_id"]].merge(
        pts[["subject_id", "gender", "anchor_age"]], on="subject_id"
    ).merge(
        adm[["hadm_id", "race", "insurance", "admission_type"]], on="hadm_id"
    )

    demo = pd.DataFrame(index=merged["stay_id"])
    demo["age"] = merged["anchor_age"].values
    demo["gender_male"] = (merged["gender"].values == "M").astype(int)
    demo["race_white"] = merged["race"].str.contains("WHITE", case=False, na=False).astype(int).values
    demo["insurance_medicare"] = (merged["insurance"].values == "Medicare").astype(int)
    demo["admission_type_emergency"] = merged["admission_type"].str.contains(
        "EMERGENCY|URGENT", case=False, na=False
    ).astype(int).values

    return demo


def _has_diagnosis(
    dx: pd.DataFrame,
    hadm_ids: pd.Series,
    icd9_codes: list[str] | None = None,
    icd9_prefix: list[str] | None = None,
    icd10_prefix: list[str] | None = None,
) -> pd.Series:
    """Check if any diagnosis matches ICD-9 or ICD-10 codes."""
    matches = set()
    for _, row in dx.iterrows():
        code = str(row["icd_code"])
        ver = int(row["icd_version"])
        hadm = row["hadm_id"]

        if ver == 9:
            if icd9_codes and code in icd9_codes:
                matches.add(hadm)
            if icd9_prefix:
                for p in icd9_prefix:
                    if code.startswith(p):
                        matches.add(hadm)
        elif ver == 10:
            if icd10_prefix:
                for p in icd10_prefix:
                    if code.startswith(p):
                        matches.add(hadm)

    return hadm_ids.isin(matches).astype(int)


def extract_diagnoses(
    cohort: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Extract diagnosis-based features. Handles both ICD-9 and ICD-10."""
    dx = tables["diagnoses_icd"]
    dx_cohort = dx[dx["hadm_id"].isin(cohort["hadm_id"])]
    hadm_ids = cohort.set_index("stay_id")["hadm_id"]

    diag = pd.DataFrame(index=cohort["stay_id"])
    diag["n_diagnoses"] = (
        dx_cohort.groupby("hadm_id")["icd_code"]
        .count()
        .reindex(hadm_ids.values)
        .fillna(0)
        .values
    )

    dx_cfg = config["diagnoses"]
    diag["has_sepsis"] = _has_diagnosis(
        dx_cohort, hadm_ids,
        icd9_codes=dx_cfg["sepsis"].get("icd9"),
        icd10_prefix=dx_cfg["sepsis"].get("icd10_prefix"),
    ).values
    diag["has_aki"] = _has_diagnosis(
        dx_cohort, hadm_ids,
        icd9_prefix=dx_cfg["aki"].get("icd9_prefix"),
        icd10_prefix=dx_cfg["aki"].get("icd10_prefix"),
    ).values
    diag["has_heart_failure"] = _has_diagnosis(
        dx_cohort, hadm_ids,
        icd9_prefix=dx_cfg["heart_failure"].get("icd9_prefix"),
        icd10_prefix=dx_cfg["heart_failure"].get("icd10_prefix"),
    ).values
    diag["has_respiratory_failure"] = _has_diagnosis(
        dx_cohort, hadm_ids,
        icd9_prefix=dx_cfg["respiratory_failure"].get("icd9_prefix"),
        icd10_prefix=dx_cfg["respiratory_failure"].get("icd10_prefix"),
    ).values

    surgical = config["surgical_units"]
    diag["icu_unit_surgical"] = cohort["first_careunit"].isin(surgical).astype(int).values

    return diag


def _extract_first_values_chartevents(
    cohort: pd.DataFrame,
    chartevents: pd.DataFrame,
    item_map: dict[str, int],
    window_hours: float,
) -> pd.DataFrame:
    """Extract first chartevent value per item within time window."""
    result = pd.DataFrame(index=cohort["stay_id"])

    for name, itemid in item_map.items():
        vals = {}
        sub = chartevents[chartevents["itemid"] == itemid]
        for _, row in cohort.iterrows():
            sid = row["stay_id"]
            t0 = row["intime"]
            t1 = t0 + pd.Timedelta(hours=window_hours)
            mask = (sub["stay_id"] == sid) & (sub["charttime"] >= t0) & (sub["charttime"] < t1)
            matched = sub.loc[mask].sort_values("charttime")
            if len(matched) > 0:
                v = matched.iloc[0]["valuenum"]
                vals[sid] = v if pd.notna(v) else np.nan
            else:
                vals[sid] = np.nan
        result[name] = pd.Series(vals)

    return result


def extract_labs_windowed(
    cohort: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
    window_hours: float = 6.0,
) -> pd.DataFrame:
    """Extract first lab value per patient within the 0-Xh window."""
    labs = tables["labevents"]
    lab_items = config["itemids"]["labs"]
    hadm_map = cohort.set_index("hadm_id")[["stay_id", "intime"]]

    result = pd.DataFrame(index=cohort["stay_id"])

    for name, itemid in lab_items.items():
        sub = labs[labs["itemid"] == itemid].copy()
        vals = {}
        for hadm_id, info in hadm_map.iterrows():
            sid = info["stay_id"]
            t0 = info["intime"]
            t1 = t0 + pd.Timedelta(hours=window_hours)
            mask = (sub["hadm_id"] == hadm_id) & (sub["charttime"] >= t0) & (sub["charttime"] < t1)
            matched = sub.loc[mask].sort_values("charttime")
            if len(matched) > 0:
                v = matched.iloc[0]["valuenum"]
                vals[sid] = v if pd.notna(v) else np.nan
            else:
                vals[sid] = np.nan
        col = f"{name}_first_{int(window_hours)}h"
        result[col] = pd.Series(vals)

    return result


def extract_vitals_windowed(
    cohort: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
    window_hours: float = 6.0,
) -> pd.DataFrame:
    """Extract first vital values within time window.

    MAP uses arterial (220052) with NIBP (220181) fallback for full coverage.
    """
    ce = tables["chartevents"]
    vit_cfg = config["itemids"]["vitals"]

    # Simple vitals: HR, RR, SpO2, FiO2, Temp
    simple_items = {
        f"hr_first_{int(window_hours)}h": vit_cfg["hr"],
        f"rr_first_{int(window_hours)}h": vit_cfg["rr"],
        f"spo2_first_{int(window_hours)}h": vit_cfg["spo2"],
        f"fio2_first_{int(window_hours)}h": vit_cfg["fio2_chart"],
        f"temp_f_first_{int(window_hours)}h": vit_cfg["temp_f"],
    }
    result = _extract_first_values_chartevents(
        cohort, ce, simple_items, window_hours
    )

    # MAP: arterial with NIBP fallback
    map_art = _extract_first_values_chartevents(
        cohort, ce, {"map_art": vit_cfg["map_arterial"]}, window_hours
    )
    map_nibp = _extract_first_values_chartevents(
        cohort, ce, {"map_nibp": vit_cfg["map_nibp"]}, window_hours
    )
    col = f"map_first_{int(window_hours)}h"
    result[col] = map_art["map_art"].combine_first(map_nibp["map_nibp"])

    return result


def extract_gcs(
    cohort: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
    window_hours: float = 6.0,
) -> pd.DataFrame:
    """Extract GCS components (first value in window)."""
    ce = tables["chartevents"]
    gcs_cfg = config["itemids"]["gcs"]

    items = {
        "gcs_eye": gcs_cfg["eye"],
        "gcs_verbal": gcs_cfg["verbal"],
        "gcs_motor": gcs_cfg["motor"],
    }
    result = _extract_first_values_chartevents(cohort, ce, items, window_hours)
    result["gcs_total"] = result[["gcs_eye", "gcs_verbal", "gcs_motor"]].sum(axis=1)
    return result


# ---------------------------------------------------------------------------
# SOFA score
# ---------------------------------------------------------------------------

def compute_sofa(
    labs_df: pd.DataFrame,
    vitals_df: pd.DataFrame,
    gcs_df: pd.DataFrame,
    cohort: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Compute SOFA score components from extracted labs, vitals, GCS."""
    wh = int(config["feature_windows"]["confounders_hours"])
    sofa = pd.DataFrame(index=cohort["stay_id"])

    # Renal: creatinine thresholds
    cr = labs_df.get(f"creatinine_first_{wh}h", pd.Series(dtype=float))
    sofa["sofa_renal"] = np.select(
        [cr >= 5.0, cr >= 3.5, cr >= 2.0, cr >= 1.2],
        [4, 3, 2, 1],
        default=0,
    )
    sofa.loc[cr.isna(), "sofa_renal"] = np.nan

    # Hepatic: bilirubin thresholds
    bili = labs_df.get(f"bilirubin_first_{wh}h", pd.Series(dtype=float))
    sofa["sofa_hepatic"] = np.select(
        [bili >= 12.0, bili >= 6.0, bili >= 2.0, bili >= 1.2],
        [4, 3, 2, 1],
        default=0,
    )
    sofa.loc[bili.isna(), "sofa_hepatic"] = np.nan

    # Coagulation: platelets (inverted — lower is worse)
    plt_val = labs_df.get(f"platelets_first_{wh}h", pd.Series(dtype=float))
    sofa["sofa_coagulation"] = np.select(
        [plt_val < 20, plt_val < 50, plt_val < 100, plt_val < 150],
        [4, 3, 2, 1],
        default=0,
    )
    sofa.loc[plt_val.isna(), "sofa_coagulation"] = np.nan

    # Respiratory: PaO2/FiO2 when available, else SpO2 heuristic
    pao2 = labs_df.get(f"pao2_first_{wh}h", pd.Series(dtype=float))
    fio2_lab = labs_df.get(f"fio2_lab_first_{wh}h", pd.Series(dtype=float))
    fio2_chart = vitals_df.get(f"fio2_first_{wh}h", pd.Series(dtype=float))
    spo2 = vitals_df.get(f"spo2_first_{wh}h", pd.Series(dtype=float))

    # Combine FiO2 sources (lab takes priority, then chart)
    fio2 = fio2_lab.combine_first(fio2_chart)
    # FiO2 may be stored as percentage (0-100) or fraction (0-1)
    fio2 = fio2.where(fio2 <= 1.0, fio2 / 100.0)

    pf_ratio = pao2 / fio2.replace(0, np.nan)

    sofa_resp = pd.Series(np.nan, index=sofa.index)
    # Use PaO2/FiO2 where available
    has_pf = pf_ratio.notna()
    sofa_resp[has_pf] = np.select(
        [pf_ratio[has_pf] < 100, pf_ratio[has_pf] < 200,
         pf_ratio[has_pf] < 300, pf_ratio[has_pf] < 400],
        [4, 3, 2, 1],
        default=0,
    )
    # Fallback: SpO2-based heuristic (assumes approximate room air)
    no_pf = ~has_pf & spo2.notna()
    sofa_resp[no_pf] = np.select(
        [spo2[no_pf] < 88, spo2[no_pf] < 92, spo2[no_pf] < 96],
        [3, 2, 1],
        default=0,
    )
    sofa["sofa_respiratory"] = sofa_resp

    # Cardiovascular: MAP + vasopressor at admission
    map_col = f"map_first_{wh}h"
    map_val = vitals_df.get(map_col, pd.Series(dtype=float))

    # Get vasopressor-at-admission status
    vaso_items = config["itemids"]["vasopressors"]
    inp = tables["inputevents"]
    vaso_at_adm = {}
    for _, row in cohort.iterrows():
        sid = row["stay_id"]
        t0 = row["intime"]
        t1 = t0 + pd.Timedelta(hours=1)
        mask = (
            (inp["stay_id"] == sid)
            & (inp["itemid"].isin(vaso_items))
            & (inp["starttime"] >= t0)
            & (inp["starttime"] < t1)
        )
        vaso_at_adm[sid] = int(mask.any())
    vaso_series = pd.Series(vaso_at_adm)

    sofa_cv = pd.Series(0, index=sofa.index)
    sofa_cv[vaso_series == 1] = 2  # vasopressor at admission = at least 2
    sofa_cv[(map_val < 70) & (vaso_series == 0)] = 1  # low MAP without vasopressor
    sofa["sofa_cardiovascular"] = sofa_cv

    # Neurological: GCS-based
    gcs = gcs_df.get("gcs_total", pd.Series(dtype=float))
    sofa["sofa_neurological"] = np.select(
        [gcs < 6, gcs < 10, gcs < 13, gcs < 15],
        [4, 3, 2, 1],
        default=0,
    )
    sofa.loc[gcs.isna(), "sofa_neurological"] = np.nan

    # Total
    sofa["sofa_total"] = sofa[
        ["sofa_renal", "sofa_hepatic", "sofa_coagulation",
         "sofa_respiratory", "sofa_cardiovascular", "sofa_neurological"]
    ].sum(axis=1, min_count=1)  # NaN if all components are NaN

    return sofa


# ---------------------------------------------------------------------------
# Pre-treatment interventions
# ---------------------------------------------------------------------------

def extract_pre_treatment_interventions(
    cohort: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Extract binary pre-treatment intervention flags."""
    result = pd.DataFrame(index=cohort["stay_id"])

    # Vasopressor at admission (within 1h)
    vaso_items = config["itemids"]["vasopressors"]
    inp = tables["inputevents"]
    vaso = {}
    for _, row in cohort.iterrows():
        sid = row["stay_id"]
        t0 = row["intime"]
        t1 = t0 + pd.Timedelta(hours=1)
        mask = (
            (inp["stay_id"] == sid)
            & (inp["itemid"].isin(vaso_items))
            & (inp["starttime"] >= t0)
            & (inp["starttime"] < t1)
        )
        vaso[sid] = int(mask.any())
    result["vasopressor_at_admission"] = pd.Series(vaso)

    # Mechanical ventilation at admission (within 6h)
    pe = tables["procedureevents"]
    vent_id = config["itemids"]["ventilation"]["invasive"]
    vent = {}
    for _, row in cohort.iterrows():
        sid = row["stay_id"]
        t0 = row["intime"]
        t1 = t0 + pd.Timedelta(hours=6)
        mask = (
            (pe["stay_id"] == sid)
            & (pe["itemid"] == vent_id)
            & (pe["starttime"] >= t0)
            & (pe["starttime"] < t1)
        )
        vent[sid] = int(mask.any())
    result["mechanical_vent_at_admission"] = pd.Series(vent)

    return result


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

def add_missingness_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary missingness indicators for columns with any NaN."""
    indicators = {}
    for col in df.columns:
        if df[col].isna().any():
            indicators[f"{col}_missing"] = df[col].isna().astype(int)
    return pd.concat([df, pd.DataFrame(indicators, index=df.index)], axis=1)


def impute_median(df: pd.DataFrame) -> pd.DataFrame:
    """Impute NaN values with column median for numeric columns."""
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


# ---------------------------------------------------------------------------
# Feature curation
# ---------------------------------------------------------------------------

# Columns to DROP — these are either perfectly redundant (components when
# totals are present) or extremely sparse (>70% missing).
# Everything else is kept, including missingness indicators (they carry
# genuine prognostic signal — e.g., missing lactate indicates lower acuity).
_DROP_COLUMNS = [
    # SOFA components — redundant with sofa_total
    "sofa_renal", "sofa_hepatic", "sofa_coagulation",
    "sofa_respiratory", "sofa_cardiovascular", "sofa_neurological",
    # GCS components — redundant with gcs_total
    "gcs_eye", "gcs_verbal", "gcs_motor",
    # Extremely sparse labs (>70% missing) and their indicators
    "bilirubin_first_6h", "fio2_lab_first_6h",
    "bilirubin_first_6h_missing", "fio2_lab_first_6h_missing",
    # SOFA component missingness — redundant with the raw lab missingness
    "sofa_renal_missing", "sofa_hepatic_missing", "sofa_coagulation_missing",
]


def select_core_features(X: pd.DataFrame) -> pd.DataFrame:
    """Drop redundant/sparse columns, keeping all else."""
    to_drop = [c for c in _DROP_COLUMNS if c in X.columns]
    return X.drop(columns=to_drop)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def build_dataset(
    config: dict[str, Any],
    cohort_type: str = "primary",
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """End-to-end dataset construction.

    Returns (X, A, Y, patient_info) where:
    - X: Imputed feature matrix with missingness indicators
    - A: Binary treatment (0=restrictive, 1=liberal)
    - Y: log(ICU LOS in days)
    - patient_info: Display-friendly patient metadata
    """
    tables = load_tables(config)
    cohort = build_cohort(tables, config)

    window_h = config["feature_windows"]["confounders_hours"]

    # Treatment
    fluid_balance = compute_fluid_balance(cohort, tables, config)
    A = binarize_treatment(fluid_balance, config)
    A = A.reindex(cohort["stay_id"])

    # Outcome
    Y = pd.Series(
        np.log(cohort["los"].values),
        index=cohort["stay_id"],
        name="log_los",
    )

    # Features
    demographics = extract_demographics(cohort, tables)
    diagnoses = extract_diagnoses(cohort, tables, config)
    labs = extract_labs_windowed(cohort, tables, config, window_hours=window_h)
    vitals = extract_vitals_windowed(cohort, tables, config, window_hours=window_h)
    gcs = extract_gcs(cohort, tables, config, window_hours=window_h)
    sofa = compute_sofa(labs, vitals, gcs, cohort, tables, config)
    interventions = extract_pre_treatment_interventions(cohort, tables, config)

    # Assemble
    X = pd.concat(
        [demographics, diagnoses, labs, vitals, gcs, sofa, interventions],
        axis=1,
    )
    X = add_missingness_indicators(X)
    X = impute_median(X)
    X = select_core_features(X)

    # Patient info for display
    adm = tables["admissions"]
    patient_info = cohort[["stay_id", "subject_id", "hadm_id", "first_careunit", "los"]].copy()
    patient_info = patient_info.set_index("stay_id")
    patient_info["fluid_balance_ml"] = fluid_balance.reindex(patient_info.index)
    patient_info["treatment_label"] = A.map({1: "Liberal", 0: "Restrictive"}).reindex(patient_info.index)
    patient_info["age"] = demographics["age"].values
    patient_info["gender"] = demographics["gender_male"].map({1: "Male", 0: "Female"}).values

    # Merge hospital_expire_flag
    expire = adm.set_index("hadm_id")["hospital_expire_flag"]
    patient_info["hospital_expire_flag"] = (
        expire.reindex(patient_info["hadm_id"]).values
    )

    return X, A, Y, patient_info
