"""
Preprocess eICU-CRD into the SAME artifact schema as preprocess.py's MIMIC-IV
output (cohort.parquet, timeseries.npz, features_agg.parquet), so every
downstream script -- partition.py, fed_train.py, run_experiments.py -- works
unchanged.

Why eICU-CRD is worth adding: MIMIC-IV is one hospital system. eICU-CRD is
~200 REAL, separate hospitals (patient.hospitalid), monitored by the Philips
eICU telehealth program. Using hospital as the client key turns the "care
units simulate institutions" limitation into an actual multi-institution
federation.

Schema notes (verified against the eICU-CRD documentation, not assumed):
  - Primary key: patientunitstayid (one ICU unit stay)
  - hospitalid: the real hospital -- this becomes our client key
  - unitvisitnumber == 1: first ICU stay for that hospitalization (eICU
    provides this directly; MIMIC-IV required a ROW_NUMBER() to derive it)
  - All time offsets are in MINUTES from ICU admission (unitAdmitOffset is
    always 0), NOT wall-clock timestamps
  - hospitaldischargestatus ('Alive'/'Expired'): in-hospital mortality, the
    same outcome definition used for MIMIC-IV's hospital_expire_flag
  - age is a string, '> 89' for the oldest bucket -- must be coerced

Known gaps versus the MIMIC-IV extraction, disclosed rather than hidden:
  - GCS (eyes/motor/verbal) is only available as ONE value per stay (from
    apacheApsVar, the worst value in the APACHE scoring window), not hourly.
    It is added to the STATIC/tabular branch, not the time-series branch.
  - Comorbidities are not extracted for eICU rows in this version (would
    need a separate ICD-9-based Charlson mapping from the diagnosis table).
    Left as zero/missing for eICU rows; do not interpret them for eICU.

Run:  python preprocess_eicu.py
Out:  <EICU_OUT_DIR>/cohort.parquet, timeseries.npz, features_agg.parquet
"""
import numpy as np
import pandas as pd
import duckdb

import config as C

# ---------------------------------------------------------------------------
# Point this at the folder containing the eICU-CRD CSVs (patient.csv.gz,
# vitalPeriodic.csv.gz, lab.csv.gz, apacheApsVar.csv.gz, infusionDrug.csv.gz,
# treatment.csv.gz, intakeOutput.csv.gz). Layout-agnostic, same rule as
# config.find_table: .csv.gz, extracted .csv, or nested folders all work.
# ---------------------------------------------------------------------------
from pathlib import Path

EICU_ROOT = Path(r"D:\fl_ly\physionet.org_eicu\files\eicu-crd\2.0")   # <-- CHANGE ME
EICU_OUT_DIR = C.OUT_DIR.parent / "artifacts_eicu"
EICU_OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CLIENT_SIZE_HOSPITAL = 300   # a hospital needs this many stays to be its
                                  # own federated client; smaller ones are
                                  # dropped downstream by partition.py anyway


def eicu_table(name):
    """Layout-agnostic locator, mirroring config.find_table for MIMIC."""
    for cand in [EICU_ROOT / f"{name}.csv.gz", EICU_ROOT / f"{name}.csv",
                EICU_ROOT / f"{name}.csv" / f"{name}.csv",
                EICU_ROOT / name.lower() / f"{name.lower()}.csv.gz",
                EICU_ROOT / f"{name.lower()}.csv.gz",
                EICU_ROOT / f"{name.lower()}.csv"]:
        if cand.is_file():
            opts = "compression='gzip'" if cand.suffix == ".gz" else ""
            return f"read_csv_auto('{cand.as_posix()}'" + (f",{opts})" if opts else ")")
    raise FileNotFoundError(f"eICU table '{name}' not found under {EICU_ROOT}")


# ---------------------------------------------------------------------------
# Map eICU's free-text lab names onto the same feature names used for MIMIC,
# so the resulting tensor has an IDENTICAL feature_names array. Confirmed
# against the standard eICU-CRD lab naming used across published benchmarks.
# ---------------------------------------------------------------------------
LAB_NAME_MAP = {
    "creatinine": ["creatinine"],
    "potassium": ["potassium"],
    "sodium": ["sodium"],
    "chloride": ["chloride"],
    "bicarbonate": ["bicarbonate", "HCO3"],
    "hematocrit": ["Hct"],
    "wbc": ["WBC x 1000"],
    "glucose": ["glucose"],
    "magnesium": ["magnesium"],
    "calcium": ["calcium"],
    "lactate": ["lactate"],
    "platelets": ["platelets x 1000"],
    "bilirubin": ["total bilirubin"],
    "bun": ["BUN"],
    "inr": ["PT - INR"],
    "ptt": ["PTT"],
    "albumin": ["albumin"],
    "alt": ["ALT (SGPT)"],
    "ast": ["AST (SGOT)"],
    "hemoglobin": ["Hgb"],
    "anion_gap": ["anion gap"],
    "ph": ["pH"],
    "po2": ["paO2"],
    "pco2": ["paCO2"],
    "base_excess": ["Base Excess"],
    "phosphate": ["phosphate"],
    "troponin": ["troponin - I"],
}

VASOPRESSOR_DRUGS = ["norepinephrine", "epinephrine", "dopamine",
                     "dobutamine", "vasopressin", "phenylephrine"]


def build_cohort(con):
    print("[1/5] Building eICU cohort ...")
    con.execute(f"""
    CREATE OR REPLACE TABLE cohort AS
    SELECT
        p.patientunitstayid AS stay_id,
        p.uniquepid AS subject_id,
        'eICU_Hospital_' || CAST(p.hospitalid AS VARCHAR) AS first_careunit,
        p.hospitalid,
        CASE WHEN p.age = '> 89' THEN 90.0
             WHEN p.age IS NULL OR p.age = '' THEN NULL
             ELSE TRY_CAST(p.age AS DOUBLE) END AS age,
        p.gender,
        'EMER.' AS admission_type,      -- eICU has no direct admission_type
                                         -- equivalent; kept for schema parity
        p.unitdischargeoffset / 60.0 AS los_hours,
        CASE WHEN p.hospitaldischargestatus = 'Expired' THEN 1 ELSE 0 END AS label,
        p.hospitaldischargeoffset
    FROM {eicu_table('patient')} p
    WHERE p.unitvisitnumber = 1
      AND (p.age = '> 89' OR TRY_CAST(p.age AS DOUBLE) >= {C.MIN_AGE})
      AND p.unitdischargeoffset / 60.0 >= {C.MIN_LOS_H}
      AND p.hospitaldischargestatus IS NOT NULL
    """)
    if C.DROP_DEATH_IN_WINDOW:
        # eICU has no explicit death timestamp -- approximate with hospital
        # discharge offset: if the patient died and the hospital stay ended
        # within the observation window, exclude (same leakage guard as MIMIC).
        con.execute(f"""
        DELETE FROM cohort
        WHERE label = 1 AND hospitaldischargeoffset <= {C.OBS_WINDOW_H * 60}
        """)
    df = con.execute("SELECT * FROM cohort").df()
    print(f"      {len(df):,} eICU stays | {df.subject_id.nunique():,} patients "
          f"| mortality {df.label.mean()*100:.2f}% | "
          f"{df.hospitalid.nunique()} distinct hospitals")
    n_big = (df.hospitalid.value_counts() >= MIN_CLIENT_SIZE_HOSPITAL).sum()
    print(f"      {n_big} hospitals have >= {MIN_CLIENT_SIZE_HOSPITAL} stays "
          f"(these become federated clients)")
    return df


def extract_vitals(con):
    print("[2/5] Extracting vitalPeriodic (hourly-binned) ...")
    con.execute(f"""
    CREATE OR REPLACE TABLE ev_vital AS
    SELECT v.patientunitstayid AS stay_id, feature, value,
           v.observationoffset / 60.0 AS hours_in
    FROM (
        SELECT patientunitstayid, observationoffset,
               unnest(['heart_rate','resp_rate','spo2','sbp','dbp','map','temp_c']) AS feature,
               unnest([heartrate, respiration, sao2, systemicsystolic,
                       systemicdiastolic, systemicmean, temperature]) AS value
        FROM {eicu_table('vitalPeriodic')}
    ) v
    JOIN cohort co ON v.patientunitstayid = co.stay_id
    WHERE v.observationoffset >= 0
      AND v.observationoffset < {C.OBS_WINDOW_H * 60}
      AND v.value IS NOT NULL
    """)
    n = con.execute("SELECT COUNT(*) FROM ev_vital").fetchone()[0]
    print(f"      {n:,} vital-sign readings")


def extract_labs(con):
    print("[3/5] Extracting labs ...")
    cases = " ".join(f"WHEN labname IN ({','.join(repr(n) for n in names)}) "
                     f"THEN '{feat}'"
                     for feat, names in LAB_NAME_MAP.items())
    con.execute(f"""
    CREATE OR REPLACE TABLE ev_lab AS
    SELECT co.stay_id, CASE {cases} END AS feature,
           TRY_CAST(l.labresult AS DOUBLE) AS value,
           l.labresultoffset / 60.0 AS hours_in
    FROM {eicu_table('lab')} l
    JOIN cohort co ON l.patientunitstayid = co.stay_id
    WHERE l.labresultoffset >= 0
      AND l.labresultoffset < {C.OBS_WINDOW_H * 60}
      AND TRY_CAST(l.labresult AS DOUBLE) IS NOT NULL
    """)
    con.execute("DELETE FROM ev_lab WHERE feature IS NULL")
    n = con.execute("SELECT COUNT(*) FROM ev_lab").fetchone()[0]
    print(f"      {n:,} lab results")


def extract_interventions(con):
    print("[4/5] Extracting vasopressors and ventilation ...")
    drug_pat = "|".join(VASOPRESSOR_DRUGS)
    try:
        con.execute(f"""
        CREATE OR REPLACE TABLE ev_vaso AS
        SELECT co.stay_id, 'vasopressor' AS feature, 1.0 AS value,
               d.infusionoffset / 60.0 AS hours_in
        FROM {eicu_table('infusionDrug')} d
        JOIN cohort co ON d.patientunitstayid = co.stay_id
        WHERE regexp_matches(lower(d.drugname), '{drug_pat}')
          AND d.infusionoffset >= 0 AND d.infusionoffset < {C.OBS_WINDOW_H*60}
        """)
    except Exception as e:
        print(f"      [WARN] infusionDrug unavailable ({e}); vasopressor "
              f"channel will be all-zero for eICU rows")
        con.execute("CREATE OR REPLACE TABLE ev_vaso AS "
                    "SELECT NULL::BIGINT stay_id, ''::VARCHAR feature, "
                    "NULL::DOUBLE value, NULL::DOUBLE hours_in WHERE FALSE")

    try:
        con.execute(f"""
        CREATE OR REPLACE TABLE ev_vent AS
        SELECT co.stay_id, 'ventilation' AS feature, 1.0 AS value,
               t.treatmentoffset / 60.0 AS hours_in
        FROM {eicu_table('treatment')} t
        JOIN cohort co ON t.patientunitstayid = co.stay_id
        WHERE lower(t.treatmentstring) LIKE '%ventilat%'
          AND t.treatmentoffset >= 0 AND t.treatmentoffset < {C.OBS_WINDOW_H*60}
        """)
    except Exception as e:
        print(f"      [WARN] treatment table unavailable ({e}); ventilation "
              f"channel will be all-zero for eICU rows")
        con.execute("CREATE OR REPLACE TABLE ev_vent AS "
                    "SELECT NULL::BIGINT stay_id, ''::VARCHAR feature, "
                    "NULL::DOUBLE value, NULL::DOUBLE hours_in WHERE FALSE")


def extract_urine(con):
    try:
        con.execute(f"""
        CREATE OR REPLACE TABLE ev_urine AS
        SELECT co.stay_id, 'urine' AS feature, io.outputtotal AS value,
               io.intakeoutputoffset / 60.0 AS hours_in
        FROM {eicu_table('intakeOutput')} io
        JOIN cohort co ON io.patientunitstayid = co.stay_id
        WHERE lower(io.celllabel) LIKE '%urine%'
          AND io.outputtotal IS NOT NULL AND io.outputtotal BETWEEN 0 AND 5000
          AND io.intakeoutputoffset >= 0
          AND io.intakeoutputoffset < {C.OBS_WINDOW_H*60}
        """)
    except Exception as e:
        print(f"      [WARN] intakeOutput unavailable ({e}); urine channel "
              f"will be all-zero for eICU rows")
        con.execute("CREATE OR REPLACE TABLE ev_urine AS "
                    "SELECT NULL::BIGINT stay_id, ''::VARCHAR feature, "
                    "NULL::DOUBLE value, NULL::DOUBLE hours_in WHERE FALSE")


def extract_apache_gcs(con, cohort):
    """GCS is only available as ONE worst-in-window value per stay (from
    apacheApsVar), not hourly -- so it goes into the static/tabular table,
    not the time series. This is disclosed in the module docstring."""
    try:
        gcs = con.execute(f"""
            SELECT co.stay_id, a.eyes, a.motor, a.verbal
            FROM {eicu_table('apacheApsVar')} a
            JOIN cohort co ON a.patientunitstayid = co.stay_id
        """).df()
        return gcs
    except Exception as e:
        print(f"      [WARN] apacheApsVar unavailable ({e}); GCS omitted")
        return pd.DataFrame({"stay_id": cohort.stay_id, "eyes": np.nan,
                             "motor": np.nan, "verbal": np.nan})


def build_timeseries(con, cohort):
    print("[5/5] Binning into hourly time series ...")
    con.execute("""
        CREATE OR REPLACE TABLE events AS
            SELECT * FROM ev_vital
            UNION ALL SELECT * FROM ev_lab
            UNION ALL SELECT * FROM ev_urine WHERE feature IS NOT NULL AND feature <> ''
            UNION ALL SELECT * FROM ev_vaso WHERE feature IS NOT NULL AND feature <> ''
            UNION ALL SELECT * FROM ev_vent WHERE feature IS NOT NULL AND feature <> ''
    """)
    hourly = con.execute(f"""
        SELECT stay_id, feature, CAST(FLOOR(hours_in) AS INTEGER) AS hr,
               CASE WHEN feature = 'urine' THEN SUM(value)
                    WHEN feature IN ('vasopressor','ventilation') THEN MAX(value)
                    ELSE AVG(value) END AS value
        FROM events
        WHERE hours_in >= 0 AND hours_in < {C.OBS_WINDOW_H}
        GROUP BY stay_id, feature, CAST(FLOOR(hours_in) AS INTEGER)
    """).df()

    stay_ids = cohort.stay_id.to_numpy()
    stay_idx = {s: i for i, s in enumerate(stay_ids)}
    # Use the SAME feature list/order as MIMIC's TS_FEATURES, so the resulting
    # tensor is directly concatenable with the MIMIC one by combine_datasets.py.
    feat_idx = {f: j for j, f in enumerate(C.TS_FEATURES)}

    N, T, F = len(stay_ids), C.OBS_WINDOW_H, len(C.TS_FEATURES)
    X = np.full((N, T, F), np.nan, dtype=np.float32)

    hourly = hourly[hourly.feature.isin(feat_idx)]
    rows = hourly.stay_id.map(stay_idx).to_numpy()
    cols = hourly.feature.map(feat_idx).to_numpy()
    hrs = hourly.hr.to_numpy()
    keep = ~pd.isna(rows)
    X[rows[keep].astype(int), hrs[keep], cols[keep]] = hourly.value.to_numpy()[keep]

    mask = (~np.isnan(X)).astype(np.float32)
    for t in range(1, T):
        prev = X[:, t - 1, :]
        cur = X[:, t, :]
        X[:, t, :] = np.where(np.isnan(cur), prev, cur)
    for name in ("vasopressor", "ventilation"):
        if name in feat_idx:
            j = feat_idx[name]
            X[:, :, j] = np.nan_to_num(X[:, :, j], nan=0.0)
            X[:, :, j] = np.maximum.accumulate(X[:, :, j], axis=1)
            mask[:, :, j] = 1.0

    gcs = extract_apache_gcs(con, cohort)
    gcs = gcs.set_index("stay_id").reindex(stay_ids).reset_index()
    static = np.stack([
        cohort.age.fillna(cohort.age.median()).to_numpy(dtype=np.float32),
        (cohort.gender == "Male").to_numpy(dtype=np.float32),
        np.ones(N, dtype=np.float32),   # admission_emergency (unknown -> 1)
        gcs.eyes.to_numpy(dtype=np.float32),
        gcs.motor.to_numpy(dtype=np.float32),
        gcs.verbal.to_numpy(dtype=np.float32),
    ], axis=1)

    y = cohort.label.to_numpy(dtype=np.float32)
    np.savez_compressed(
        EICU_OUT_DIR / "timeseries.npz", X=X, mask=mask, static=static, y=y,
        stay_id=stay_ids, subject_id=cohort.subject_id.to_numpy(),
        feature_names=np.array(C.TS_FEATURES),
        static_names=np.array(["age", "gender_m", "admission_emergency",
                               "gcs_eyes", "gcs_motor", "gcs_verbal"]),
    )
    print(f"      time series tensor: {X.shape} -> "
          f"{EICU_OUT_DIR / 'timeseries.npz'}")

    # aggregated (tabular) features, matching MIMIC's min/max/mean/count shape
    agg = con.execute("""
        SELECT stay_id, feature, MIN(value) v_min, MAX(value) v_max,
               AVG(value) v_mean, COUNT(*) v_count
        FROM events GROUP BY stay_id, feature
    """).df()
    wide = agg.pivot(index="stay_id", columns="feature",
                     values=["v_min", "v_max", "v_mean", "v_count"])
    wide.columns = [f"{b}_{a}" for a, b in wide.columns]
    wide = wide.reset_index()
    out = cohort[["stay_id", "subject_id", "age", "gender",
                 "admission_type", "label"]].merge(wide, on="stay_id", how="left")
    out["gender_m"] = (out.gender == "Male").astype(float)
    out["admission_emergency"] = 1.0
    out = out.merge(gcs.rename(columns={"eyes": "gcs_eyes", "motor": "gcs_motor",
                                        "verbal": "gcs_verbal"}),
                    on="stay_id", how="left")
    out = out.drop(columns=["gender", "admission_type"])
    out.to_parquet(EICU_OUT_DIR / "features_agg.parquet", index=False)
    print(f"      aggregated features: {out.shape} -> "
          f"{EICU_OUT_DIR / 'features_agg.parquet'}")


def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("PRAGMA memory_limit='16GB'")

    cohort = build_cohort(con)
    cohort.to_parquet(EICU_OUT_DIR / "cohort.parquet", index=False)

    extract_vitals(con)
    extract_labs(con)
    extract_interventions(con)
    extract_urine(con)
    build_timeseries(con, cohort)

    print(f"\nDone. eICU artifacts in {EICU_OUT_DIR}")
    print("Next: python combine_datasets.py")


if __name__ == "__main__":
    main()
