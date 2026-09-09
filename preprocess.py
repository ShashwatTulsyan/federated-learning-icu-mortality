"""
Step 1: Cohort extraction + feature engineering from raw MIMIC-IV v3.1 CSVs.

Uses DuckDB because chartevents.csv.gz is ~300M rows / tens of GB uncompressed.
DuckDB streams it out-of-core and pushes the itemid filter down, so this runs in
a few GB of RAM instead of blowing up pandas.

Run:  python preprocess.py
Out:  artifacts/cohort.parquet, artifacts/timeseries.npz, artifacts/features_agg.parquet
"""
import json

import numpy as np
import pandas as pd
import duckdb

import config as C

# If recover_truncated.py has been run, apply its cohort restriction and allow
# the damaged file to be read past its final broken line.
RECOVERY = {}
_rec_path = C.OUT_DIR / "recovery.json"
if _rec_path.exists():
    RECOVERY = json.loads(_rec_path.read_text())

    # If chartevents has since been replaced with a COMPLETE copy, this file
    # would silently keep the cohort restricted to a third of its size. Check
    # the file's last row: a complete CSV ends with a full-width row.
    def _still_truncated():
        import gzip
        path = C.find_table("icu", "chartevents")
        if path is None:
            return True
        try:
            if path.suffix == ".gz":
                with gzip.open(path, "rt", errors="replace") as f:
                    hdr = f.readline().rstrip("\r\n").split(",")
                    last = None
                    for line in f:
                        last = line
            else:
                with open(path, "r", errors="replace") as f:
                    hdr = f.readline().rstrip("\r\n").split(",")
                size = path.stat().st_size
                with open(path, "rb") as f:
                    f.seek(max(0, size - 65536))
                    tail = f.read().decode("utf-8", errors="replace").splitlines()
                last = tail[-1] if tail else ""
            return len((last or "").rstrip("\r\n").split(",")) != len(hdr)
        except Exception:
            return True

    if not _still_truncated():
        print("=" * 70)
        print("STOP -- chartevents now appears COMPLETE, but artifacts/recovery.json")
        print("still exists and would restrict the cohort to ~22,565 stays instead")
        print("of the full ~67,000.")
        print()
        print("If you have replaced chartevents with a complete copy, delete the")
        print("recovery file and re-run:")
        print(f"    del {(_rec_path).as_posix().replace('/', chr(92))}")
        print()
        print("If you intended to keep the restriction, ignore this and re-run;")
        print("set KEEP_RECOVERY=1 in your environment to silence the check.")
        print("=" * 70)
        import os, sys
        if not os.environ.get("KEEP_RECOVERY"):
            sys.exit(1)


# DuckDB infers column types from a sample of the first rows. Two MIMIC columns
# routinely break that:
#   - `deathtime` is NULL for the ~87% of admissions that survive, so a sample of
#     early rows can be entirely empty -> inferred VARCHAR -> timestamp
#     comparisons raise a BinderError.
#   - `valuenum` is NULL wherever the charted value is non-numeric, same problem.
# Pinning the types makes extraction independent of row ordering.
COLUMN_TYPES = {
    "admissions":   {"admittime": "TIMESTAMP", "dischtime": "TIMESTAMP",
                     "deathtime": "TIMESTAMP"},
    "icustays":     {"intime": "TIMESTAMP", "outtime": "TIMESTAMP", "los": "DOUBLE"},
    "chartevents":  {"charttime": "TIMESTAMP", "valuenum": "DOUBLE"},
    "labevents":    {"charttime": "TIMESTAMP", "valuenum": "DOUBLE"},
    "outputevents": {"charttime": "TIMESTAMP", "value": "DOUBLE"},
    "patients":     {"anchor_age": "INTEGER", "anchor_year": "INTEGER"},
}


def _csv(module, name):
    """Layout-agnostic DuckDB reader (.csv.gz, .csv, or nested folder)."""
    ignore = bool(RECOVERY.get("chartevents_truncated")) and name == "chartevents"
    return C.csv_reader(module, name, types=COLUMN_TYPES.get(name),
                        ignore_errors=ignore)


# ---------------------------------------------------------------------------
# 1. Cohort
# ---------------------------------------------------------------------------
def build_cohort(con):
    """Builds the cohort AND records the TRIPOD-style exclusion cascade."""
    print("[1/5] Building cohort ...")
    flow = []

    def _n(sql):
        return con.execute(sql).fetchone()[0]

    icu_csv = _csv('icu', 'icustays')
    adm_csv = _csv('hosp', 'admissions')
    pat_csv = _csv('hosp', 'patients')

    flow.append(("All ICU stays in icustays", _n(f"SELECT COUNT(*) FROM {icu_csv}")))
    con.execute(f"""
    CREATE OR REPLACE TABLE cohort AS
    WITH stays AS (
        SELECT
            i.subject_id, i.hadm_id, i.stay_id,
            i.first_careunit, i.intime, i.outtime, i.los,
            ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime) AS stay_rank
        FROM {_csv('icu', 'icustays')} i
    )
    SELECT
        s.subject_id, s.hadm_id, s.stay_id,
        s.first_careunit, s.intime, s.outtime,
        s.los * 24.0 AS los_hours,
        -- age at ICU admission (anchor_age is age in the shifted anchor_year)
        p.anchor_age + (EXTRACT(YEAR FROM s.intime) - p.anchor_year) AS age,
        p.gender,
        p.anchor_year_group,
        a.admission_type,
        a.deathtime,
        a.hospital_expire_flag AS label
    FROM stays s
    JOIN {_csv('hosp', 'admissions')} a ON s.hadm_id = a.hadm_id
    JOIN {_csv('hosp', 'patients')}   p ON s.subject_id = p.subject_id
    WHERE 1=1
      {"AND s.stay_rank = 1" if C.FIRST_STAY_ONLY else ""}
      AND (p.anchor_age + (EXTRACT(YEAR FROM s.intime) - p.anchor_year)) >= {C.MIN_AGE}
      AND s.los * 24.0 >= {C.MIN_LOS_H}
      {f"AND s.subject_id <= {RECOVERY['max_safe_subject_id']}"
       if RECOVERY.get("max_safe_subject_id") else ""}
    """)

    if RECOVERY.get("max_safe_subject_id"):
        print(f"      [RECOVERY MODE] cohort restricted to subject_id <= "
              f"{RECOVERY['max_safe_subject_id']:,} because chartevents is "
              f"truncated. This must be disclosed in the paper.")

    _n_dropped = 0
    if C.DROP_DEATH_IN_WINDOW:
        _n_dropped = con.execute(f"""
            SELECT COUNT(*) FROM cohort WHERE deathtime IS NOT NULL
            AND deathtime <= intime + INTERVAL {C.OBS_WINDOW_H} HOUR
        """).fetchone()[0]
        # Drop stays where the patient died at or before the end of the input
        # window -- otherwise the "prediction" is retrospective.
        con.execute(f"""
        DELETE FROM cohort
        WHERE deathtime IS NOT NULL
          AND deathtime <= intime + INTERVAL {C.OBS_WINDOW_H} HOUR
        """)

    # cascade counts, applied in the same order as the WHERE clause above
    if C.FIRST_STAY_ONLY:
        flow.append(("First ICU stay per hospital admission", _n(f"""
            SELECT COUNT(*) FROM (SELECT ROW_NUMBER() OVER
            (PARTITION BY hadm_id ORDER BY intime) rn FROM {icu_csv}) WHERE rn=1""")))

    # Report the clinical filters and the data-availability restriction as
    # SEPARATE steps. Bundling them would misattribute an ~44k data-loss
    # exclusion to the age/LOS criteria in the TRIPOD diagram.
    cutoff = RECOVERY.get("max_safe_subject_id")
    n_clinical = _n(f"""
        SELECT COUNT(*) FROM (
            SELECT s.subject_id,
                   ROW_NUMBER() OVER (PARTITION BY s.hadm_id ORDER BY s.intime) rn,
                   s.los,
                   p.anchor_age + (EXTRACT(YEAR FROM s.intime) - p.anchor_year) age
            FROM {icu_csv} s JOIN {pat_csv} p ON s.subject_id = p.subject_id
        ) WHERE rn = 1 AND los * 24.0 >= {C.MIN_LOS_H} AND age >= {C.MIN_AGE}""")
    flow.append((f"Adults (age >= {C.MIN_AGE}) and ICU LOS >= {C.MIN_LOS_H}h",
                 n_clinical))
    if cutoff:
        flow.append((f"Restricted to subject_id <= {cutoff:,} "
                     f"(incomplete chartevents)",
                     _n("SELECT COUNT(*) FROM cohort") + _n_dropped))
    flow.append((f"Excluded death within {C.OBS_WINDOW_H}h observation window",
                 _n("SELECT COUNT(*) FROM cohort")))

    # ORDER BY is essential, not cosmetic. DuckDB scans in parallel and gives no
    # row-order guarantee, so two runs over identical data can return different
    # orderings. Partition files store row INDICES, so a reordered cohort
    # silently invalidates them: the indices still load, but they point at
    # different patients and the care-unit clients become random subsets.
    # Sorting by stay_id makes the ordering reproducible across runs.
    df = con.execute("SELECT * FROM cohort ORDER BY stay_id").df()
    flow_df = pd.DataFrame(flow, columns=["step", "n_stays"])
    flow_df["excluded"] = flow_df.n_stays.shift(1) - flow_df.n_stays
    flow_df.to_csv(C.OUT_DIR / "cohort_flow.csv", index=False)

    print("\n      --- Cohort flow (TRIPOD) ---")
    print(flow_df.to_string(index=False))
    print(f"\n      final cohort: {len(df):,} ICU stays | "
          f"{df.subject_id.nunique():,} unique patients | "
          f"mortality {df.label.mean()*100:.2f}% | "
          f"{df.first_careunit.nunique()} care units")
    return df


# ---------------------------------------------------------------------------
# 2. Events (vitals / labs / urine) restricted to the observation window
# ---------------------------------------------------------------------------
def _feature_case(feature_map):
    """Build a SQL CASE expression mapping itemid -> feature name."""
    branches = []
    for name, (itemids, _) in feature_map.items():
        ids = ",".join(str(i) for i in itemids)
        branches.append(f"WHEN itemid IN ({ids}) THEN '{name}'")
    return "CASE " + " ".join(branches) + " END"


def _range_filter(feature_map):
    """Build a SQL expression dropping physiologically impossible values."""
    branches = []
    for name, (_, (lo, hi)) in feature_map.items():
        branches.append(f"WHEN feature = '{name}' THEN value BETWEEN {lo} AND {hi}")
    return "CASE " + " ".join(branches) + " ELSE FALSE END"


def extract_events(con):
    print("[2/5] Extracting chartevents (this is the slow one) ...")
    chart_ids = sorted({i for ids, _ in C.CHART_FEATURES.values() for i in ids})
    con.execute(f"""
    CREATE OR REPLACE TABLE ev_chart AS
    SELECT
        c.stay_id,
        {_feature_case(C.CHART_FEATURES)} AS feature,
        c.valuenum AS value,
        date_diff('minute', co.intime, c.charttime) / 60.0 AS hours_in
    FROM {_csv('icu', 'chartevents')} c
    JOIN cohort co ON c.stay_id = co.stay_id
    WHERE c.itemid IN ({",".join(map(str, chart_ids))})
      AND c.valuenum IS NOT NULL
      AND c.charttime >= co.intime
      AND c.charttime <  co.intime + INTERVAL {C.OBS_WINDOW_H} HOUR
    """)
    con.execute(f"DELETE FROM ev_chart WHERE NOT ({_range_filter(C.CHART_FEATURES)})")
    # Fahrenheit -> Celsius, then fold into a single temp_c feature
    con.execute("""
        UPDATE ev_chart SET value = (value - 32.0) * 5.0/9.0, feature = 'temp_c'
        WHERE feature = 'temp_f'
    """)

    print("[3/5] Extracting labevents ...")
    lab_ids = sorted({i for ids, _ in C.LAB_FEATURES.values() for i in ids})
    con.execute(f"""
    CREATE OR REPLACE TABLE ev_lab AS
    SELECT
        co.stay_id,
        {_feature_case(C.LAB_FEATURES)} AS feature,
        l.valuenum AS value,
        date_diff('minute', co.intime, l.charttime) / 60.0 AS hours_in
    FROM {_csv('hosp', 'labevents')} l
    JOIN cohort co ON l.hadm_id = co.hadm_id
    WHERE l.itemid IN ({",".join(map(str, lab_ids))})
      AND l.valuenum IS NOT NULL
      AND l.hadm_id IS NOT NULL
      AND l.charttime >= co.intime
      AND l.charttime <  co.intime + INTERVAL {C.OBS_WINDOW_H} HOUR
    """)
    con.execute(f"DELETE FROM ev_lab WHERE NOT ({_range_filter(C.LAB_FEATURES)})")

    print("[4/5] Extracting urine output ...")
    con.execute(f"""
    CREATE OR REPLACE TABLE ev_urine AS
    SELECT
        o.stay_id,
        'urine' AS feature,
        o.value AS value,
        date_diff('minute', co.intime, o.charttime) / 60.0 AS hours_in
    FROM {_csv('icu', 'outputevents')} o
    JOIN cohort co ON o.stay_id = co.stay_id
    WHERE o.itemid IN ({",".join(map(str, C.URINE_ITEMIDS))})
      AND o.value IS NOT NULL
      AND o.value BETWEEN 0 AND 5000
      AND o.charttime >= co.intime
      AND o.charttime <  co.intime + INTERVAL {C.OBS_WINDOW_H} HOUR
    """)

    print("[4b/5] Extracting vasopressors and ventilation ...")
    vaso_ids = sorted({i for ids in C.VASOPRESSOR_ITEMIDS.values() for i in ids})
    try:
        # any pressor running during the hour -> 1. Rate units differ per drug,
        # so a presence flag is more robust than trying to harmonise doses.
        con.execute(f"""
        CREATE OR REPLACE TABLE ev_vaso AS
        SELECT co.stay_id, 'vasopressor' AS feature, 1.0 AS value,
               date_diff('minute', co.intime, iv.starttime) / 60.0 AS hours_in
        FROM {_csv('icu', 'inputevents')} iv
        JOIN cohort co ON iv.stay_id = co.stay_id
        WHERE iv.itemid IN ({",".join(map(str, vaso_ids))})
          AND iv.starttime >= co.intime
          AND iv.starttime <  co.intime + INTERVAL {C.OBS_WINDOW_H} HOUR
        """)
    except Exception as e:
        print(f"      [WARN] vasopressors unavailable ({e}); continuing without")
        con.execute("CREATE OR REPLACE TABLE ev_vaso AS "
                    "SELECT NULL::BIGINT stay_id, ''::VARCHAR feature, "
                    "NULL::DOUBLE value, NULL::DOUBLE hours_in WHERE FALSE")

    try:
        con.execute(f"""
        CREATE OR REPLACE TABLE ev_vent AS
        SELECT co.stay_id, 'ventilation' AS feature, 1.0 AS value,
               date_diff('minute', co.intime, pe.starttime) / 60.0 AS hours_in
        FROM {_csv('icu', 'procedureevents')} pe
        JOIN cohort co ON pe.stay_id = co.stay_id
        WHERE pe.itemid IN ({",".join(map(str, C.VENT_ITEMIDS))})
          AND pe.starttime >= co.intime
          AND pe.starttime <  co.intime + INTERVAL {C.OBS_WINDOW_H} HOUR
        """)
    except Exception as e:
        print(f"      [WARN] ventilation unavailable ({e}); continuing without")
        con.execute("CREATE OR REPLACE TABLE ev_vent AS "
                    "SELECT NULL::BIGINT stay_id, ''::VARCHAR feature, "
                    "NULL::DOUBLE value, NULL::DOUBLE hours_in WHERE FALSE")

    con.execute("""
    CREATE OR REPLACE TABLE events AS
        SELECT * FROM ev_chart WHERE feature IS NOT NULL
        UNION ALL SELECT * FROM ev_lab WHERE feature IS NOT NULL
        UNION ALL SELECT * FROM ev_urine
        UNION ALL SELECT * FROM ev_vaso WHERE feature IS NOT NULL AND feature <> ''
        UNION ALL SELECT * FROM ev_vent WHERE feature IS NOT NULL AND feature <> ''
    """)
    n = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    print(f"      {n:,} events in the {C.OBS_WINDOW_H}h window")
    _check_coverage(con)


def _check_coverage(con):
    """Integrity gate: catch silently-incomplete source files.

    The MIMIC-IV paper reports that 99% of ICU stays have at least one heart rate
    measurement. If our coverage is far below that, the source CSV was truncated
    or rows were skipped -- the pipeline would otherwise run to completion on
    partial data and produce quietly wrong results.
    """
    n_cohort = con.execute("SELECT COUNT(*) FROM cohort").fetchone()[0]
    stats = con.execute("""
        SELECT
          COUNT(DISTINCT CASE WHEN feature='heart_rate' THEN stay_id END) AS hr,
          COUNT(DISTINCT CASE WHEN feature='creatinine' THEN stay_id END) AS creat,
          COUNT(DISTINCT stay_id) AS any_ev
        FROM events
    """).fetchone()
    hr_cov = stats[0] / max(n_cohort, 1)
    creat_cov = stats[1] / max(n_cohort, 1)
    any_cov = stats[2] / max(n_cohort, 1)

    print(f"\n      --- Coverage check ---")
    print(f"      stays with any event : {any_cov*100:6.2f}%")
    print(f"      stays with heart rate: {hr_cov*100:6.2f}%   (expected ~99%)")
    print(f"      stays with creatinine: {creat_cov*100:6.2f}%   (expected ~90%+)")

    if hr_cov < 0.90:
        raise RuntimeError(
            f"\nHeart-rate coverage is only {hr_cov*100:.1f}% of the cohort, but "
            f"MIMIC-IV should be ~99%.\nThis almost always means chartevents.csv "
            f"is truncated or rows were skipped during reading.\n"
            f"Run `python diagnose_csv.py` to check file integrity.\n"
            f"Do not proceed -- results computed on partial data will look "
            f"plausible and be wrong.")
    if creat_cov < 0.60:
        print(f"      [WARN] creatinine coverage is low; check labevents.csv "
              f"integrity with diagnose_csv.py")


# ---------------------------------------------------------------------------
# 3. Hourly time-series tensor  (for the GRU model)
# ---------------------------------------------------------------------------
def build_timeseries(con, cohort):
    print("[5/5] Binning into hourly time series ...")
    # urine is a volume -> sum within the hour; everything else -> mean
    hourly = con.execute(f"""
        SELECT stay_id, feature,
               CAST(FLOOR(hours_in) AS INTEGER) AS hr,
               CASE WHEN feature = 'urine' THEN SUM(value)
                    WHEN feature IN ('vasopressor','ventilation') THEN MAX(value)
                    ELSE AVG(value) END AS value
        FROM events
        WHERE hours_in >= 0 AND hours_in < {C.OBS_WINDOW_H}
        GROUP BY stay_id, feature, CAST(FLOOR(hours_in) AS INTEGER)
    """).df()

    stay_ids = cohort.stay_id.to_numpy()
    stay_idx = {s: i for i, s in enumerate(stay_ids)}
    feat_idx = {f: j for j, f in enumerate(C.TS_FEATURES)}

    N, T, F = len(stay_ids), C.OBS_WINDOW_H, len(C.TS_FEATURES)
    X = np.full((N, T, F), np.nan, dtype=np.float32)

    hourly = hourly[hourly.feature.isin(feat_idx)]
    rows = hourly.stay_id.map(stay_idx).to_numpy()
    cols = hourly.feature.map(feat_idx).to_numpy()
    hrs = hourly.hr.to_numpy()
    keep = ~pd.isna(rows)
    X[rows[keep].astype(int), hrs[keep], cols[keep]] = hourly.value.to_numpy()[keep]

    # mask: 1 where a real measurement exists (before any imputation)
    mask = (~np.isnan(X)).astype(np.float32)

    # Intervention channels are different in kind from measurements: no record
    # means the patient was NOT on a pressor/ventilator, not that we failed to
    # observe it. Fill with 0 and forward-fill so "started at hour 3" persists.
    for name in ("vasopressor", "ventilation"):
        if name in feat_idx:
            j = feat_idx[name]
            X[:, :, j] = np.nan_to_num(X[:, :, j], nan=0.0)
            X[:, :, j] = np.maximum.accumulate(X[:, :, j], axis=1)
            mask[:, :, j] = 1.0

    # Causal forward-fill only (uses each patient's own past, never the future,
    # never other patients). Leading gaps are left as NaN on purpose.
    for t in range(1, T):
        prev = X[:, t - 1, :]
        cur = X[:, t, :]
        X[:, t, :] = np.where(np.isnan(cur), prev, cur)

    # NOTE: imputation of leading NaNs and z-score normalisation are deliberately
    # NOT done here. Computing medians/means over the whole dataset would
    #   (a) leak test-set statistics into training, and
    #   (b) violate the federated setting -- a global mean requires pooling every
    #       client's raw data, which is precisely what FL forbids.
    # Both are done per-client on the local TRAIN split in fed_train.py.

    # static features (raw; standardised per-client downstream)
    static = np.stack([
        cohort.age.to_numpy(dtype=np.float32),
        (cohort.gender == "M").to_numpy(dtype=np.float32),
        cohort.admission_type.fillna("").str.contains("EMER").to_numpy(dtype=np.float32),
    ], axis=1)

    y = cohort.label.to_numpy(dtype=np.float32)

    np.savez_compressed(
        C.TS_NPZ, X=X, mask=mask, static=static, y=y,
        stay_id=stay_ids,
        subject_id=cohort.subject_id.to_numpy(),   # needed for grouped splits
        feature_names=np.array(C.TS_FEATURES),
    )
    print(f"      time series tensor: {X.shape} (N, T, F) -> {C.TS_NPZ}")
    print(f"      (raw scale; imputation + normalisation happen per-client)")
    return X, mask, static, y


# ---------------------------------------------------------------------------
# 4. Aggregated features (for the MLP baseline / Track 1)
# ---------------------------------------------------------------------------
def build_comorbidities(con, cohort):
    """Charlson comorbidity flags + weighted index, per hospital admission.

    These describe chronic disease burden, which the 24h physiology window
    cannot see at all: a 65-year-old with metastatic cancer and one with none
    can present identically on day 1 and have very different outcomes.
    """
    if not getattr(C, "USE_COMORBIDITIES", False):
        return None

    source = getattr(C, "COMORBIDITY_SOURCE", "prior")
    if source == "both":
        # extract each variant separately and prefix the columns so downstream
        # code can select either without another pass over diagnoses_icd
        out = None
        for sub, tag in (("prior", "prior"), ("current", "curr")):
            C.COMORBIDITY_SOURCE = sub
            part = build_comorbidities(con, cohort)
            C.COMORBIDITY_SOURCE = "both"
            if part is None:
                continue
            part = part.rename(columns={c: c.replace("cci_", f"cci_{tag}_")
                                        for c in part.columns if c.startswith("cci_")})
            part = part.rename(columns={"charlson_score": f"charlson_{tag}",
                                        "n_diagnoses": f"n_diagnoses_{tag}"})
            out = part if out is None else out.merge(part, on="stay_id", how="outer")
        return out

    if source == "current":
        print("      [LEAKAGE WARNING] using the CURRENT admission's ICD codes.")
        print("      These are assigned at DISCHARGE and are not available at")
        print("      prediction time. Results will be inflated and unpublishable.")
        sql = f"""
            SELECT co.stay_id, d.icd_code, d.icd_version
            FROM {_csv('hosp', 'diagnoses_icd')} d
            JOIN cohort co ON d.hadm_id = co.hadm_id
        """
    else:
        # Only diagnoses from admissions that had already ENDED before this ICU
        # stay began. Those codes were finalised in the past, so they are
        # genuinely available at prediction time.
        sql = f"""
            SELECT co.stay_id, d.icd_code, d.icd_version
            FROM {_csv('hosp', 'diagnoses_icd')} d
            JOIN {_csv('hosp', 'admissions')} a ON d.hadm_id = a.hadm_id
            JOIN cohort co ON a.subject_id = co.subject_id
            WHERE a.dischtime < co.intime
              AND a.hadm_id <> co.hadm_id
        """
    try:
        dx = con.execute(sql).df()
    except Exception as e:
        print(f"      [WARN] diagnoses_icd unavailable ({e}); skipping comorbidities")
        return None

    dx["icd_code"] = dx.icd_code.astype(str).str.strip().str.upper()
    out = pd.DataFrame({"stay_id": cohort.stay_id})
    out["n_diagnoses"] = dx.groupby("stay_id").size().reindex(out.stay_id).fillna(0).values

    charlson = np.zeros(len(out), dtype=np.float32)
    pos = {s: i for i, s in enumerate(out.stay_id)}
    for cond, (w, icd9, icd10) in C.CHARLSON.items():
        m9 = (dx.icd_version == 9) & dx.icd_code.str.startswith(tuple(icd9))
        m10 = (dx.icd_version == 10) & dx.icd_code.str.startswith(tuple(icd10))
        hit = dx.loc[m9 | m10, "stay_id"].unique()
        col = np.zeros(len(out), dtype=np.float32)
        idx = [pos[s] for s in hit if s in pos]
        col[idx] = 1.0
        out[f"cci_{cond}"] = col
        charlson[idx] += w
    out["charlson_score"] = charlson

    with_hx = float((out.n_diagnoses > 0).mean())
    print(f"      comorbidities [{source}]: {len(C.CHARLSON)} conditions | "
          f"{with_hx*100:.1f}% of stays have any prior-coded history | "
          f"median Charlson {np.median(charlson):.1f}")
    if source == "prior":
        print(f"      (patients with no previous admission correctly get zeros --")
        print(f"       that is real missing information, not a data problem)")
    return out


def build_aggregate(con, cohort):
    print("      building aggregated features (MLP baseline) ...")
    agg = con.execute("""
        SELECT stay_id, feature,
               MIN(value) AS v_min, MAX(value) AS v_max,
               AVG(value) AS v_mean, COUNT(*) AS v_count
        FROM events
        GROUP BY stay_id, feature
    """).df()

    wide = agg.pivot(index="stay_id", columns="feature",
                     values=["v_min", "v_max", "v_mean", "v_count"])
    wide.columns = [f"{b}_{a}" for a, b in wide.columns]
    wide = wide.reset_index()

    out = cohort[["stay_id", "subject_id", "age", "gender", "admission_type", "label"]].merge(
        wide, on="stay_id", how="left")
    out["gender_m"] = (out.gender == "M").astype(float)
    out["admission_emergency"] = out.admission_type.fillna("").str.contains("EMER").astype(float)
    out = out.drop(columns=["gender", "admission_type"])

    com = build_comorbidities(con, cohort)
    if com is not None:
        out = out.merge(com, on="stay_id", how="left")

    # Left RAW on purpose -- imputation and standardisation are done per-client
    # on the local train split in fed_train.py (see note in build_timeseries).
    out.to_parquet(C.AGG_PQ, index=False)
    print(f"      aggregated features: {out.shape} -> {C.AGG_PQ} (raw scale)")


def main():
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={C.DUCKDB_THREADS}")
    con.execute(f"PRAGMA memory_limit='{C.DUCKDB_MEMORY}'")

    cohort = build_cohort(con)
    cohort.to_parquet(C.COHORT_PQ, index=False)

    extract_events(con)
    build_timeseries(con, cohort)
    build_aggregate(con, cohort)

    print("\nDone. Next: python partition.py")


if __name__ == "__main__":
    main()