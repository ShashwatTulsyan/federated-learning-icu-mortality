"""
Combine MIMIC-IV and eICU-CRD into ONE federated dataset.

Client design: MIMIC-IV's 7 care units stay as separate clients (unchanged),
and each sufficiently large eICU hospital becomes ITS OWN client. This is what
converts "simulated" federation (care units within one hospital system) into a
genuine multi-institution federation -- eICU hospitals are real, separate
health systems, monitored independently by the Philips eICU program.

Because eICU rows are written into the SAME `first_careunit` column MIMIC uses
for its client key (values like "eICU_Hospital_264"), partition.py's existing
partition_careunit() function works on the combined cohort with NO changes.

Run:  python combine_datasets.py
Out:  <combined_dir>/cohort.parquet, timeseries.npz, features_agg.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

MIMIC_DIR = C.OUT_DIR
EICU_DIR = C.OUT_DIR.parent / "artifacts_eicu"
COMBINED_DIR = C.OUT_DIR.parent / "artifacts_combined"
COMBINED_DIR.mkdir(parents=True, exist_ok=True)

# eICU patientunitstayid and MIMIC-IV stay_id occupy overlapping integer
# ranges. Offset eICU's IDs well clear of MIMIC's so they can never collide
# once concatenated.
EICU_ID_OFFSET = 900_000_000


def main():
    for name, d in (("MIMIC", MIMIC_DIR), ("eICU", EICU_DIR)):
        if not (d / "timeseries.npz").exists():
            print(f"Missing {d / 'timeseries.npz'}.")
            if name == "MIMIC":
                print("Run preprocess.py first.")
            else:
                print("Run preprocess_eicu.py first.")
            sys.exit(1)

    print(f"Loading MIMIC-IV from {MIMIC_DIR} ...")
    dm = np.load(MIMIC_DIR / "timeseries.npz", allow_pickle=True)
    cm = pd.read_parquet(MIMIC_DIR / "cohort.parquet")
    fm = pd.read_parquet(MIMIC_DIR / "features_agg.parquet")

    print(f"Loading eICU-CRD from {EICU_DIR} ...")
    de = np.load(EICU_DIR / "timeseries.npz", allow_pickle=True)
    ce = pd.read_parquet(EICU_DIR / "cohort.parquet")
    fe = pd.read_parquet(EICU_DIR / "features_agg.parquet")

    # ---- verify the time-series feature axes are identical and alignable ---
    names_m = [str(x) for x in dm["feature_names"]]
    names_e = [str(x) for x in de["feature_names"]]
    if names_m != names_e:
        print("FATAL: feature_names differ between MIMIC and eICU tensors.")
        print(f"  MIMIC: {names_m}")
        print(f"  eICU : {names_e}")
        print("Both must come from the same config.TS_FEATURES list -- check")
        print("that neither pipeline was run with a stale config.py.")
        sys.exit(1)
    print(f"  {len(names_m)} time-series features match exactly -- safe to "
          f"concatenate")

    # ---- concatenate the time-series tensors --------------------------------
    X = np.concatenate([dm["X"], de["X"]], axis=0)
    mask = np.concatenate([dm["mask"], de["mask"]], axis=0)
    y = np.concatenate([dm["y"], de["y"]], axis=0)
    stay_id = np.concatenate([dm["stay_id"], de["stay_id"] + EICU_ID_OFFSET])
    subject_id_m = dm["subject_id"]
    # eICU's uniquepid is a string, MIMIC's subject_id is numeric -- keep both
    # as strings in the combined file so grouped splitting still works
    subject_id = np.concatenate([
        np.array([f"mimic_{s}" for s in subject_id_m]),
        np.array([f"eicu_{s}" for s in de["subject_id"]]),
    ])

    # eICU's static array has extra GCS columns MIMIC's doesn't (MIMIC's GCS
    # is already in the time series, not static). Pad MIMIC's static array
    # with NaN in those extra columns rather than discarding eICU's GCS.
    static_m = dm["static"]
    static_e = de["static"]
    n_static = max(static_m.shape[1], static_e.shape[1])
    pad_m = np.full((static_m.shape[0], n_static - static_m.shape[1]), np.nan,
                    dtype=np.float32)
    pad_e = np.full((static_e.shape[0], n_static - static_e.shape[1]), np.nan,
                    dtype=np.float32)
    static = np.concatenate([
        np.concatenate([static_m, pad_m], axis=1),
        np.concatenate([static_e, pad_e], axis=1),
    ], axis=0)

    np.savez_compressed(
        COMBINED_DIR / "timeseries.npz", X=X, mask=mask, static=static, y=y,
        stay_id=stay_id, subject_id=subject_id, feature_names=dm["feature_names"],
    )
    print(f"  combined tensor: {X.shape} -> {COMBINED_DIR / 'timeseries.npz'}")

    # ---- combine cohorts (for partitioning + Table 1) -----------------------
    keep_cols = ["stay_id", "subject_id", "first_careunit", "anchor_year_group",
                "age", "gender", "admission_type", "los_hours", "label"]
    cm2 = cm.copy()
    if "anchor_year_group" not in cm2.columns:
        cm2["anchor_year_group"] = "unknown"
    cm2["dataset"] = "mimic"
    cm2["subject_id"] = "mimic_" + cm2.subject_id.astype(str)

    ce2 = ce.rename(columns={"gender": "gender"}).copy()
    ce2["stay_id"] = ce2.stay_id + EICU_ID_OFFSET
    ce2["subject_id"] = "eicu_" + ce2.subject_id.astype(str)
    ce2["anchor_year_group"] = "eicu_2014-2015"      # eICU-CRD collection period
    ce2["dataset"] = "eicu"
    for col in keep_cols:
        if col not in ce2.columns:
            ce2[col] = np.nan

    combined = pd.concat(
        [cm2[keep_cols + ["dataset"]], ce2[keep_cols + ["dataset"]]],
        ignore_index=True)
    combined.to_parquet(COMBINED_DIR / "cohort.parquet", index=False)
    print(f"  combined cohort: {len(combined):,} stays "
          f"({len(cm2):,} MIMIC + {len(ce2):,} eICU) -> "
          f"{COMBINED_DIR / 'cohort.parquet'}")

    # ---- combine aggregate feature tables ------------------------------------
    fm2 = fm.copy(); fm2["stay_id"] = fm2.stay_id.astype(np.int64)
    fm2["subject_id"] = "mimic_" + fm2.subject_id.astype(str)
    fe2 = fe.copy(); fe2["stay_id"] = fe2.stay_id.astype(np.int64) + EICU_ID_OFFSET
    fe2["subject_id"] = "eicu_" + fe2.subject_id.astype(str)
    combined_feat = pd.concat([fm2, fe2], ignore_index=True, sort=False)
    combined_feat.to_parquet(COMBINED_DIR / "features_agg.parquet", index=False)
    print(f"  combined tabular features: {combined_feat.shape} -> "
          f"{COMBINED_DIR / 'features_agg.parquet'}")

    # ---- client summary -------------------------------------------------------
    print("\nClients after combining (before MIN_CLIENT_SIZE filtering):")
    counts = combined.groupby(["dataset", "first_careunit"]).agg(
        n=("stay_id", "size"), mortality=("label", "mean")).reset_index()
    counts = counts.sort_values("n", ascending=False)
    pd.set_option("display.width", 120)
    print(counts.head(20).to_string(index=False))
    n_eicu_clients = (counts[counts.dataset == "eicu"].n >=
                      C.MIN_CLIENT_SIZE).sum()
    n_mimic_clients = (counts[counts.dataset == "mimic"].n >=
                       C.MIN_CLIENT_SIZE).sum()
    print(f"\n  {n_mimic_clients} MIMIC-IV care-unit clients + "
          f"{n_eicu_clients} eICU hospital clients "
          f"(>= MIN_CLIENT_SIZE={C.MIN_CLIENT_SIZE}) once partitioned")

    print(f"\nDone. To use this combined dataset:")
    print(f"  1. In config.py, set:  OUT_DIR = Path(r'{COMBINED_DIR}')")
    print(f"  2. python partition.py")
    print(f"  3. python check_partition.py")
    print(f"  4. python run_experiments.py --jobs=8")


if __name__ == "__main__":
    main()
