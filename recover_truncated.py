"""
Salvage a usable cohort from a TRUNCATED chartevents file.

Only run this if the complete file is genuinely unavailable. It is a mitigation,
not a fix, and it must be disclosed in the paper (wording is printed at the end).

The idea
--------
chartevents is written ordered by subject_id. If the file was cut off partway,
then every subject below the cut has COMPLETE records and every subject above it
has NONE. Restricting the cohort to subjects below the cut therefore yields a
smaller but internally complete dataset -- not a dataset with 59% of each
patient's vitals missing, which would be unusable.

Because MIMIC-IV subject_ids are randomly assigned during de-identification, a
subject_id prefix behaves like a random sample of patients. That is what makes
this defensible rather than a biased convenience sample.

This script VERIFIES the ordering assumption rather than trusting it, and falls
back to a per-stay completeness filter (with a bias warning) if it fails.

Run:  python recover_truncated.py
Out:  artifacts/recovery.json   (read automatically by preprocess.py)
"""
import json
import sys

import duckdb

import config as C


def reader(ignore_errors=True):
    return C.csv_reader("icu", "chartevents",
                        types={"charttime": "TIMESTAMP", "valuenum": "DOUBLE"},
                        ignore_errors=ignore_errors)


def check_sorted(con):
    """Count places where subject_id decreases. 0 => file is ordered."""
    print("  Verifying chartevents is ordered by subject_id ...")
    print("  (full scan of a ~13 GB file -- expect a few minutes)")
    q = f"""
        SELECT COUNT(*) FROM (
            SELECT subject_id, LAG(subject_id) OVER () AS prev
            FROM {reader()}
        ) WHERE prev IS NOT NULL AND subject_id < prev
    """
    inversions = con.execute(q).fetchone()[0]
    print(f"  order violations: {inversions:,}")
    return inversions == 0


def main():
    path = C.find_table("icu", "chartevents")
    if path is None:
        print("chartevents not found; check config.MIMIC_ROOT")
        sys.exit(1)
    print(f"Recovering from: {path}\n")

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={C.DUCKDB_THREADS}")
    con.execute(f"PRAGMA memory_limit='{C.DUCKDB_MEMORY}'")

    is_sorted = check_sorted(con)

    stats = con.execute(f"""
        SELECT MIN(subject_id), MAX(subject_id), COUNT(*), COUNT(DISTINCT subject_id)
        FROM {reader()}
    """).fetchone()
    lo, hi, n_rows, n_subj = stats
    print(f"\n  rows readable      : {n_rows:,}")
    print(f"  distinct subjects  : {n_subj:,}")
    print(f"  subject_id range   : {lo:,} .. {hi:,}")

    if is_sorted:
        # The highest subject present is the one that was cut mid-stream, so its
        # records are incomplete. Drop it, and keep everything strictly below.
        cutoff = con.execute(f"""
            SELECT MAX(subject_id) FROM {reader()} WHERE subject_id < {hi}
        """).fetchone()[0]
        method = "subject_prefix"
        print(f"\n  ORDERED confirmed. Safe cutoff: subject_id <= {cutoff:,}")
        print(f"  (subject {hi:,} dropped -- it was cut mid-write)")
    else:
        cutoff = None
        method = "per_stay_completeness"
        print("\n  [WARN] chartevents is NOT ordered by subject_id.")
        print("  Falling back to a per-stay completeness filter, which keeps only")
        print("  stays with enough charted vitals. This IS selection-biased --")
        print("  better-monitored (typically sicker) stays are retained.")

    rec = {"method": method, "max_safe_subject_id": cutoff,
           "chartevents_truncated": True, "rows_readable": n_rows,
           "distinct_subjects": n_subj, "verified_sorted": is_sorted}
    (C.OUT_DIR).mkdir(parents=True, exist_ok=True)
    (C.OUT_DIR / "recovery.json").write_text(json.dumps(rec, indent=2))

    # estimate the surviving cohort using the same filters preprocess applies
    est = con.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT i.subject_id, i.hadm_id,
                   ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime) rn,
                   i.los
            FROM {C.csv_reader('icu', 'icustays')} i
        ) s
        WHERE rn = 1 AND los * 24 >= {C.MIN_LOS_H}
          {f'AND subject_id <= {cutoff}' if cutoff else ''}
    """).fetchone()[0]

    print(f"\n  Estimated surviving cohort: ~{est:,} ICU stays")
    print(f"  (was ~67,000 with the complete file)")
    print(f"\n  Saved -> {C.OUT_DIR / 'recovery.json'}")
    print("  preprocess.py will pick this up automatically.\n")

    print("=" * 70)
    print("DISCLOSE THIS IN THE PAPER. Suggested wording:")
    print("=" * 70)
    print(f"""
  "Due to an incomplete local copy of the chartevents table, the analysis
   cohort was restricted to patients with subject_id <= {cutoff:,}, yielding
   {est:,} ICU stays. Because MIMIC-IV subject identifiers are randomly
   assigned during de-identification, this restriction is independent of
   patient characteristics and is equivalent to a random subsample of the
   full cohort."
""" if cutoff else """
  Report that the cohort was restricted by data completeness, and explicitly
  acknowledge the resulting selection bias toward more heavily monitored
  patients. This is a real limitation, not a formality.
""")
    print("=" * 70)
    print("\nSanity check to run afterwards: compare Table 1 (age, sex, mortality)")
    print("against published full-cohort MIMIC-IV figures. Mortality should still")
    print("land near 10-13%. If it does not, the subsample is not random and the")
    print("claim above does not hold.")


if __name__ == "__main__":
    main()
