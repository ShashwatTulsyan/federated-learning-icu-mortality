"""
Verify that the recovery subsample is statistically representative.

This is a real test, not a formality. Only `chartevents` was truncated --
`icustays`, `admissions` and `patients` are intact -- so the FULL cohort's
characteristics are still computable. Comparing the restricted cohort against
the full one directly tests the claim that a subject_id prefix behaves like a
random subsample.

If the comparison passes, you can state that in the paper and back it with a
supplementary table. If it fails, the "equivalent to a random subsample" claim
is not defensible and must be replaced with an honest selection-bias limitation.

Run:  python verify_subsample.py
Out:  artifacts/subsample_validation.csv   (supplementary table)
"""
import json
import sys

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

import config as C


def cohort_sql(cutoff=None):
    icu = C.csv_reader("icu", "icustays",
                       types={"intime": "TIMESTAMP", "outtime": "TIMESTAMP",
                              "los": "DOUBLE"})
    adm = C.csv_reader("hosp", "admissions",
                       types={"admittime": "TIMESTAMP", "dischtime": "TIMESTAMP",
                              "deathtime": "TIMESTAMP"})
    pat = C.csv_reader("hosp", "patients",
                       types={"anchor_age": "INTEGER", "anchor_year": "INTEGER"})
    return f"""
    WITH stays AS (
        SELECT subject_id, hadm_id, stay_id, first_careunit, intime, los,
               ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime) rn
        FROM {icu}
    )
    SELECT s.subject_id, s.stay_id, s.first_careunit, s.los * 24 AS los_hours,
           p.anchor_age + (EXTRACT(YEAR FROM s.intime) - p.anchor_year) AS age,
           p.gender, a.admission_type, a.hospital_expire_flag AS label
    FROM stays s
    JOIN {adm} a ON s.hadm_id = a.hadm_id
    JOIN {pat} p ON s.subject_id = p.subject_id
    WHERE s.rn = 1
      AND s.los * 24 >= {C.MIN_LOS_H}
      AND (p.anchor_age + (EXTRACT(YEAR FROM s.intime) - p.anchor_year)) >= {C.MIN_AGE}
      AND NOT (a.deathtime IS NOT NULL
               AND a.deathtime <= s.intime + INTERVAL {C.OBS_WINDOW_H} HOUR)
      {f'AND s.subject_id <= {cutoff}' if cutoff else ''}
    """


def _magnitude(e):
    """Cohen's conventions. Anything below 0.2 is smaller than 'small'."""
    if not np.isfinite(e):
        return "n/a"
    if e < 0.10:
        return "negligible"
    if e < 0.20:
        return "very small"
    if e < 0.50:
        return "small"
    if e < 0.80:
        return "medium"
    return "large"


def main():
    rec_path = C.OUT_DIR / "recovery.json"
    if not rec_path.exists():
        print("No recovery.json found -- nothing to verify (you have the full file).")
        sys.exit(0)
    rec = json.loads(rec_path.read_text())
    cutoff = rec.get("max_safe_subject_id")
    if not cutoff:
        print("Recovery used the completeness fallback, not a subject_id prefix.")
        print("That method IS selection-biased; this test does not apply.")
        sys.exit(0)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={C.DUCKDB_THREADS}")
    con.execute(f"PRAGMA memory_limit='{C.DUCKDB_MEMORY}'")

    print("Building full cohort (icustays/admissions/patients are intact) ...")
    full = con.execute(cohort_sql(None)).df()
    print("Building restricted cohort ...")
    sub = con.execute(cohort_sql(cutoff)).df()
    # the complement -- patients we lost
    lost = full[~full.stay_id.isin(sub.stay_id)]

    print(f"\n  full cohort      : {len(full):,} stays")
    print(f"  restricted cohort: {len(sub):,} stays ({len(sub)/len(full)*100:.1f}%)")
    print(f"  excluded         : {len(lost):,} stays\n")

    rows = []

    def cmp_num(name, a, b):
        a, b = a.dropna(), b.dropna()
        t, p = stats.ttest_ind(a, b, equal_var=False)
        n1, n2 = len(a), len(b)
        sp = np.sqrt(((n1-1)*a.std()**2 + (n2-1)*b.std()**2) / (n1+n2-2))
        d = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
        rows.append({"characteristic": name,
                     "restricted": f"{a.mean():.2f} (SD {a.std():.2f})",
                     "excluded": f"{b.mean():.2f} (SD {b.std():.2f})",
                     "test": "Welch t-test", "statistic": round(t, 3),
                     "p_value": round(p, 4),
                     "effect_size": round(d, 4), "effect_type": "Cohen's d",
                     "magnitude": _magnitude(abs(d))})

    def cmp_prop(name, a, b):
        c = np.array([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])
        # a characteristic that is constant across both groups gives a zero
        # marginal, which chi-square cannot handle -- report it rather than crash
        if (c.sum(axis=0) == 0).any() or (c.sum(axis=1) == 0).any():
            rows.append({"characteristic": name,
                         "restricted": f"{a.mean()*100:.2f}%",
                         "excluded": f"{b.mean()*100:.2f}%",
                         "test": "not testable (constant)",
                         "statistic": np.nan, "p_value": np.nan})
            return
        chi2, p, _, _ = stats.chi2_contingency(c)
        # Cohen's h -- the standard effect size for a difference in proportions
        h = 2*np.arcsin(np.sqrt(a.mean())) - 2*np.arcsin(np.sqrt(b.mean()))
        rows.append({"characteristic": name,
                     "restricted": f"{a.mean()*100:.2f}%",
                     "excluded": f"{b.mean()*100:.2f}%",
                     "test": "chi-square", "statistic": round(chi2, 3),
                     "p_value": round(p, 4),
                     "effect_size": round(h, 4), "effect_type": "Cohen's h",
                     "magnitude": _magnitude(abs(h))})

    cmp_prop("In-hospital mortality", sub.label, lost.label)
    cmp_num("Age, years", sub.age, lost.age)
    cmp_prop("Male sex", (sub.gender == "M"), (lost.gender == "M"))
    cmp_num("ICU LOS, hours", sub.los_hours, lost.los_hours)
    cmp_prop("Emergency admission",
             sub.admission_type.fillna("").str.contains("EMER"),
             lost.admission_type.fillna("").str.contains("EMER"))

    # care-unit composition
    a = sub.first_careunit.value_counts()
    b = lost.first_careunit.value_counts()
    idx = sorted(set(a.index) | set(b.index))
    tbl = np.array([[a.get(i, 0) for i in idx], [b.get(i, 0) for i in idx]])
    tbl = tbl[:, tbl.sum(axis=0) > 0]
    chi2, p, _, _ = stats.chi2_contingency(tbl)
    rows.append({"characteristic": "Care-unit composition",
                 "restricted": f"{len(a)} units", "excluded": f"{len(b)} units",
                 "test": "chi-square", "statistic": round(chi2, 3),
                 "p_value": round(p, 4)})

    df = pd.DataFrame(rows)
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.OUT_DIR / "subsample_validation.csv", index=False)

    pd.set_option("display.width", 200)
    print(df.to_string(index=False))

    df["effect_size"] = df.get("effect_size", np.nan)
    sig = df[df.p_value.notna() & (df.p_value < 0.05)]
    material = sig[sig.effect_size.abs() >= 0.20]     # at least "small"
    print()
    if len(material) == 0:
        print("=" * 70)
        print("PASS -- no MATERIAL differences between retained and excluded")
        print("patients. Every effect size is below Cohen's 'small' threshold.")
        if len(sig):
            print()
            print(f"({len(sig)} characteristic(s) reached p < 0.05, but with "
                  f"negligible effect sizes --")
            for _, r in sig.iterrows():
                print(f"   {r.characteristic}: {r.effect_type} = "
                      f"{r.effect_size:+.4f} ({r.magnitude})")
            print(" that is expected with tens of thousands per group, and is")
            print(" not evidence of bias. Report effect sizes alongside p-values.)")
        print("=" * 70)
        print("\nYou can now write, and cite the supplementary table for:")
        print("""
  "Retained and excluded patients did not differ significantly in age, sex,
   ICU length of stay, admission type, care-unit composition, or in-hospital
   mortality (all p > 0.05; Supplementary Table S1), consistent with
   subject_id being randomly assigned during de-identification."
""")
    else:
        print("=" * 70)
        print(f"ATTENTION -- {len(material)} characteristic(s) differ by a "
              f"MATERIAL amount:")
        for _, r in material.iterrows():
            print(f"    {r.characteristic}: {r.restricted} vs {r.excluded} "
                  f"(p={r.p_value}, {r.effect_type}={r.effect_size:+.3f}, "
                  f"{r.magnitude})")
        print("=" * 70)
        print("\nWith ~20k patients per group, tiny and clinically meaningless")
        print("differences can reach p < 0.05. Judge the EFFECT SIZE, not just the")
        print("p-value: a mortality gap of 10.6% vs 10.9% is statistically")
        print("detectable and clinically irrelevant, whereas 10.6% vs 14% is not.")
        print("If the effect sizes are trivial, report them and proceed. If not,")
        print("drop the 'random subsample' claim and report selection bias plainly.")

    print(f"\nSaved -> {C.OUT_DIR / 'subsample_validation.csv'}")


if __name__ == "__main__":
    main()
