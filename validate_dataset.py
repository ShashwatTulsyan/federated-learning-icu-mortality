"""
Step 0: PREFLIGHT CHECK -- run this before anything else.

Verifies, against YOUR actual MIMIC-IV download:
  - every required file exists and is readable
  - every column the pipeline references actually exists (by name)
  - every itemid the pipeline uses actually appears in d_items / d_labitems
  - the anchor_year_group values match what v3.1 is supposed to contain

Catches filename/column mismatches in ~30 seconds instead of 40 minutes into
preprocessing. Exits non-zero on any hard failure.

Run:  python validate_dataset.py
"""
import sys

import duckdb

import config as C

# (module, table) -> columns the pipeline actually reads.
# Files are located by config.find_table(), which handles .csv.gz, extracted
# .csv, and nested-folder layouts alike -- you do NOT need to reorganise your
# download to match any particular structure.
REQUIRED = {
    ("hosp", "patients"): [
        "subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"],
    ("hosp", "admissions"): [
        "subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
        "admission_type", "hospital_expire_flag"],
    ("hosp", "labevents"): [
        "subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
    ("hosp", "d_labitems"): ["itemid", "label"],   # loinc_code dropped in v3.x
    ("icu", "icustays"): [
        "subject_id", "hadm_id", "stay_id", "first_careunit", "last_careunit",
        "intime", "outtime", "los"],
    ("icu", "chartevents"): [
        "subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"],
    ("icu", "outputevents"): [
        "subject_id", "hadm_id", "stay_id", "charttime", "itemid", "value"],
    ("icu", "d_items"): ["itemid", "label", "linksto"],
}

OPTIONAL = {
    ("hosp", "diagnoses_icd"): ["subject_id", "hadm_id", "icd_code", "icd_version"],
    ("hosp", "prescriptions"): ["subject_id", "hadm_id", "drug", "starttime"],
    ("icu", "inputevents"): ["stay_id", "itemid", "starttime", "amount", "rate"],
    ("icu", "procedureevents"): ["stay_id", "itemid", "starttime"],
}

EXPECTED_YEAR_GROUPS = {
    "2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022"}


def header_of(con, module, name):
    """Read only the header row -- does not decompress the whole file."""
    q = f"SELECT * FROM {C.csv_reader(module, name)} LIMIT 0"
    return [d[0] for d in con.execute(q).description]


def _rel(path):
    try:
        return path.relative_to(C.MIMIC_ROOT)
    except Exception:
        return path


def check_files(con, spec, hard=True):
    ok = True
    for (module, name), cols in spec.items():
        path = C.find_table(module, name)
        if path is None:
            tag = "FAIL" if hard else "skip"
            print(f"  [{tag}] {module}/{name}: not found "
                  f"(looked for {name}.csv.gz, {name}.csv, and nested variants)")
            ok = ok and not hard
            continue
        try:
            have = header_of(con, module, name)
        except Exception as e:
            print(f"  [FAIL] {module}/{name}: cannot read -> {e}")
            ok = False
            continue
        missing = [c for c in cols if c not in have]
        if missing:
            print(f"  [FAIL] {module}/{name}: missing column(s) {missing}")
            print(f"         columns present: {have}")
            ok = False
        else:
            kind = "gz" if path.suffix == ".gz" else "csv"
            print(f"  [ ok ] {module}/{name:<16} {len(have):>2} cols  "
                  f"[{kind}]  {_rel(path)}")
    return ok


def check_itemids(con):
    ok = True
    chart_ids = sorted({i for ids, _ in C.CHART_FEATURES.values() for i in ids})
    lab_ids = sorted({i for ids, _ in C.LAB_FEATURES.values() for i in ids})

    vaso_ids = sorted({i for ids in getattr(C, "VASOPRESSOR_ITEMIDS", {}).values()
                       for i in ids})
    vent_ids = getattr(C, "VENT_ITEMIDS", [])

    for label, ids, (mod, dic) in [
        ("chartevents", chart_ids, ("icu", "d_items")),
        ("outputevents (urine)", C.URINE_ITEMIDS, ("icu", "d_items")),
        ("inputevents (vasopressors)", vaso_ids, ("icu", "d_items")),
        ("procedureevents (ventilation)", vent_ids, ("icu", "d_items")),
        ("labevents", lab_ids, ("hosp", "d_labitems")),
    ]:
        if not ids:
            continue
        found = con.execute(
            f"SELECT itemid FROM {C.csv_reader(mod, dic)} "
            f"WHERE itemid IN ({','.join(map(str, ids))})").df()["itemid"].tolist()
        missing = sorted(set(ids) - set(found))
        if missing:
            # urine itemids vary by version; treat as a warning not a failure
            # a wrong itemid here yields an EMPTY channel rather than an error,
            # so surface it loudly
            level = "WARN" if ("urine" in label or "vaso" in label
                               or "vent" in label) else "FAIL"
            print(f"  [{level}] {label}: {len(missing)} itemid(s) not in dictionary: {missing}")
            if level == "FAIL":
                ok = False
        else:
            print(f"  [ ok ] {label}: all {len(ids)} itemids present")
    return ok


def check_values(con):
    ok = True
    groups = set(con.execute(
        f"SELECT DISTINCT anchor_year_group FROM {C.csv_reader('hosp', 'patients')}"
    ).df()["anchor_year_group"].dropna().tolist())
    unexpected = groups - EXPECTED_YEAR_GROUPS
    if unexpected:
        print(f"  [WARN] unexpected anchor_year_group value(s): {unexpected}")
    if "2020 - 2022" not in groups:
        print(f"  [WARN] '2020 - 2022' bucket absent -- this looks like MIMIC-IV "
              f"< v3.0, not v3.1. Found: {sorted(groups)}")
    else:
        print(f"  [ ok ] anchor_year_group: {len(groups)} buckets incl. 2020-2022 (v3.x)")

    units = con.execute(
        f"SELECT first_careunit, COUNT(*) n FROM {C.csv_reader('icu', 'icustays')} "
        f"GROUP BY 1 ORDER BY n DESC").df()
    print(f"  [info] care units available for partitioning:")
    for _, r in units.iterrows():
        flag = "" if r.n >= C.MIN_CLIENT_SIZE else "  <-- below MIN_CLIENT_SIZE, will be dropped"
        print(f"         {r.first_careunit:<40} {int(r.n):>7,}{flag}")
    usable = (units.n >= C.MIN_CLIENT_SIZE).sum()
    if usable < 2:
        print(f"  [FAIL] only {usable} care unit(s) meet MIN_CLIENT_SIZE="
              f"{C.MIN_CLIENT_SIZE}; need >= 2 clients")
        ok = False

    n_stays = con.execute(
        f"SELECT COUNT(*) FROM {C.csv_reader('icu', 'icustays')}").fetchone()[0]
    print(f"  [info] total ICU stays in icustays: {n_stays:,} "
          f"(v3.1 reference: ~94,458)")
    return ok


def _diagnose():
    """Show what actually lives under MIMIC_ROOT so the fix is obvious."""
    print(f"\nWhat is actually under {C.MIMIC_ROOT}:")
    for module in ("hosp", "icu"):
        base = C.MIMIC_ROOT / module
        if not base.is_dir():
            print(f"  {module}/  -- MISSING. Is MIMIC_ROOT pointing one level too "
                  f"high or too low?")
            continue
        entries = sorted(base.iterdir())[:12]
        print(f"  {module}/  ({len(list(base.iterdir()))} entries)")
        for e in entries:
            kind = "DIR " if e.is_dir() else "file"
            print(f"      [{kind}] {e.name}")
        if len(list(base.iterdir())) > 12:
            print("      ...")
    print("\nAll of these layouts are supported automatically:")
    print("    hosp/admissions.csv.gz")
    print("    hosp/admissions.csv")
    print("    hosp/admissions.csv/admissions.csv     <- nested extraction")
    print("    hosp/admissions/admissions.csv")


def main():
    print(f"Validating MIMIC-IV at: {C.MIMIC_ROOT}\n")
    if not C.MIMIC_ROOT.exists():
        print(f"[FAIL] MIMIC_ROOT does not exist. Edit config.MIMIC_ROOT.")
        sys.exit(1)

    con = duckdb.connect()
    print("1. Required files and columns")
    ok = check_files(con, REQUIRED, hard=True)
    print("\n2. Optional files (only needed if you extend features)")
    check_files(con, OPTIONAL, hard=False)
    if not ok:
        print("\nVALIDATION FAILED at step 1 -- fix the [FAIL] items above.")
        print("Nothing further can be checked until every required file is found.")
        _diagnose()
        sys.exit(1)

    print("\n3. Feature itemids present in dictionaries")
    ok &= check_itemids(con)
    print("\n4. Value sanity / version check")
    ok &= check_values(con)

    print()
    if ok:
        print("VALIDATION PASSED -- safe to run: python preprocess.py")
        sys.exit(0)
    print("VALIDATION FAILED -- fix the [FAIL] items above before preprocessing.")
    sys.exit(1)


if __name__ == "__main__":
    main()
