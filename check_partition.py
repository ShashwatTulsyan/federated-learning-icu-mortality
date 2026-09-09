"""Diagnose whether the partition file actually matches the current cohort.

Symptom this catches: per-client mortality rates collapse toward the pooled
mean. The partition still loads and training still runs, but the "care units"
have become random subsets of patients, which destroys the non-IID premise the
whole study rests on.

Run:  python check_partition.py
"""
import json
import sys

import numpy as np
import pandas as pd

import config as C
from partition import cohort_fingerprint


def main():
    cohort = pd.read_parquet(C.COHORT_PQ).reset_index(drop=True)
    path = C.OUT_DIR / f"partition_{C.PARTITION_SCHEME}.json"
    if not path.exists():
        print(f"No partition file at {path}. Run partition.py.")
        sys.exit(1)

    raw = json.load(open(path))
    fp_now = cohort_fingerprint(cohort.stay_id.to_numpy())

    if "clients" in raw:
        fp_file, part = raw.get("__fingerprint__"), raw["clients"]
        print(f"  cohort fingerprint : {fp_now}")
        print(f"  partition built for: {fp_file}")
        print("  " + ("MATCH" if fp_file == fp_now else "*** MISMATCH ***"))
    else:
        part, fp_file = raw, None
        print("  partition file predates fingerprinting; checking by content\n")

    careunit = C.PARTITION_SCHEME == "careunit"
    if not careunit:
        print(f"\n  scheme = '{C.PARTITION_SCHEME}': clients are not care units,")
        print("  so per-unit membership is not checked. Reporting mortality")
        print("  spread instead -- for the IID control it should be NARROW.\n")

    print(f"\n{'client':<50}{'n_test':>8}{'events':>8}{'test %':>9}"
          f"{'unit %':>9}{'':>4}")
    print("-" * 90)
    bad = 0
    rates = []
    for name, sp in part.items():
        te = np.array(sp["test"], dtype=int)
        sub = cohort.loc[te]
        obs = sub.label.mean() * 100
        rates.append(obs)
        if careunit:
            unit = cohort[cohort.first_careunit == name].label.mean() * 100
            frac_right = float((sub.first_careunit == name).mean())
            flag = "" if frac_right > 0.99 else \
                   f"  <-- only {frac_right*100:.0f}% from this unit"
            if frac_right <= 0.99:
                bad += 1
        else:
            unit = float("nan")
            flag = ""
        print(f"{name:<50}{len(te):>8}{int(sub.label.sum()):>8}"
              f"{obs:>8.1f}%{unit:>8.2f}%{flag}")

    print()
    spread = max(rates) - min(rates)
    print(f"  mortality spread across clients: {min(rates):.1f}% - {max(rates):.1f}% "
          f"(range {spread:.1f} pp)")
    if not careunit:
        print("  For the 'shuffled' IID control a NARROW spread (a few pp) is")
        print("  correct -- it confirms heterogeneity was removed.")
        if fp_file is None or fp_file == fp_now:
            print("\n  PARTITION IS VALID (fingerprint matches the current cohort).")
            sys.exit(0)
        print("\n  FINGERPRINT MISMATCH -- re-run partition.py.")
        sys.exit(1)

    if bad == 0 and (fp_file is None or fp_file == fp_now):
        print("  PARTITION IS VALID -- every client's rows come from its own care unit.")
        sys.exit(0)

    print(f"  PARTITION IS STALE: {bad} client(s) contain rows from other units.")
    print("  The stored indices were built for a different cohort ordering.")
    print("\n  Fix, in order:")
    print("    python partition.py          # rebuild against the current cohort")
    print("    python check_partition.py    # confirm it now matches")
    print("    python run_experiments.py --jobs=8")
    print("    python fed_train.py")
    print("\n  Any results produced with this partition must be discarded: the")
    print("  clients were random subsets, not care units, so the non-IID premise")
    print("  did not hold.")
    sys.exit(1)


if __name__ == "__main__":
    main()