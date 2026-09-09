"""
Integrity check for eICU-CRD source files -- catches gzip corruption, not just
text-level truncation (that's diagnose_csv.py, for MIMIC-IV).

Why a separate check is needed here: your chartevents.csv failure earlier was
a file DuckDB could still decompress, just with the final row cut short.
This vitalPeriodic failure is different -- "gzip stream data error" means the
COMPRESSED bytes themselves are broken partway through, which a tail-only
check can miss entirely (the corruption could be anywhere in the stream, and
gzip decompression is sequential -- you can't decode past a broken block no
matter where you start reading). This script instead attempts a full
sequential decompression of each file and reports exactly how far it got.

Run:  python diagnose_eicu.py
"""
import gzip
import sys
from pathlib import Path

import preprocess_eicu as PE

TABLES = ["patient", "vitalPeriodic", "vitalAperiodic", "lab", "apacheApsVar",
          "infusionDrug", "treatment", "intakeOutput", "hospital"]


def check(name):
    try:
        path = None
        for cand in [PE.EICU_ROOT / f"{name}.csv.gz", PE.EICU_ROOT / f"{name}.csv",
                    PE.EICU_ROOT / f"{name}.csv" / f"{name}.csv",
                    PE.EICU_ROOT / name.lower() / f"{name.lower()}.csv.gz",
                    PE.EICU_ROOT / f"{name.lower()}.csv.gz",
                    PE.EICU_ROOT / f"{name.lower()}.csv"]:
            if cand.is_file():
                path = cand
                break
    except Exception:
        path = None

    if path is None:
        print(f"  [ -- ] {name}: not found")
        return None

    size_gb = path.stat().st_size / 1024**3
    if path.suffix != ".gz":
        print(f"  [ ok ] {name}: {size_gb*1024:.1f} MB (uncompressed, gzip "
              f"check does not apply)")
        return True

    header = None
    n_rows = 0
    bad_row = None
    try:
        with gzip.open(path, "rt", errors="replace") as f:
            header = f.readline().rstrip("\n").split(",")
            n_cols = len(header)
            for line in f:
                n_rows += 1
                if line.rstrip("\n").count(",") + 1 != n_cols and line.strip():
                    bad_row = n_rows
                    # keep going -- a single malformed row can be a quoted
                    # comma inside a field, not real corruption; we only care
                    # if decompression itself later throws
    except (EOFError, OSError, gzip.BadGzipFile) as e:
        pct = "?" if n_rows == 0 else f"~row {n_rows:,}"
        print(f"  [FAIL] {name}: {size_gb:.2f} GB, CORRUPTED -- gzip stream "
              f"broke after {pct} rows")
        print(f"         {type(e).__name__}: {e}")
        print(f"         {path}")
        return False
    except Exception as e:
        print(f"  [FAIL] {name}: unexpected error -> {type(e).__name__}: {e}")
        return False

    print(f"  [ ok ] {name}: {size_gb:.2f} GB, {n_rows:,} rows decompressed "
          f"cleanly, {n_cols} columns")
    return True


def main():
    print(f"Checking eICU-CRD files under {PE.EICU_ROOT}\n")
    if not PE.EICU_ROOT.exists():
        print("EICU_ROOT does not exist -- check preprocess_eicu.py's path.")
        sys.exit(1)
    results = [check(t) for t in TABLES]
    bad = [t for t, r in zip(TABLES, results) if r is False]
    print()
    if not bad:
        print("All present files decompressed cleanly.")
        sys.exit(0)
    print(f"{len(bad)} file(s) are corrupted: {bad}")
    print("\nRe-download these specifically (resume-capable):")
    for t in bad:
        print(f"  wget -c https://physionet.org/files/eicu-crd/2.0/{t}.csv.gz "
              f"-P \"{PE.EICU_ROOT}\"")
    print("\nDo NOT try to work around this -- a corrupted gzip stream cannot")
    print("be partially salvaged the way a text-truncated file can; DuckDB")
    print("will simply refuse to read past the break point every time.")
    sys.exit(1)


if __name__ == "__main__":
    main()
