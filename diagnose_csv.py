"""
Detect truncated or corrupt MIMIC CSV files.

A CSV that was interrupted during extraction ends mid-field. DuckDB reports this
as "Expected Number of Columns: N Found: M" at some line number, which looks like
a parser problem but is actually a damaged file. Telling DuckDB to
`ignore_errors=true` in that situation makes the error disappear while silently
discarding every row after the break -- the worst possible outcome, because the
pipeline then runs to completion on a fraction of the data.

This script checks the end of each file directly, which is decisive.

Run:  python diagnose_csv.py
"""
import gzip
import sys

import config as C

# The DECISIVE check is structural: does the final row have the same number of
# fields as the header, and does the file end with a newline? A file cut off
# mid-write fails both.
#
# File size is only a weak secondary signal, so it is advisory (WARN) and applied
# ONLY to large tables. Applying a percentage threshold to a dictionary table
# measured in kilobytes produces meaningless failures -- an earlier version of
# this script did exactly that and flagged three perfectly good files.
SIZE_CHECK_MIN_GB = 1.0          # below this, size tells you nothing useful
APPROX_GB = {
    ("icu", "chartevents"): 32.0,
    ("hosp", "labevents"): 17.0,
    ("icu", "inputevents"): 2.4,
    ("hosp", "prescriptions"): 3.6,
}

TABLES = [
    ("hosp", "patients"), ("hosp", "admissions"), ("hosp", "labevents"),
    ("hosp", "d_labitems"), ("icu", "icustays"), ("icu", "chartevents"),
    ("icu", "outputevents"), ("icu", "d_items"),
]


def tail_lines(path, n_bytes=65536):
    """Last complete-ish chunk of the file, as text lines."""
    if path.suffix == ".gz":
        # gzip must be read sequentially; also validates the CRC on the way
        last = []
        with gzip.open(path, "rt", errors="replace") as f:
            for line in f:
                last.append(line)
                if len(last) > 3:
                    last.pop(0)
        return last
    size = path.stat().st_size
    with open(path, "rb") as f:
        f.seek(max(0, size - n_bytes))
        data = f.read().decode("utf-8", errors="replace")
    return data.splitlines(keepends=True)


def header_fields(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", errors="replace") as f:
            return f.readline().rstrip("\r\n").split(",")
    with open(path, "r", errors="replace") as f:
        return f.readline().rstrip("\r\n").split(",")


def check(module, name):
    path = C.find_table(module, name)
    if path is None:
        print(f"  [ -- ] {module}/{name}: not found")
        return None

    size_gb = path.stat().st_size / 1024 ** 3
    expect = APPROX_GB.get((module, name))
    hdr = header_fields(path)
    n_cols = len(hdr)

    lines = tail_lines(path)
    lines = [ln for ln in lines if ln.strip()]
    if not lines:
        print(f"  [FAIL] {module}/{name}: file is empty")
        return False

    last = lines[-1]
    ends_clean = last.endswith("\n") or last.endswith("\r\n")
    last_fields = len(last.rstrip("\r\n").split(","))

    status, notes = "ok", []

    # --- decisive structural check ---
    if last_fields != n_cols:
        status = "FAIL"
        notes.append(f"last row has {last_fields} fields, header has {n_cols}")
        if not ends_clean:
            notes.append("file does not end with a newline")
        notes.append("-> TRUNCATED (write was interrupted)")
    elif not ends_clean:
        notes.append("no trailing newline, but the last row is complete "
                     "(harmless)")

    # --- advisory size check, large tables only ---
    if expect and expect >= SIZE_CHECK_MIN_GB and size_gb < expect * 0.6:
        pct = size_gb / expect * 100
        if status == "ok":
            notes.append(f"[WARN] size {size_gb:.2f} GB vs ~{expect:.0f} GB "
                         f"expected ({pct:.0f}%) -- verify this is the full table")
        else:
            notes.append(f"size {size_gb:.2f} GB vs ~{expect:.0f} GB expected "
                         f"({pct:.0f}% -- consistent with truncation)")

    tag = "[FAIL]" if status == "FAIL" else "[ ok ]"
    size_str = f"{size_gb*1024:>7.1f} MB" if size_gb < 1 else f"{size_gb:>7.2f} GB"
    print(f"  {tag} {module}/{name:<14} {size_str}  {n_cols:>2} cols")
    for nt in notes:
        print(f"         {nt}")
    if status == "FAIL":
        print(f"         last line: {last.rstrip()[:90]}...")
    return status == "ok"


def main():
    print(f"Checking file integrity under {C.MIMIC_ROOT}\n")
    results = [check(m, n) for m, n in TABLES]
    bad = [r for r in results if r is False]

    print()
    if not bad:
        print("All files look complete.")
        sys.exit(0)

    print(f"{len(bad)} file(s) are TRUNCATED (last row is incomplete).")
    print("Everything else is fine -- only the file(s) listed above need action.\n")
    print("How to fix, in order of preference:")
    print("  1. Use the original .csv.gz instead of the extracted .csv.")
    print("     gzip carries a CRC, so a complete .gz is verified-intact, and")
    print("     DuckDB reads it directly -- there is no need to extract at all.")
    print("     If the .gz still exists next to the extracted copy, just delete")
    print("     (or rename) the extracted folder; the loader prefers .gz.")
    print("  2. Re-download only the damaged table(s) from PhysioNet, e.g.")
    print("       wget -N -c https://physionet.org/files/mimiciv/3.1/icu/chartevents.csv.gz")
    print("     The -c flag resumes rather than restarting the download.")
    print("  3. Re-extract, making sure the tool reports success and that the")
    print("     drive did not run out of space mid-extraction.")
    print("\nDo NOT work around this with ignore_errors=true. On a truncated file")
    print("that skips every row after the break -- silently -- and the pipeline")
    print("will finish successfully on a fraction of the data.")
    sys.exit(1)


if __name__ == "__main__":
    main()