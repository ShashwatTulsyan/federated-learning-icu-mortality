"""
Step 2: Split the cohort into federated clients (Section 5 of the design doc).

Three schemes:
  careunit    -- natural, clinically meaningful non-IID (PRIMARY, headline result)
  year_group  -- temporal drift via anchor_year_group (v3.1 has a COVID-era bucket)
  dirichlet   -- synthetic label skew with a tunable alpha (ABLATION)

Also prints a heterogeneity report so you can *demonstrate* the split is non-IID
rather than just assert it.

Run:  python partition.py
Out:  artifacts/partition_<scheme>.json
"""
import hashlib
import json

import numpy as np
import pandas as pd


def cohort_fingerprint(stay_ids):
    """Hash of the cohort's stay_id SEQUENCE.

    Partition files store row INDICES, which are only meaningful for the exact
    row ordering they were built from. DuckDB does not guarantee stable row
    order across runs (parallel scan, no ORDER BY), so re-running preprocess.py
    can silently invalidate an existing partition: the indices still load, but
    they now point at different patients. The symptom is subtle -- per-client
    mortality rates collapse toward the pooled mean because the "care units"
    have become random subsets.
    """
    b = np.asarray(stay_ids, dtype=np.int64).tobytes()
    return hashlib.sha256(b).hexdigest()[:16]

import config as C


# ---------------------------------------------------------------------------
# Partition schemes
# ---------------------------------------------------------------------------
def partition_careunit(cohort):
    counts = cohort.first_careunit.value_counts()
    valid = counts[counts >= C.MIN_CLIENT_SIZE].index
    dropped = counts[counts < C.MIN_CLIENT_SIZE]
    if len(dropped):
        print(f"  dropping {len(dropped)} care unit(s) below "
              f"MIN_CLIENT_SIZE={C.MIN_CLIENT_SIZE}: {list(dropped.index)}")
    return {u: cohort.index[cohort.first_careunit == u].to_numpy() for u in valid}


def partition_year_group(cohort):
    counts = cohort.anchor_year_group.value_counts()
    valid = counts[counts >= C.MIN_CLIENT_SIZE].index
    return {g: cohort.index[cohort.anchor_year_group == g].to_numpy()
            for g in sorted(valid)}


def partition_shuffled(cohort, seed):
    """Size-matched IID control: care-unit client SIZES, random membership.

    This is the control for the capacity experiment. Dirichlet(alpha=10) would
    also produce near-IID clients, but it equalises client sizes at the same
    time -- your care units range 1,544 to 4,619 stays, so equal sizes would
    give every client more data than the smallest real unit and reduce
    federation's benefit for a reason unrelated to heterogeneity. Changing two
    things at once cannot isolate either.

    Here each client keeps the exact size of the care unit it replaces, and only
    membership is randomised. Patients (not stays) are assigned, so the
    patient-level grouping constraint is preserved.
    """
    rng = np.random.default_rng(seed)
    sizes = cohort.first_careunit.value_counts()
    sizes = sizes[sizes >= C.MIN_CLIENT_SIZE]

    # shuffle whole patients, then fill clients to their target stay counts
    subs = cohort.subject_id.unique().copy()
    rng.shuffle(subs)
    by_sub = cohort.groupby("subject_id").indices     # subject -> row positions

    out, cursor = {}, 0
    for unit, target in sizes.items():
        rows, n = [], 0
        while cursor < len(subs) and n < target:
            idx = by_sub[subs[cursor]]
            rows.extend(idx)
            n += len(idx)
            cursor += 1
        out[unit] = np.sort(np.array(rows))
    return out


def partition_dirichlet(cohort, n_clients, alpha, seed):
    """Lower alpha => more skewed label distribution across clients."""
    rng = np.random.default_rng(seed)
    client_idx = [[] for _ in range(n_clients)]
    for label in sorted(cohort.label.unique()):
        # .copy() -- pandas may hand back a read-only view, which rng.shuffle
        # cannot write into
        idx = cohort.index[cohort.label == label].to_numpy().copy()
        rng.shuffle(idx)
        props = rng.dirichlet(alpha * np.ones(n_clients))
        cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for i, part in enumerate(np.split(idx, cuts)):
            client_idx[i].extend(part.tolist())
    return {f"client_{i}": np.array(sorted(v)) for i, v in enumerate(client_idx) if len(v)}


# ---------------------------------------------------------------------------
# Train / val / test split, stratified within each client
# ---------------------------------------------------------------------------
def split_client(idx, labels, seed, groups=None):
    """Stratified train/val/test split within one client.

    CRITICAL: splits by PATIENT (subject_id), not by ICU stay. A patient with
    several hospital admissions contributes several stays; letting those land on
    both sides of the split leaks patient-specific physiology into the test set
    and inflates every metric. Grouping is standard practice in the MIMIC
    literature and reviewers check for it.
    """
    if groups is None:
        raise ValueError("split_client requires `groups` (subject_id per row)")

    rng = np.random.default_rng(seed)
    idx = np.asarray(idx)
    g = np.asarray(groups)[idx]

    # one label per patient (positive if ANY of their stays ended in death) so
    # stratification survives the grouping
    df = pd.DataFrame({"idx": idx, "g": g, "y": np.asarray(labels)[idx]})
    per_patient = df.groupby("g")["y"].max()

    tr_g, va_g, te_g = [], [], []
    for cls in (0, 1):
        pts = per_patient.index[per_patient == cls].to_numpy().copy()
        rng.shuffle(pts)
        n = len(pts)
        n_te = int(round(n * C.TEST_FRAC))
        n_va = int(round(n * C.VAL_FRAC))
        te_g.extend(pts[:n_te])
        va_g.extend(pts[n_te:n_te + n_va])
        tr_g.extend(pts[n_te + n_va:])

    tr_g, va_g, te_g = set(tr_g), set(va_g), set(te_g)
    tr = df.idx[df.g.isin(tr_g)].to_numpy()
    va = df.idx[df.g.isin(va_g)].to_numpy()
    te = df.idx[df.g.isin(te_g)].to_numpy()
    return np.sort(tr), np.sort(va), np.sort(te)


def assert_no_patient_leakage(part, groups):
    """Hard check: no subject_id may appear in more than one split, anywhere."""
    groups = np.asarray(groups)
    seen = {}
    for client, sp in part.items():
        for split in ("train", "val", "test"):
            for s in set(groups[np.array(sp[split], dtype=int)]):
                key = (client, s)
                if key in seen and seen[key] != split:
                    raise AssertionError(
                        f"patient {s} appears in both {seen[key]} and {split} "
                        f"for client {client}")
                seen[key] = split
    print("  leakage check: no patient appears in more than one split  [OK]")


# ---------------------------------------------------------------------------
# Heterogeneity report (Section 5.4)
# ---------------------------------------------------------------------------
def jensen_shannon(p, q, eps=1e-12):
    p, q = np.asarray(p) + eps, np.asarray(q) + eps
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def check_clients(clients, scheme):
    """Fail loudly and usefully rather than KeyError'ing deep in pandas."""
    if not clients:
        raise ValueError(
            f"Partition '{scheme}' produced 0 clients: every group was smaller "
            f"than MIN_CLIENT_SIZE={C.MIN_CLIENT_SIZE}. Lower MIN_CLIENT_SIZE "
            f"in config.py, or check that the cohort actually built correctly."
        )
    if len(clients) < 2:
        raise ValueError(
            f"Partition '{scheme}' produced only {len(clients)} client — "
            f"federated learning needs at least 2. Lower MIN_CLIENT_SIZE."
        )
    return clients


def heterogeneity_report(cohort, clients):
    global_rate = cohort.label.mean()
    global_dist = [1 - global_rate, global_rate]

    rows = []
    for name, idx in clients.items():
        sub = cohort.loc[idx]
        rate = sub.label.mean()
        rows.append({
            "client": name,
            "n": len(idx),
            "mortality_%": round(rate * 100, 2),
            "mean_age": round(sub.age.mean(), 1),
            "median_los_h": round(sub.los_hours.median(), 1),
            "JS_div_vs_global": round(jensen_shannon([1 - rate, rate], global_dist), 5),
        })
    rep = pd.DataFrame(rows).sort_values("n", ascending=False)

    print("\n--- Heterogeneity report -------------------------------------")
    print(rep.to_string(index=False))
    print(f"\n  pooled mortality rate : {global_rate*100:.2f}%")
    print(f"  mortality rate spread : {rep['mortality_%'].min():.2f}% "
          f"-> {rep['mortality_%'].max():.2f}%")
    print(f"  mean JS divergence    : {rep['JS_div_vs_global'].mean():.5f}")
    print("  (larger spread / JS divergence = more non-IID)")
    print("---------------------------------------------------------------\n")
    return rep


# ---------------------------------------------------------------------------
def main():
    cohort = pd.read_parquet(C.COHORT_PQ).reset_index(drop=True)
    labels = cohort.label.to_numpy()
    groups = cohort.subject_id.to_numpy()      # grouped-split key

    scheme = C.PARTITION_SCHEME
    print(f"Partition scheme: {scheme}")
    if scheme == "careunit":
        clients = partition_careunit(cohort)
    elif scheme == "year_group":
        clients = partition_year_group(cohort)
    elif scheme == "shuffled":
        clients = partition_shuffled(cohort, C.SEED)
        print("  IID CONTROL: care-unit client sizes, randomised membership")
        print("  (only heterogeneity is removed; sizes are held fixed)")
    elif scheme == "dirichlet":
        clients = partition_dirichlet(cohort, C.DIRICHLET_N_CLIENTS,
                                      C.DIRICHLET_ALPHA, C.SEED)
        print(f"  alpha = {C.DIRICHLET_ALPHA} ({C.DIRICHLET_N_CLIENTS} clients)")
    else:
        raise ValueError(f"unknown PARTITION_SCHEME: {scheme}")

    clients = check_clients(clients, scheme)
    rep = heterogeneity_report(cohort, clients)

    out = {}
    for i, (name, idx) in enumerate(clients.items()):
        tr, va, te = split_client(idx, labels, C.SEED + i, groups=groups)
        out[name] = {"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()}
        print(f"  {name:<12} train={len(tr):>6}  val={len(va):>5}  test={len(te):>5}  "
              f"pos_train={int(labels[tr].sum()):>4}")

    assert_no_patient_leakage(out, groups)

    # Event-count warning. Discrimination metrics need enough POSITIVES, not just
    # enough rows: a client with 5 deaths in its test split will produce an AUROC
    # that swings wildly between seeds. Flag it now rather than after training.
    thin = []
    for name, sp in out.items():
        n_pos_tr = int(labels[np.array(sp["train"])].sum())
        n_pos_te = int(labels[np.array(sp["test"])].sum())
        n_pos_va = int(labels[np.array(sp["val"])].sum())
        if n_pos_tr < 50 or n_pos_te < 25:
            thin.append((name, n_pos_tr, n_pos_va, n_pos_te))

    if thin:
        print("\n  [WARN] client(s) with few outcome events:")
        for name, a, b, c in thin:
            print(f"         {name}: {a} train / {b} val / {c} test positives")
        print("         Per-client metrics for these will have very wide")
        print("         confidence intervals -- report the CIs, and do not read")
        print("         much into small AUROC differences for them.")
        print("         This is realistic (small sites genuinely have few events)")
        print("         and is precisely where federation should beat local-only,")
        print("         so it is worth keeping and discussing rather than dropping.")

    path = C.OUT_DIR / f"partition_{scheme}.json"
    payload = {"__fingerprint__": cohort_fingerprint(cohort.stay_id.to_numpy()),
               "__n_stays__": int(len(cohort)),
               "clients": out}
    with open(path, "w") as f:
        json.dump(payload, f)
    rep.to_csv(C.OUT_DIR / f"heterogeneity_{scheme}.csv", index=False)
    print(f"\nSaved -> {path}")
    print("Next: python fed_train.py")


if __name__ == "__main__":
    main()