"""
The four experiments for the paper, all multi-seed.

  E1  Capacity series      -- the headline finding
  E2  Algorithm comparison -- FedAvg vs FedProx vs FedAdam, controlled
  E3  ICD leakage          -- quantifies the discharge-code inflation
  E4  Component ablation   -- what each piece contributes

On uncertainty
--------------
These report mean +/- SD ACROSS SEEDS, not bootstrap CIs, and that is deliberate.
Seed SD measures training variability, which is what differs between the arms
being compared. Bootstrap CIs measure patient-sampling variability, which is
identical across arms and so cancels in a comparison. Comparisons are PAIRED by
seed, which removes seed variance from the contrast.

Bootstrap CIs still belong in the paper for the single headline model -- run
`fed_train.py` once for those.

On touching the test set
------------------------
These experiments read the test set, which is legitimate because they are
ANALYSES, not selections: every arm is reported, nothing is chosen on the basis
of test performance. Hyperparameters were already fixed by tune.py on validation.
Do not add arms after seeing results and then report only the best.

Run:  python run_experiments.py            (all four, ~4-6 h)
      python run_experiments.py e1 e2      (a subset)
Out:  artifacts/exp1_capacity.csv, exp2_algorithms.csv,
      exp3_leakage.csv, exp4_ablation.csv  (+ *_paired.csv)
"""
import copy
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats

import config as C
import fed_train as FT
import metrics as MET
from models import build_model

SEEDS = [42, 101, 2026, 7, 1234]

# E2 compares 7 arms against one reference, so Bonferroni demands a much smaller
# alpha than a single comparison. Five seeds gave differences that were nominally
# significant but did not survive correction, so E2 gets extra seeds.
E2_SEEDS = SEEDS + [11, 77, 314, 2718, 1618]

MAX_ROUNDS = 60
PATIENCE = 10

# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------
# Runs are independent, so they parallelise across PROCESSES. This is the right
# axis: a small GRU is kernel-launch bound, so one process leaves the GPU mostly
# idle between launches and several processes interleave to fill it.
#
# Each worker needs ~0.5 GB of CUDA context plus ~1 GB of working memory, so 8
# workers use ~12 GB of 24 GB. Going past 12 risks OOM without helping, because
# the GPU saturates first.
#
# Set to 1 to run sequentially (useful when debugging -- worker tracebacks are
# harder to read).
N_JOBS = 8

_W = {}          # per-worker cache, populated by _worker_init


def _worker_init():
    """Load the dataset ONCE per worker process, not once per job."""
    import config as _C
    global _W
    _W["D"] = load_full()
    if torch.cuda.is_available():
        torch.cuda.init()


def _worker(job):
    """Train one arm at one seed. Must be module-level so it is picklable."""
    D = _W.get("D") or load_full()
    X, mask, static, y, part = D[0], D[1], D[2], D[3], D[4]
    for k, v in job.get("overrides", {}).items():
        setattr(C, k, v)
    ti = job.get("ts_idx")
    xi = job.get("tab_idx")
    Xs = X[:, :, ti] if ti is not None else X
    Ms = mask[:, :, ti] if ti is not None else mask
    Ss = static[:, xi] if xi is not None else static
    nf, ns = Xs.shape[2], Ss.shape[1]
    seed = job["seed"]

    out = dict(job.get("tag", {}))
    out["seed"] = seed
    g, pc = fit_federated(Xs, Ms, Ss, y, part, nf, ns, seed)
    out.update({"auroc": g["auroc"], "auprc": g["auprc"],
                "brier": g.get("brier"), "calib_slope": g.get("calib_slope")})
    if job.get("with_baselines"):
        cent = fit_pooled(Xs, Ms, Ss, y, part, nf, ns, seed)
        loc = fit_pooled(Xs, Ms, Ss, y, part, nf, ns, seed, per_client_only=True)
        out["cent_auroc"] = cent["auroc"]
        out["cent_auprc"] = cent["auprc"]
        out["gap_auprc"] = cent["auprc"] - g["auprc"]
        out["fl_gain_auprc"] = float(np.mean(
            [pc[k]["auprc"] - loc[k]["auprc"] for k in part]))
    free_gpu()
    return out


def run_jobs(jobs, label=""):
    """Execute jobs in parallel, falling back to sequential on any failure.

    Determinism is preserved: every job seeds torch and numpy from its own
    `seed` field before training, so results do not depend on how many workers
    ran or in what order they finished.

    Parallelism only pays when jobs are long. Each worker costs roughly 20 s of
    startup (interpreter + torch import + CUDA context + data load), so for jobs
    of a few seconds the overhead dominates. On the real cohort jobs run for
    tens of seconds to minutes, where the overhead is negligible.
    """
    n = min(N_JOBS, len(jobs)) if N_JOBS > 1 else 1
    if n > 1 and len(jobs) < 2 * n:
        print(f"  only {len(jobs)} jobs for {n} workers -- running serially "
              f"(startup overhead would exceed the saving)")
        n = 1
    t0 = time.time()
    if n <= 1:
        res = []
        for i, j in enumerate(jobs, 1):
            res.append(_worker(j))
            print(f"    [{i}/{len(jobs)}] {res[-1].get('auprc', float('nan')):.4f}")
        return res

    import multiprocessing as mp
    ctx = mp.get_context("spawn")     # required for CUDA, and on Windows
    print(f"  running {len(jobs)} jobs on {n} workers ...")
    try:
        with ctx.Pool(n, initializer=_worker_init) as pool:
            res = []
            for i, r in enumerate(pool.imap_unordered(_worker, jobs), 1):
                res.append(r)
                if i % max(1, len(jobs) // 20) == 0 or i == len(jobs):
                    el = time.time() - t0
                    eta = el / i * (len(jobs) - i)
                    print(f"    [{i:>3}/{len(jobs)}] {el/60:.0f}m elapsed, "
                          f"~{eta/60:.0f}m left")
        return res
    except Exception as e:
        print(f"  [WARN] parallel execution failed ({e}); falling back to serial")
        return [_worker(j) for j in jobs]


def load_partition(path, stay_ids):
    """Load a partition file and REFUSE it if it was built for a different
    cohort ordering. See partition.cohort_fingerprint for why this matters."""
    import hashlib
    with open(path) as f:
        raw = json.load(f)
    if "clients" not in raw:            # legacy file, no fingerprint
        print("  [WARN] partition file has no cohort fingerprint (old format).")
        print("         Cannot verify it matches this cohort. Re-run "
              "partition.py to make this checkable.")
        return raw
    want = hashlib.sha256(
        np.asarray(stay_ids, dtype=np.int64).tobytes()).hexdigest()[:16]
    got = raw.get("__fingerprint__")
    if got != want:
        raise RuntimeError(
            f"\nPartition/cohort MISMATCH.\n"
            f"  partition built for cohort {got}, current cohort is {want}.\n"
            f"  The stored row indices point at different patients, which\n"
            f"  silently scrambles client membership -- per-client mortality\n"
            f"  rates collapse toward the pooled mean.\n"
            f"  Fix: re-run `python partition.py`.")
    return raw["clients"]


# ---------------------------------------------------------------------------
# Data, with feature names so experiments can subset without re-preprocessing
# ---------------------------------------------------------------------------
def load_full():
    d = np.load(C.TS_NPZ, allow_pickle=True)
    X = torch.from_numpy(d["X"]).float()
    mask = torch.from_numpy(d["mask"]).float()
    y = torch.from_numpy(d["y"]).float()
    stay_ids = d["stay_id"]
    ts_names = [str(x) for x in d["feature_names"]]

    agg = pd.read_parquet(C.AGG_PQ).set_index("stay_id").reindex(stay_ids)
    agg = agg.drop(columns=[c for c in ("label", "subject_id") if c in agg.columns])
    # E3 needs both variants present so it can compare them; it selects columns
    # explicitly per arm. Every other experiment picks with exclude=["cci_curr_",
    # ...] so the leaky columns are never reachable by accident.
    tab_names = list(agg.columns)
    static = torch.from_numpy(agg.to_numpy(dtype=np.float32))

    part = load_partition(C.OUT_DIR / f"partition_{C.PARTITION_SCHEME}.json",
                          stay_ids)
    return X, mask, static, y, part, ts_names, tab_names, stay_ids


def pick(names, include=None, exclude=None, prefix=None):
    """Column indices by name / prefix, with optional exclusions."""
    idx = []
    for i, n in enumerate(names):
        ok = True
        if include is not None:
            ok = any(n == a or n.startswith(a) for a in include)
        if ok and prefix is not None:
            ok = any(n.startswith(p) for p in prefix)
        if ok and exclude is not None:
            ok = not any(n == e or n.startswith(e) for e in exclude)
        if ok:
            idx.append(i)
    return idx


# ---------------------------------------------------------------------------
# Training arms
# ---------------------------------------------------------------------------
def _loaders(X, mask, static, y, part, names, seed):
    out = {}
    for n in names:
        sp = part[n]
        tr = np.array(sp["train"])
        norm = FT.fit_normalizer(X, static, tr)
        out[n] = {
            "train": FT.build_loader(X, mask, static, y, tr, norm,
                                     C.BATCH_SIZE, True, weighted=True),
            "val": FT.build_loader(X, mask, static, y, np.array(sp["val"]),
                                   norm, 1024, False),
            "test": FT.build_loader(X, mask, static, y, np.array(sp["test"]),
                                    norm, 1024, False),
            "n": FT.agg_weight_for(y, tr),
            "pos_weight": FT.pos_weight_for(y, tr),
        }
    return out


def _probs(z, cal):
    return (cal.predict_proba(z.reshape(-1, 1))[:, 1] if cal is not None
            else 1.0 / (1.0 + np.exp(-z)))


def fit_federated(X, mask, static, y, part, nf, ns, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    names = list(part)
    gm = build_model(C, nf, ns)
    gstate = copy.deepcopy(gm.state_dict())
    sopt = FT.ServerAdam(gstate, C.SERVER_LR) if C.FL_ALGO == "fedadam" else None
    L = _loaders(X, mask, static, y, part, names, seed)

    probe = build_model(C, nf, ns)
    LOCAL = FT.local_keys(probe)
    heads = {n: None for n in names} if (C.PERSONALIZE or LOCAL) else None
    bns = {}

    best, best_state, best_heads, patience = -1.0, None, None, 0
    for _ in range(MAX_ROUNDS):
        states, ws = [], []
        for n in names:
            m = build_model(C, nf, ns); m.load_state_dict(gstate)
            if C.FEDBN and n in bns:
                m.load_state_dict(bns[n], strict=False)
            if heads is not None and heads.get(n) is not None:
                m.load_state_dict(heads[n], strict=False)
            FT.local_train(m, L[n]["train"], C.LOCAL_EPOCHS, C.LR,
                           global_state=gstate if C.FL_ALGO == "fedprox" else None,
                           mu=C.PROX_MU if C.FL_ALGO == "fedprox" else 0.0,
                           pos_weight=L[n]["pos_weight"])
            sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
            if C.FEDBN:
                bk = [k for k in sd if "running_" in k or "num_batches" in k]
                for k in bk:
                    bns.setdefault(n, {})[k] = sd[k]
                sd = {k: v for k, v in sd.items() if k not in bk}
            keep = (set(m.head_keys()) if C.PERSONALIZE else set()) | LOCAL
            if keep:
                heads[n] = {k: sd[k] for k in keep if k in sd}
                sd = {k: v for k, v in sd.items() if k not in keep}
            states.append(sd); ws.append(L[n]["n"])

        agg = FT.fedavg(states, ws)
        merged = copy.deepcopy(gstate); merged.update(agg)
        gstate = sopt.step(gstate, merged) if sopt else merged

        zs, ys = [], []
        for n in names:
            m = build_model(C, nf, ns); m.load_state_dict(gstate)
            if C.FEDBN and n in bns:
                m.load_state_dict(bns[n], strict=False)
            if heads is not None and heads.get(n) is not None:
                m.load_state_dict(heads[n], strict=False)
            z, yy = FT.raw_logits(m, L[n]["val"])
            zs.append(z); ys.append(yy)
        yv = np.concatenate(ys)
        auprc = (MET.average_precision_score(yv, 1/(1+np.exp(-np.concatenate(zs))))
                 if len(np.unique(yv)) > 1 else np.nan)
        if np.isfinite(auprc) and auprc > best:
            best, best_state = auprc, copy.deepcopy(gstate)
            best_heads = copy.deepcopy(heads) if heads else None
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    # test evaluation, threshold + calibration from validation
    zv, yv, zt, yt, per_client = [], [], [], [], {}
    for n in names:
        m = build_model(C, nf, ns); m.load_state_dict(best_state)
        if C.FEDBN and n in bns:
            m.load_state_dict(bns[n], strict=False)
        if best_heads is not None and best_heads.get(n) is not None:
            m.load_state_dict(best_heads[n], strict=False)
        a, b = FT.raw_logits(m, L[n]["val"]); zv.append(a); yv.append(b)
        c, d_ = FT.raw_logits(m, L[n]["test"]); zt.append(c); yt.append(d_)
        cal_c = FT.fit_calibrator_from(a, b)
        th = FT.pick_threshold(b, _probs(a, cal_c))
        per_client[n] = MET.point_metrics(d_, _probs(c, cal_c), threshold=th)
    zv, yv = np.concatenate(zv), np.concatenate(yv)
    zt, yt = np.concatenate(zt), np.concatenate(yt)
    cal = FT.fit_calibrator_from(zv, yv)
    th = FT.pick_threshold(yv, _probs(zv, cal))
    return MET.point_metrics(yt, _probs(zt, cal), threshold=th), per_client


def fit_pooled(X, mask, static, y, part, nf, ns, seed, per_client_only=False):
    """Centralized (all data pooled) or local-only (each client alone)."""
    torch.manual_seed(seed); np.random.seed(seed)
    names = list(part)
    if per_client_only:
        out = {}
        for n in names:
            sp = part[n]
            tr, va, te = (np.array(sp[k]) for k in ("train", "val", "test"))
            norm = FT.fit_normalizer(X, static, tr)
            tl = FT.build_loader(X, mask, static, y, tr, norm, C.BATCH_SIZE, True, True)
            vl = FT.build_loader(X, mask, static, y, va, norm, 1024, False)
            el = FT.build_loader(X, mask, static, y, te, norm, 1024, False)
            m = build_model(C, nf, ns)
            FT.local_train(m, tl, MAX_ROUNDS // 2, C.LR,
                           pos_weight=FT.pos_weight_for(y, tr))
            zv, yv = FT.raw_logits(m, vl); zt, yt = FT.raw_logits(m, el)
            cal = FT.fit_calibrator_from(zv, yv)
            th = FT.pick_threshold(yv, _probs(zv, cal))
            out[n] = MET.point_metrics(yt, _probs(zt, cal), threshold=th)
        return out

    tr = np.concatenate([part[n]["train"] for n in names])
    va = np.concatenate([part[n]["val"] for n in names])
    te = np.concatenate([part[n]["test"] for n in names])
    norm = FT.fit_normalizer(X, static, tr)
    tl = FT.build_loader(X, mask, static, y, tr, norm, C.BATCH_SIZE, True, True)
    vl = FT.build_loader(X, mask, static, y, va, norm, 1024, False)
    el = FT.build_loader(X, mask, static, y, te, norm, 1024, False)
    m = build_model(C, nf, ns)
    pw = FT.pos_weight_for(y, tr)
    best, bs, patience = -1.0, None, 0
    for _ in range(MAX_ROUNDS):
        FT.local_train(m, tl, 1, C.LR, pos_weight=pw)
        zv, yv = FT.raw_logits(m, vl)
        a = MET.average_precision_score(yv, 1/(1+np.exp(-zv)))
        if a > best:
            best, bs, patience = a, copy.deepcopy(m.state_dict()), 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    m.load_state_dict(bs)
    zv, yv = FT.raw_logits(m, vl); zt, yt = FT.raw_logits(m, el)
    cal = FT.fit_calibrator_from(zv, yv)
    th = FT.pick_threshold(yv, _probs(zv, cal))
    return MET.point_metrics(yt, _probs(zt, cal), threshold=th)


free_gpu = FT.free_gpu     # shared implementation, see fed_train.py


def gpu_report(tag=""):
    if not torch.cuda.is_available():
        return ""
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    t = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{tag}VRAM {a:.2f} GB in use / {r:.2f} reserved / {t:.0f} total"


def run_arm(X, mask, static, y, part, seed, want=("fed", "cent", "local")):
    nf, ns = X.shape[2], static.shape[1]
    out = {}
    if "fed" in want:
        g, pc = fit_federated(X, mask, static, y, part, nf, ns, seed)
        out["fed"], out["fed_clients"] = g, pc
    if "cent" in want:
        out["cent"] = fit_pooled(X, mask, static, y, part, nf, ns, seed)
    if "local" in want:
        out["local_clients"] = fit_pooled(X, mask, static, y, part, nf, ns,
                                          seed, per_client_only=True)
    return out


def tag_name(out_name):
    """Suffix output files with the partition scheme.

    Without this, running the IID control would overwrite exp1_capacity.csv from
    the care-unit run -- destroying the very result the control is meant to be
    compared against.
    """
    sch = getattr(C, "PARTITION_SCHEME", "careunit")
    return out_name if sch == "careunit" else f"{out_name}_{sch}"


def summarise(rows, group_cols, out_name):
    out_name = tag_name(out_name)
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT_DIR / f"{out_name}_raw.csv", index=False)
    agg = df.groupby(group_cols).agg(
        auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
        auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std"),
        n_seeds=("auroc", "count")).round(4).reset_index()
    agg.to_csv(C.OUT_DIR / f"{out_name}.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + agg.to_string(index=False))
    return df, agg


def paired_tests(df, key, ref, metric="auprc", out_name=None):
    """Paired-by-seed comparison of every level against a reference level."""
    piv = df.pivot_table(index="seed", columns=key, values=metric)
    if ref not in piv.columns:
        return None
    rows = []
    for c in piv.columns:
        if c == ref:
            continue
        a, b = piv[c].values, piv[ref].values
        d = a - b
        if np.allclose(d, 0):
            t = p = sd = np.nan
        else:
            t, p = stats.ttest_rel(a, b)
            sd = d.std(ddof=1)
        rows.append({"arm": c, "vs": ref, "mean_diff": round(np.nanmean(d), 5),
                     "sd_diff": round(sd, 5) if sd == sd else np.nan,
                     "paired_p": round(p, 4) if p == p else np.nan,
                     "wins": int((d > 0).sum()), "of": len(d)})
    r = pd.DataFrame(rows)
    if out_name:
        out_name = tag_name(out_name)
    # Comparing many arms against one reference inflates the false-positive rate.
    # With 6 arms at alpha=0.05 there is a ~26% chance of at least one spurious
    # "significant" result, so correct for it and report both thresholds.
    n_comp = len(r)
    alpha_b = 0.05 / max(n_comp, 1)
    r["sig_uncorrected"] = r.paired_p < 0.05
    r["sig_bonferroni"] = r.paired_p < alpha_b
    if out_name:
        r.to_csv(C.OUT_DIR / f"{out_name}_paired.csv", index=False)
    print(f"\n  paired vs {ref} (metric={metric}); "
          f"Bonferroni alpha = {alpha_b:.4f} for {n_comp} comparisons:")
    print("  " + r.to_string(index=False).replace("\n", "\n  "))
    n_b = int(r.sig_bonferroni.sum())
    n_u = int(r.sig_uncorrected.sum())
    if n_u and not n_b:
        print(f"\n  NOTE: {n_u} comparison(s) reach p < 0.05 but NONE survive")
        print("  Bonferroni. Report as 'nominally significant, not robust to")
        print("  multiple comparisons' and raise the seed count if the claim")
        print("  matters to your argument.")
    return r


# ===========================================================================
# E1 — CAPACITY SERIES  (the headline)
# ===========================================================================
VITALS = ["heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2", "temp_c"]
CORE = VITALS + ["gcs_eye", "gcs_verbal", "gcs_motor", "urine",
                 "creatinine", "potassium", "sodium", "chloride", "bicarbonate",
                 "hematocrit", "wbc", "glucose", "magnesium", "calcium", "lactate"]


def experiment_capacity(D):
    X, mask, static, y, part, ts_names, tab_names, _ = D
    print("\n" + "=" * 74)
    print("E1 — CAPACITY SERIES")
    print("  Architecture FIXED (hybrid). Only the feature set varies, so")
    print("  capacity is not confounded with architecture.")
    print("=" * 74)

    base_tab = pick(tab_names, include=["age", "gender_m", "admission_emergency"])
    no_cci = pick(tab_names, exclude=["cci_", "charlson_", "n_diagnoses"])
    prior_cci = no_cci + pick(tab_names, prefix=["cci_prior_", "charlson_prior",
                                                 "n_diagnoses_prior"])
    LEVELS = [
        ("L1 vitals",        pick(ts_names, include=VITALS), base_tab),
        ("L2 vitals+labs",   pick(ts_names, include=CORE),   base_tab),
        ("L3 all series",    list(range(len(ts_names))),     base_tab),
        ("L4 +tabular",      list(range(len(ts_names))),     no_cci),
        ("L5 +comorbidity",  list(range(len(ts_names))),     prior_cci),
    ]

    jobs = []
    for name, ti, xi in LEVELS:
        nparam = sum(p.numel() for p in build_model(C, len(ti), len(xi)).parameters())
        print(f"  {name}: {len(ti)} series + {len(xi)} tabular = {nparam:,} params")
        for sd in SEEDS:
            jobs.append({"seed": sd, "ts_idx": ti, "tab_idx": xi,
                         "with_baselines": True,
                         "tag": {"level": name, "n_ts": len(ti),
                                 "n_tab": len(xi), "n_param": nparam}})
    rows = run_jobs(jobs, "E1")

    df, agg = summarise(rows, ["level"], "exp1_capacity")
    trend = df.groupby("level").agg(
        n_param=("n_param", "first"),
        gap=("gap_auprc", "mean"), gap_sd=("gap_auprc", "std"),
        fl_gain=("fl_gain_auprc", "mean"), fl_gain_sd=("fl_gain_auprc", "std")
    ).reset_index().sort_values("n_param")
    trend.to_csv(C.OUT_DIR / f"{tag_name('exp1_capacity')}_trend.csv", index=False)
    print("\n  CAPACITY TREND (the headline table):")
    print("  " + trend.round(4).to_string(index=False).replace("\n", "\n  "))

    for col, label in (("gap_auprc", "federated-centralized gap"),
                       ("fl_gain_auprc", "FL gain over local-only")):
        r, p = stats.spearmanr(df.n_param, df[col])
        print(f"\n  Spearman(parameters, {label}) = {r:+.3f}, p = {p:.4f}"
              f"   [n = {len(df)} runs]")
    return df


# ===========================================================================
# E2 — ALGORITHM COMPARISON
# ===========================================================================
def experiment_algorithms(D):
    X, mask, static, y, part, _, _, _ = D
    print("\n" + "=" * 74)
    print("E2 — ALGORITHM COMPARISON (controlled, paired by seed)")
    print("=" * 74)
    arms = [("FedAvg", {"FL_ALGO": "fedavg", "PROX_MU": 0.0}),
            ("FedProx mu=0.001", {"FL_ALGO": "fedprox", "PROX_MU": 0.001}),
            ("FedProx mu=0.01", {"FL_ALGO": "fedprox", "PROX_MU": 0.01}),
            ("FedProx mu=0.1", {"FL_ALGO": "fedprox", "PROX_MU": 0.1}),
            ("FedAdam slr=0.003", {"FL_ALGO": "fedadam", "SERVER_LR": 0.003}),
            ("FedAdam slr=0.01", {"FL_ALGO": "fedadam", "SERVER_LR": 0.01}),
            ("FedAdam slr=0.03", {"FL_ALGO": "fedadam", "SERVER_LR": 0.03})]
    seeds = E2_SEEDS
    print(f"  {len(arms)} arms x {len(seeds)} seeds")
    jobs = [{"seed": sd, "overrides": ov, "tag": {"algorithm": name}}
            for name, ov in arms for sd in seeds]
    rows = run_jobs(jobs, "E2")

    df, _ = summarise(rows, ["algorithm"], "exp2_algorithms")
    paired_tests(df, "algorithm", "FedAvg", "auprc", "exp2_algorithms")
    return df


# ===========================================================================
# E3 — ICD LEAKAGE
# ===========================================================================
def experiment_leakage(D):
    X, mask, static, y, part, _, tab_names, _ = D
    print("\n" + "=" * 74)
    print("E3 — ICD LEAKAGE")
    print("  Quantifies how much discharge-assigned diagnosis codes inflate")
    print("  a model that is supposed to predict from the first 24h.")
    print("=" * 74)
    no_cci = pick(tab_names, exclude=["cci_", "charlson_", "n_diagnoses"])
    prior = no_cci + pick(tab_names, prefix=["cci_prior_", "charlson_prior",
                                             "n_diagnoses_prior"])
    curr = no_cci + pick(tab_names, prefix=["cci_curr_", "charlson_curr",
                                            "n_diagnoses_curr"])
    if len(curr) == len(no_cci):
        print("  [SKIP] current-admission columns absent. Set")
        print("         COMORBIDITY_SOURCE='both' and re-run preprocess.py")
        return None

    jobs = [{"seed": sd, "tab_idx": xi, "tag": {"variant": name}}
            for name, xi in (("no comorbidities", no_cci),
                             ("prior admissions (valid)", prior),
                             ("current admission (LEAKY)", curr))
            for sd in SEEDS]
    rows = run_jobs(jobs, "E3")

    df, _ = summarise(rows, ["variant"], "exp3_leakage")
    paired_tests(df, "variant", "prior admissions (valid)", "auprc", "exp3_leakage")
    print("\n  The difference between the leaky and valid arms is the amount by")
    print("  which models using discharge codes overstate 24h prediction.")
    return df


# ===========================================================================
# E4 — COMPONENT ABLATION
# ===========================================================================
def experiment_ablation(D):
    X, mask, static, y, part, ts_names, tab_names, _ = D
    print("\n" + "=" * 74)
    print("E4 — COMPONENT ABLATION (each row removes ONE thing)")
    print("=" * 74)
    full_tab = pick(tab_names, exclude=["cci_curr_", "charlson_curr",
                                        "n_diagnoses_curr"])
    base_tab = pick(tab_names, include=["age", "gender_m", "admission_emergency"])
    no_cci = pick(tab_names, exclude=["cci_", "charlson_", "n_diagnoses"])
    no_interv = pick(ts_names, exclude=["vasopressor", "ventilation"])
    allts = list(range(len(ts_names)))

    saved = {k: getattr(C, k) for k in ("MODEL", "FEDBN", "CALIBRATE",
                                        "IMBALANCE", "PERSONALIZE")}
    arms = [
        ("full model",            allts, full_tab, {}),
        ("- comorbidities",       allts, no_cci,   {}),
        ("- tabular branch",      allts, base_tab, {}),
        ("- vasopressor/vent",    no_interv, full_tab, {}),
        ("- FedBN",               allts, full_tab, {"FEDBN": False}),
        ("- calibration",         allts, full_tab, {"CALIBRATE": False}),
        ("- imbalance handling",  allts, full_tab, {"IMBALANCE": "none"}),
        ("+ personalisation",     allts, full_tab, {"PERSONALIZE": True}),
    ]
    jobs = [{"seed": sd, "ts_idx": ti, "tab_idx": xi, "overrides": ov,
             "tag": {"variant": name}}
            for name, ti, xi, ov in arms for sd in SEEDS]
    rows = run_jobs(jobs, "E4")

    df, _ = summarise(rows, ["variant"], "exp4_ablation")
    paired_tests(df, "variant", "full model", "auprc", "exp4_ablation")

    # Calibration is RANK-PRESERVING, so AUPRC and AUROC cannot detect it by
    # construction. Its effect shows up in Brier and calibration slope, so test
    # those separately rather than reporting a spurious zero.
    print("\n  Calibration and FedBN are invisible to ranking metrics.")
    print("  Their effect is in Brier / calibration slope:")
    b = df.groupby("variant").agg(brier_mean=("brier", "mean"),
                                  brier_sd=("brier", "std"),
                                  slope_mean=("calib_slope", "mean")).round(4)
    print("  " + b.to_string().replace("\n", "\n  "))
    paired_tests(df, "variant", "full model", "brier", "exp4_ablation_brier")

    if C.MODEL == "hybrid":
        print("\n  NOTE: the '- FedBN' arm is a no-op for MODEL='hybrid', which")
        print("  uses LayerNorm rather than BatchNorm. Include that arm only when")
        print("  ablating MODEL='mlp', and say so in the paper rather than")
        print("  reporting an uninformative zero.")
    return df


# ===========================================================================
def main():
    global N_JOBS
    args = [a.lower() for a in sys.argv[1:]]
    for a in list(args):
        if a.startswith("--jobs="):
            N_JOBS = int(a.split("=")[1]); args.remove(a)
    which = args or ["e1", "e2", "e3", "e4"]
    print(f"Parallel workers: {N_JOBS}")
    print(f"Device: {FT.DEV} | seeds: {SEEDS}")
    print(f"Running: {', '.join(which)}\n")
    D = load_full()
    n_stay, n_ts, n_tab = len(D[3]), len(D[5]), len(D[6])
    print(f"  {n_stay:,} stays | {len(D[4])} clients | "
          f"{n_ts} series features | {n_tab} tabular features")
    mb = (D[0].numel() + D[1].numel() + D[2].numel()) * 4 / 1e6
    print(f"  master tensors: {mb:.0f} MB CPU")
    rep = gpu_report("  ")
    if rep:
        print(rep)
    if n_tab < 200:
        print("\n  [WARN] only "
              f"{n_tab} tabular features. If you expected BOTH comorbidity")
        print("  variants (~213), COMORBIDITY_SOURCE='both' did not take effect")
        print("  and E3 will skip itself. Fix config.py and re-run preprocess.py.")

    t0 = time.time()
    if "e1" in which:
        experiment_capacity(D)
    if "e2" in which:
        experiment_algorithms(D)
    if "e3" in which:
        experiment_leakage(D)
    if "e4" in which:
        experiment_ablation(D)
    print(f"\nTotal: {(time.time()-t0)/60:.0f} min")
    print(f"Results in {C.OUT_DIR}/exp*.csv")
    print("\nFor bootstrap CIs on the headline model, run fed_train.py once.")


if __name__ == "__main__":
    main()