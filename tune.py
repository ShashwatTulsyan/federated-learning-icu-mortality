"""
Hyperparameter search — staged random search, parallel, validation-only.

Two rules this enforces, both easy to get wrong:

1. **Selection happens on VALIDATION. The test set is never read.**
   Tuning on test and then reporting test performance is the most common route
   to an optimistic published model. The test set stays sealed until you lock a
   configuration and run `fed_train.py` once.

2. **Configs are confirmed across seeds.** With ~60 configs sampled, the best
   single-seed score is very likely the luckiest seed rather than the best
   setting. Stage 2 re-runs the leaders on several seeds and ranks by
   `mean − SD`, preferring configs that are good AND reproducible.

Staged design (cheaper than a grid, better at finding good regions):
  Stage 1 — N_STAGE1 random configs x 1 seed    (broad)
  Stage 2 — top K configs x 3 seeds             (confirm)
  Stage 3 — winner x 5 seeds                    (report)

Run:  python tune.py
      python tune.py --jobs=8
Out:  artifacts/tuning_stage1.csv, tuning_stage2.csv, tuning_best.json
"""
import copy
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

import config as C
import fed_train as FT
import metrics as MET
from models import build_model

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
# "log" ranges sample log-uniformly, which is correct for learning rates and
# regularisation strengths: 1e-4 -> 1e-3 is as large a step as 1e-3 -> 1e-2, so
# uniform sampling would waste most draws at the top of the range.
SPACE = {
    "LR":            ("log", 1e-4, 5e-3),
    "PROX_MU":       ("log", 1e-4, 3e-1),
    "SERVER_LR":     ("log", 1e-3, 1e-1),
    "HIDDEN":        ("choice", [64, 128, 192, 256]),
    "DROPOUT":       ("choice", [0.1, 0.2, 0.3, 0.4]),
    "BATCH_SIZE":    ("choice", [128, 256, 512]),
    "LOCAL_EPOCHS":  ("choice", [1, 2, 3]),
    "FL_ALGO":       ("choice", ["fedavg", "fedprox", "fedadam"]),
    # Added after the ablation flagged both as consequential:
    #   IMBALANCE   -- removing class weighting improved AUPRC on 5/5 seeds
    #   PERSONALIZE -- per-client heads helped under genuine heterogeneity
    # Searching them here means the choice is made on VALIDATION rather than
    # lifted from a test-set ablation, which would bias the final estimate.
    "IMBALANCE":     ("choice", ["pos_weight", "sampler", "none"]),
    "PERSONALIZE":   ("choice", [True, False]),
}

N_STAGE1 = 60
TOP_K = 8
STAGE2_SEEDS = [42, 101, 2026]
STAGE3_SEEDS = [42, 101, 2026, 7, 1234]
MAX_ROUNDS = 60
PATIENCE = 10

# Parallel workers. Each needs ~0.5 GB CUDA context plus working memory that
# scales with cohort size (~1 GB at 22k stays, ~1.7 GB at 67k). Eight fit
# comfortably in 24 GB either way. Set 1 to debug -- worker tracebacks inside a
# process pool are hard to read.
N_JOBS = 8

INT_KEYS = {"HIDDEN", "BATCH_SIZE", "LOCAL_EPOCHS"}
BOOL_KEYS = {"PERSONALIZE"}

_W = {}


def sample_config(rng):
    cfg = {}
    for k, spec in SPACE.items():
        if spec[0] == "log":
            cfg[k] = float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
        else:
            cfg[k] = spec[1][rng.integers(len(spec[1]))]
    # pin irrelevant knobs to defaults so the results table stays readable
    if cfg["FL_ALGO"] != "fedprox":
        cfg["PROX_MU"] = 0.0
    if cfg["FL_ALGO"] != "fedadam":
        cfg["SERVER_LR"] = C.SERVER_LR
    return cfg


def apply(cfg):
    """Set config values, coercing numpy scalars back to Python types.

    Values round-tripped through a DataFrame return as np.int64 / np.float64 /
    np.bool_, and torch rejects np.int64 for arguments like hidden_size.
    """
    for k, v in cfg.items():
        if k in INT_KEYS:
            v = int(v)
        elif k in BOOL_KEYS:
            v = bool(v)
        elif isinstance(v, np.floating):
            v = float(v)
        elif isinstance(v, np.str_):
            v = str(v)
        setattr(C, k, v)


# ---------------------------------------------------------------------------
# One training run -> best VALIDATION AUPRC. The test split is never touched.
# ---------------------------------------------------------------------------
def fit_and_score(X, mask, static, y, part, n_features, n_static, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    names = list(part)

    gm = build_model(C, n_features, n_static)
    gstate = copy.deepcopy(gm.state_dict())
    sopt = FT.ServerAdam(gstate, C.SERVER_LR) if C.FL_ALGO == "fedadam" else None

    loaders = {}
    for n in names:
        sp = part[n]
        tr = np.array(sp["train"])
        norm = FT.fit_normalizer(X, static, tr)
        loaders[n] = {
            "train": FT.build_loader(X, mask, static, y, tr, norm,
                                     C.BATCH_SIZE, True, weighted=True),
            "val": FT.build_loader(X, mask, static, y, np.array(sp["val"]),
                                   norm, 1024, False),
            "n": FT.agg_weight_for(y, tr),
            "pos_weight": FT.pos_weight_for(y, tr),
        }

    # Which parameters stay local. PERSONALIZE adds the classifier head;
    # PERSONAL_BRANCHES (if set) adds named branches.
    probe = build_model(C, n_features, n_static)
    LOCAL = set(FT.local_keys(probe))
    if C.PERSONALIZE:
        LOCAL |= set(probe.head_keys())
    heads = {n: None for n in names} if LOCAL else None
    bns = {}

    best, best_round, patience = -1.0, 0, 0
    for rnd in range(1, MAX_ROUNDS + 1):
        states, ws = [], []
        for n in names:
            m = build_model(C, n_features, n_static)
            m.load_state_dict(gstate)
            if C.FEDBN and n in bns:
                m.load_state_dict(bns[n], strict=False)
            if heads is not None and heads.get(n) is not None:
                m.load_state_dict(heads[n], strict=False)

            FT.local_train(m, loaders[n]["train"], C.LOCAL_EPOCHS, C.LR,
                           global_state=gstate if C.FL_ALGO == "fedprox" else None,
                           mu=C.PROX_MU if C.FL_ALGO == "fedprox" else 0.0,
                           pos_weight=loaders[n]["pos_weight"])

            sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
            if C.FEDBN:
                bk = [k for k in sd if "running_" in k or "num_batches" in k]
                for k in bk:
                    bns.setdefault(n, {})[k] = sd[k]
                sd = {k: v for k, v in sd.items() if k not in bk}
            if LOCAL:
                heads[n] = {k: sd[k] for k in LOCAL if k in sd}
                sd = {k: v for k, v in sd.items() if k not in LOCAL}
            states.append(sd)
            ws.append(loaders[n]["n"])

        agg = FT.fedavg(states, ws)
        merged = copy.deepcopy(gstate)
        merged.update(agg)
        gstate = sopt.step(gstate, merged) if sopt else merged

        zs, ys = [], []
        for n in names:
            m = build_model(C, n_features, n_static)
            m.load_state_dict(gstate)
            if C.FEDBN and n in bns:
                m.load_state_dict(bns[n], strict=False)
            if heads is not None and heads.get(n) is not None:
                m.load_state_dict(heads[n], strict=False)
            z, yy = FT.raw_logits(m, loaders[n]["val"])
            zs.append(z)
            ys.append(yy)
        yv = np.concatenate(ys)
        auprc = (MET.average_precision_score(
                    yv, 1 / (1 + np.exp(-np.concatenate(zs))))
                 if len(np.unique(yv)) > 1 else np.nan)

        if np.isfinite(auprc) and auprc > best:
            best, best_round, patience = auprc, rnd, 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    FT.free_gpu()
    return best, best_round


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------
def _worker_init():
    """Load the dataset once per worker process, not once per job."""
    global _W
    _W["D"] = FT.load_data()
    if torch.cuda.is_available():
        torch.cuda.init()


def _worker(job):
    D = _W.get("D") or FT.load_data()
    X, mask, static, y, part, nf, ns = D[0], D[1], D[2], D[3], D[4], D[5], D[6]
    apply(job["cfg"])
    score, rnd = fit_and_score(X, mask, static, y, part, nf, ns, job["seed"])
    return {**job["cfg"], "seed": job["seed"],
            "val_auprc": score, "best_round": rnd, "_gid": job.get("_gid", 0)}


def run_jobs(jobs, label=""):
    n = min(N_JOBS, len(jobs)) if N_JOBS > 1 else 1
    if n > 1 and len(jobs) < 2 * n:
        n = 1
    t0 = time.time()
    if n <= 1:
        out = []
        for i, j in enumerate(jobs, 1):
            out.append(_worker(j))
            print(f"    [{i}/{len(jobs)}] val AUPRC {out[-1]['val_auprc']:.4f}")
        return out

    import multiprocessing as mp
    ctx = mp.get_context("spawn")        # required for CUDA, and on Windows
    print(f"  {label}: {len(jobs)} jobs on {n} workers ...")
    try:
        with ctx.Pool(n, initializer=_worker_init) as pool:
            out = []
            for i, r in enumerate(pool.imap_unordered(_worker, jobs), 1):
                out.append(r)
                if i % max(1, len(jobs) // 15) == 0 or i == len(jobs):
                    el = time.time() - t0
                    print(f"    [{i:>3}/{len(jobs)}] best so far "
                          f"{max(x['val_auprc'] for x in out):.4f} | "
                          f"{el/60:.0f}m elapsed, "
                          f"~{el/i*(len(jobs)-i)/60:.0f}m left")
        return out
    except Exception as e:
        print(f"  [WARN] parallel run failed ({e}); falling back to serial")
        return [_worker(j) for j in jobs]


# ---------------------------------------------------------------------------
def main():
    global N_JOBS
    for a in sys.argv[1:]:
        if a.startswith("--jobs="):
            N_JOBS = int(a.split("=")[1])

    print(f"Parallel workers: {N_JOBS}")
    print("Selection uses the VALIDATION split only. The test set is not read.\n")
    D = FT.load_data()
    print(f"  {len(D[3]):,} stays | {len(D[4])} clients | "
          f"{D[5]} series features | {D[6]} tabular features\n")

    rng = np.random.default_rng(C.SEED)
    keys = list(SPACE)
    saved = {k: getattr(C, k) for k in keys}

    # ---------------- Stage 1 ------------------------------------------
    print("=" * 72)
    print(f"STAGE 1 — random search, {N_STAGE1} configs x 1 seed")
    print("=" * 72)
    jobs = [{"cfg": sample_config(rng), "seed": C.SEED} for _ in range(N_STAGE1)]
    s1 = pd.DataFrame(run_jobs(jobs, "stage 1")).sort_values(
        "val_auprc", ascending=False).reset_index(drop=True)
    s1.to_csv(C.OUT_DIR / "tuning_stage1.csv", index=False)
    print(f"\n  best single-seed val AUPRC: {s1.val_auprc.max():.4f}")

    # ---------------- Stage 2 ------------------------------------------
    print("\n" + "=" * 72)
    print(f"STAGE 2 — top {TOP_K} configs x {len(STAGE2_SEEDS)} seeds")
    print("  (a single-seed winner is usually a lucky seed, not a better config)")
    print("=" * 72)
    top = s1.head(TOP_K).reset_index(drop=True)
    jobs = [{"cfg": {k: top.loc[i, k] for k in keys}, "seed": sd, "_gid": i}
            for i in range(len(top)) for sd in STAGE2_SEEDS]
    r2 = pd.DataFrame(run_jobs(jobs, "stage 2"))

    rows = []
    for i in range(len(top)):
        sc = r2[r2._gid == i].val_auprc.values
        rows.append({**{k: top.loc[i, k] for k in keys},
                     "mean_val_auprc": float(np.mean(sc)),
                     "std": float(np.std(sc)),
                     "stage1_auprc": float(top.loc[i, "val_auprc"]),
                     "scores": [round(float(x), 4) for x in sc]})
        print(f"  #{i+1}: {np.mean(sc):.4f} +/- {np.std(sc):.4f}  "
              f"(stage1 said {top.loc[i,'val_auprc']:.4f}) | "
              f"{top.loc[i,'FL_ALGO']} imb={top.loc[i,'IMBALANCE']} "
              f"pers={top.loc[i,'PERSONALIZE']}")

    s2 = pd.DataFrame(rows)
    # rank by mean minus one SD: good AND reproducible
    s2["score"] = s2.mean_val_auprc - s2["std"]
    s2 = s2.sort_values("score", ascending=False).reset_index(drop=True)
    s2.to_csv(C.OUT_DIR / "tuning_stage2.csv", index=False)
    best = s2.iloc[0]
    print(f"\n  WINNER: {best.mean_val_auprc:.4f} +/- {best['std']:.4f}")
    if abs(best.stage1_auprc - s1.val_auprc.max()) > 1e-9:
        print("  (stage 1's top config was NOT the winner -- exactly the")
        print("   seed-luck this stage exists to catch)")

    # ---------------- Stage 3 ------------------------------------------
    print("\n" + "=" * 72)
    print(f"STAGE 3 — winner x {len(STAGE3_SEEDS)} seeds")
    print("=" * 72)
    cfg = {k: best[k] for k in keys}
    final = [r["val_auprc"] for r in run_jobs(
        [{"cfg": cfg, "seed": sd} for sd in STAGE3_SEEDS], "stage 3")]
    print(f"  val AUPRC over {len(STAGE3_SEEDS)} seeds: "
          f"{np.mean(final):.4f} +/- {np.std(final):.4f}")
    print(f"  individual: {[round(float(x),4) for x in final]}")

    out = {}
    for k in keys:
        v = cfg[k]
        out[k] = (bool(v) if k in BOOL_KEYS else
                  int(v) if k in INT_KEYS else
                  float(v) if isinstance(v, (float, np.floating)) else str(v))
    out["val_auprc_mean"] = float(np.mean(final))
    out["val_auprc_std"] = float(np.std(final))
    out["seeds"] = STAGE3_SEEDS
    (C.OUT_DIR / "tuning_best.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 72)
    print("BEST CONFIGURATION — paste into config.py")
    print("=" * 72)
    for k in keys:
        print(f"  {k} = {out[k]!r}")
    print("=" * 72)
    print("\nNow run `python fed_train.py` to read the test set, ONCE.")
    print("Tuning further after seeing test performance makes that estimate")
    print("biased; you would need a fresh holdout to recover.")

    for k, v in saved.items():
        setattr(C, k, v)


if __name__ == "__main__":
    main()