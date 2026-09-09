"""
Controlled head-to-head comparison of aggregation algorithms.

Why this is needed even though tune.py already compared them
------------------------------------------------------------
The comparison inside a random search is OBSERVATIONAL. The FedAvg configs and
the FedProx configs also differed in learning rate, hidden size, dropout and
batch size, so part of any gap could be confounding rather than the algorithm.
That is fine for generating a hypothesis, not for claiming one.

This script runs a CONTROLLED experiment: one base configuration, every
hyperparameter held fixed, only the aggregation rule varied, and the same seeds
used for every arm so comparisons are PAIRED. Pairing removes seed variance from
the comparison, which matters a lot when the effect is ~0.007 and seed SD is
~0.006.

It also sweeps the FedProx proximal strength properly. In the random search mu
was sampled jointly with everything else, so FedProx may simply have drawn poor
mu values. Giving it several mu settings is the fair test.

Selection still happens on VALIDATION only. The test set is untouched.

Run:  python compare_algorithms.py
Out:  artifacts/algorithm_comparison.csv
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

import config as C
import fed_train as FT
import tune

SEEDS = [42, 101, 2026, 7, 1234]

# Base configuration: the best-performing setting from the search. Every arm
# below uses exactly this, changing only the aggregation rule.
BASE = {
    "LR": 0.002065,
    "HIDDEN": 128,
    "DROPOUT": 0.4,
    "BATCH_SIZE": 128,
    "LOCAL_EPOCHS": 3,
    "IMBALANCE": "pos_weight",
}

ARMS = [
    ("FedAvg",            {"FL_ALGO": "fedavg",  "PROX_MU": 0.0}),
    ("FedProx mu=0.001",  {"FL_ALGO": "fedprox", "PROX_MU": 0.001}),
    ("FedProx mu=0.01",   {"FL_ALGO": "fedprox", "PROX_MU": 0.01}),
    ("FedProx mu=0.1",    {"FL_ALGO": "fedprox", "PROX_MU": 0.1}),
    ("FedAdam slr=0.003", {"FL_ALGO": "fedadam", "SERVER_LR": 0.003}),
    ("FedAdam slr=0.01",  {"FL_ALGO": "fedadam", "SERVER_LR": 0.01}),
    ("FedAdam slr=0.03",  {"FL_ALGO": "fedadam", "SERVER_LR": 0.03}),
]


def main():
    print("Loading data ...")
    X, mask, static, y, part, n_features, n_static, _ = FT.load_data()
    print(f"  {len(y):,} stays | {len(part)} clients\n")
    print("Controlled comparison: identical config in every arm, only the")
    print(f"aggregation rule varies. Paired across {len(SEEDS)} seeds.\n")

    saved = {k: getattr(C, k, None) for k in
             ["LR", "HIDDEN", "DROPOUT", "BATCH_SIZE", "LOCAL_EPOCHS",
              "IMBALANCE", "FL_ALGO", "PROX_MU", "SERVER_LR"]}

    results = {}
    for name, override in ARMS:
        cfg = {**BASE, **override}
        cfg.setdefault("SERVER_LR", C.SERVER_LR)
        tune.apply(cfg)
        scores = []
        for sd in SEEDS:
            s, _ = tune.fit_and_score(X, mask, static, y, part,
                                      n_features, n_static, seed=sd)
            scores.append(s)
        results[name] = scores
        print(f"  {name:<20} {np.mean(scores):.4f} +/- {np.std(scores):.4f}   "
              f"{[round(x,4) for x in scores]}")

    df = pd.DataFrame(results, index=[f"seed_{s}" for s in SEEDS]).T
    df["mean"] = df.mean(axis=1)
    df["std"] = df[[c for c in df.columns if c.startswith("seed_")]].std(axis=1)
    df = df.sort_values("mean", ascending=False)
    df.to_csv(C.OUT_DIR / "algorithm_comparison.csv")

    print("\n" + "=" * 72)
    print("RANKING (validation AUPRC, controlled)")
    print("=" * 72)
    print(df[["mean", "std"]].round(4).to_string())

    # ---- paired tests against the leader --------------------------------
    ref = df.index[0]
    print("\n" + "=" * 72)
    print(f"PAIRED COMPARISONS vs {ref}")
    print("  Paired by seed, so seed variance cancels. This is the test that")
    print("  supports a causal claim about the algorithm.")
    print("=" * 72)
    a = np.array(results[ref])
    rows = []
    for name in df.index[1:]:
        b = np.array(results[name])
        diff = a - b
        if np.allclose(diff, 0):
            t, p, d, w_p = np.nan, np.nan, np.nan, np.nan
        else:
            t, p = stats.ttest_rel(a, b)
            # Cohen's d for paired data uses the SD of the differences
            sd_d = diff.std(ddof=1)
            d = diff.mean() / sd_d if sd_d > 0 else np.nan
            try:
                w_p = stats.wilcoxon(a, b).pvalue
            except Exception:
                w_p = np.nan
        if not np.isfinite(p):
            print(f"     vs {name:<20} identical scores -- no test possible")
        rows.append({"arm": name, "mean_diff": round(diff.mean(), 5),
                     "paired_t_p": round(p, 4),
                     "wilcoxon_p": round(w_p, 4) if w_p == w_p else np.nan,
                     "cohens_d": round(d, 2) if d == d else np.nan,
                     "wins": int((diff > 0).sum()), "of": len(SEEDS)})
        if np.isfinite(p):
            sig = "**" if p < 0.05 else "  "
            dstr = f"{d:+.2f}" if np.isfinite(d) else "  n/a"
            print(f"  {sig} vs {name:<20} diff {diff.mean():+.5f}  "
                  f"d={dstr}  paired-t p={p:.4f}  "
                  f"wins {int((diff>0).sum())}/{len(SEEDS)}")

    pd.DataFrame(rows).to_csv(C.OUT_DIR / "algorithm_paired_tests.csv", index=False)

    # ---- interpretation ---------------------------------------------------
    print("\n" + "=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    best_prox = max((k for k in results if k.startswith("FedProx")),
                    key=lambda k: np.mean(results[k]))
    fa = np.mean(results["FedAvg"])
    fp = np.mean(results[best_prox])
    print(f"  FedAvg              {fa:.4f}")
    print(f"  best FedProx ({best_prox.split()[-1]}) {fp:.4f}")
    print(f"  difference          {fa-fp:+.4f}")

    diff_fp = np.array(results["FedAvg"]) - np.array(results[best_prox])
    if np.allclose(diff_fp, 0):
        t, p = np.nan, np.nan
    else:
        t, p = stats.ttest_rel(results["FedAvg"], results[best_prox])

    if not np.isfinite(p):
        print("\n  -> The arms produced identical scores, so no test is possible.")
        print("     This normally means the run is degenerate (too few rounds,")
        print("     or a task the model saturates). Check the scores above are")
        print("     plausible before drawing any conclusion.")
    elif p < 0.05 and fa > fp:
        print(f"\n  -> FedAvg significantly beats FedProx even at its best mu")
        print(f"     (paired p = {p:.4f}). The proximal term does not help on")
        print("     this partition. Consistent with Dang et al. (ACM TIST), and")
        print("     contrary to Tertulino's MIMIC-IV care-unit result.")
    elif p >= 0.05:
        print(f"\n  -> No significant difference once mu is tuned (paired p = "
              f"{p:.4f}).")
        print("     The advantage seen in the random search was likely CONFOUNDED:")
        print("     FedProx drew poor mu values there. Report the controlled")
        print("     result, not the observational one -- and note that the")
        print("     apparent effect disappeared under control.")
    else:
        print(f"\n  -> FedProx beats FedAvg here (paired p = {p:.4f}).")

    print("\n  Note: with 5 seeds these tests have low power. Treat a")
    print("  non-significant result as 'not demonstrated', not 'no effect'.")
    print("  If the difference matters to your argument, raise SEEDS to 10.")

    for k, v in saved.items():
        if v is not None:
            setattr(C, k, v)


if __name__ == "__main__":
    main()
