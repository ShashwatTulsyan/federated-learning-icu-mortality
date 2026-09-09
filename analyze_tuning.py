"""
Analyse the tuning run — which hyperparameters actually matter?

The winning config is the least interesting output of a search. The 60 configs in
`tuning_stage1.csv` are a designed experiment over the hyperparameter space, and
they can answer questions the winner cannot:

  * Does FedProx genuinely beat FedAvg on this data? (Directly addresses the
    Tertulino vs Dang et al. contradiction in your literature review.)
  * Does reducing LOCAL_EPOCHS reduce client drift, as theory predicts?
  * Is performance sensitive to learning rate, or flat?
  * Is ANY hyperparameter choice worth more than seed noise?

If the answer to the last question is "no", that is a legitimate and reportable
finding: it means your results are robust to configuration, which is a stronger
claim than a marginally higher number.

Run:  python analyze_tuning.py
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

import config as C

CAT = ["FL_ALGO", "IMBALANCE", "HIDDEN", "DROPOUT", "BATCH_SIZE", "LOCAL_EPOCHS"]
NUM = ["LR", "PROX_MU", "SERVER_LR"]


def cat_effect(df, col, metric="val_auprc"):
    g = df.groupby(col)[metric].agg(["count", "mean", "std", "max"]).round(4)
    groups = [d[metric].values for _, d in df.groupby(col) if len(d) >= 2]
    if len(groups) >= 2:
        try:
            F, p = stats.f_oneway(*groups)
        except Exception:
            F, p = np.nan, np.nan
        # eta-squared: share of score variance explained by this choice
        grand = df[metric].mean()
        ss_b = sum(len(g_) * (g_.mean() - grand) ** 2 for g_ in groups)
        ss_t = ((df[metric] - grand) ** 2).sum()
        eta2 = ss_b / ss_t if ss_t > 0 else np.nan
    else:
        F, p, eta2 = np.nan, np.nan, np.nan
    return g, F, p, eta2


def main():
    p1 = C.OUT_DIR / "tuning_stage1.csv"
    if not p1.exists():
        print(f"Not found: {p1}. Run tune.py first.")
        return
    df = pd.read_csv(p1)
    m = "val_auprc"
    pd.set_option("display.width", 200)

    print("=" * 74)
    print(f"SEARCH OVERVIEW — {len(df)} configurations")
    print("=" * 74)
    print(f"  best   {df[m].max():.4f}")
    print(f"  median {df[m].median():.4f}")
    print(f"  worst  {df[m].min():.4f}")
    print(f"  IQR    {df[m].quantile(.25):.4f} - {df[m].quantile(.75):.4f}")
    print(f"  spread between best and median: {df[m].max()-df[m].median():.4f}")

    # a reference for "how big is a real difference?"
    seed_sd = None
    p3 = C.OUT_DIR / "tuning_best.json"
    if p3.exists():
        seed_sd = json.loads(p3.read_text()).get("val_auprc_std")
        if seed_sd:
            print(f"\n  seed-to-seed SD of the winner: {seed_sd:.4f}")
            print(f"  -> any difference smaller than ~{2*seed_sd:.4f} is noise")

    print("\n" + "=" * 74)
    print("WHICH HYPERPARAMETERS MATTER?")
    print("  eta-squared = share of score variance explained by that choice")
    print("=" * 74)
    summary = []
    for col in CAT:
        if col not in df.columns:
            continue
        g, F, p, eta2 = cat_effect(df, col, m)
        summary.append({"parameter": col, "eta_squared": round(eta2, 4) if eta2 == eta2 else np.nan,
                        "anova_p": round(p, 4) if p == p else np.nan,
                        "best_level": g["mean"].idxmax(),
                        "spread_of_means": round(g["mean"].max()-g["mean"].min(), 4)})
    for col in NUM:
        if col not in df.columns or df[col].nunique() < 5:
            continue
        sub = df[df[col] > 0]
        if len(sub) < 5:
            continue
        r, p = stats.spearmanr(np.log(sub[col]), sub[m])
        summary.append({"parameter": f"{col} (log)", "eta_squared": round(r**2, 4),
                        "anova_p": round(p, 4), "best_level": "—",
                        "spread_of_means": np.nan})

    s = pd.DataFrame(summary).sort_values("eta_squared", ascending=False)
    print(s.to_string(index=False))
    print("\n  Rule of thumb: eta-squared < 0.06 is a small effect, > 0.14 large.")

    print("\n" + "=" * 74)
    print("FEDERATED ALGORITHM COMPARISON  (your Tertulino-vs-Dang question)")
    print("=" * 74)
    if "FL_ALGO" in df.columns:
        g, F, p, eta2 = cat_effect(df, "FL_ALGO", m)
        print(g.to_string())
        print(f"\n  ANOVA across algorithms: F={F:.3f}, p={p:.4f}, eta2={eta2:.4f}")
        algs = {k: d[m].values for k, d in df.groupby("FL_ALGO")}
        keys = sorted(algs)
        n_pairs = len(keys) * (len(keys) - 1) // 2
        alpha_bonf = 0.05 / max(n_pairs, 1)
        print(f"\n  pairwise (Welch t-test; Bonferroni alpha = {alpha_bonf:.4f} "
              f"for {n_pairs} comparisons):")
        sig_pairs = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = algs[keys[i]], algs[keys[j]]
                t, pv = stats.ttest_ind(a, b, equal_var=False)
                d = a.mean() - b.mean()
                pooled = np.sqrt((a.var() + b.var()) / 2)
                dd = d / pooled if pooled > 0 else 0.0
                flag = ""
                if pv < alpha_bonf:
                    flag = "  ** SIGNIFICANT (Bonferroni)"
                    sig_pairs.append((keys[i], keys[j], d, dd, pv))
                elif pv < 0.05:
                    flag = "  * nominal only, fails Bonferroni"
                print(f"    {keys[i]:<9} vs {keys[j]:<9} "
                      f"diff {d:+.4f}  Cohen's d {dd:+.2f}  p={pv:.3f}{flag}")

        # An omnibus ANOVA can hide a real pairwise difference when one group is
        # highly variable -- that group inflates the within-group term. Always
        # check the pairwise results before concluding "no difference".
        if sig_pairs:
            print("\n  -> The omnibus ANOVA is not significant, but a PAIRWISE")
            print("     comparison is. This happens when one algorithm has high")
            print("     variance across configs, inflating the within-group term.")
            for a_, b_, d, dd, pv in sig_pairs:
                better, worse = (a_, b_) if d > 0 else (b_, a_)
                print(f"\n     {better} beats {worse}: diff {abs(d):.4f}, "
                      f"|d| = {abs(dd):.2f}, p = {pv:.4f}")
            print("\n     REPORT THIS. A significant pairwise difference with a")
            print("     large effect size is a finding, not a null result.")
        elif p > 0.05:
            print("\n  -> No significant difference between aggregation algorithms,")
            print("     pairwise or overall. That is itself reportable: on a")
            print("     naturally-partitioned MIMIC-IV cohort the aggregation rule")
            print("     does not materially change performance.")
        else:
            print(f"\n  -> Significant difference. Best: {g['mean'].idxmax()}")

        # stability matters as much as the mean when picking an algorithm
        print("\n  stability (SD across sampled configs -- lower = less")
        print("  sensitive to the other hyperparameters):")
        for k in sorted(algs, key=lambda x: algs[x].std()):
            print(f"    {k:<9} SD {algs[k].std():.4f}  mean {algs[k].mean():.4f}")

    print("\n" + "=" * 74)
    print("CLIENT-DRIFT HYPOTHESIS  (does fewer local epochs help?)")
    print("=" * 74)
    if "LOCAL_EPOCHS" in df.columns:
        g, F, p, eta2 = cat_effect(df, "LOCAL_EPOCHS", m)
        print(g.to_string())
        r, pr = stats.spearmanr(df.LOCAL_EPOCHS, df[m])
        print(f"\n  Spearman(LOCAL_EPOCHS, score) = {r:+.3f}, p={pr:.4f}")
        if pr > 0.05:
            print("  -> No detectable relationship. The prediction that fewer local")
            print("     epochs would narrow the federated-centralized gap is NOT")
            print("     supported here. Report that honestly.")
        elif r < 0:
            print("  -> Fewer local epochs DO score better, consistent with the")
            print("     client-drift explanation.")
        else:
            print("  -> MORE local epochs score better, contrary to the drift")
            print("     prediction. Worth investigating.")

    print("\n" + "=" * 74)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 74)
    cols = [c for c in ["val_auprc", "FL_ALGO", "LR", "HIDDEN", "DROPOUT",
                        "BATCH_SIZE", "LOCAL_EPOCHS", "IMBALANCE", "best_round"]
            if c in df.columns]
    print(df.nlargest(10, m)[cols].to_string(index=False))

    if "best_round" in df.columns:
        cap = df.best_round.max()
        n_at_cap = (df.best_round >= cap).sum()
        print(f"\n  best_round: max {cap}, {n_at_cap} config(s) at the cap")
        if n_at_cap > len(df) * 0.2:
            print("  [WARN] many configs stopped at MAX_ROUNDS -- the cap is binding")
            print("         and you are under-training. Raise MAX_ROUNDS in tune.py.")

    print("\n" + "=" * 74)
    print("WHAT TO DO WITH THIS")
    print("=" * 74)
    span = df[m].max() - df[m].median()
    if seed_sd and span < 3 * seed_sd:
        print(f"  The best config beats the median by {span:.4f}, against a")
        print(f"  seed-to-seed SD of {seed_sd:.4f}. Configuration choice is worth")
        print("  little more than seed noise on this problem.")
        print("\n  That is a good result to report: it means your findings are")
        print("  robust to hyperparameters rather than dependent on a lucky")
        print("  setting. Prefer the SIMPLEST config among the leaders, not the")
        print("  nominal winner -- simpler is easier to justify and to reproduce.")
    else:
        print("  Configuration choice matters more than seed noise. Use the tuned")
        print("  config, and report the search in the methods section.")
    print("\n  Before adopting a new config, re-run your CURRENT one with the same")
    print("  seeds and compare like with like. A change is only worth making if it")
    print("  survives that comparison.")


if __name__ == "__main__":
    main()
