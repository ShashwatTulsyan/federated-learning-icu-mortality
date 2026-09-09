"""
Generate the figures and tables for the manuscript.

Produces only what a clinical-informatics paper actually needs — four figures and
four tables — from artifacts already on disk. Nothing is retrained.

Figures (300 dpi PNG + vector PDF):
  Fig 1  Capacity series: FL gain over local vs model capacity        [headline]
  Fig 2  ROC and precision-recall curves, federated vs centralized
  Fig 3  Calibration curve + decision curve (clinical utility)
  Fig 4  Per-client performance vs outcome count

Tables (CSV + LaTeX):
  Table 1  Cohort characteristics
  Table 2  Main results with 95% CIs
  Table 3  Algorithm comparison
  Table 4  Ablation

Run:  python make_figures.py
Out:  artifacts/paper/
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# ---------------------------------------------------------------------------
# Style: greyscale-safe, colour-blind-safe, serif to match journal body text.
# Every series is distinguishable by marker and line style alone, because many
# readers print in black and white.
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 9.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "legend.frameon": False, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.4,
})
# Okabe-Ito palette
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#666666"
# Results directory. Defaults to config.OUT_DIR, but can be pointed elsewhere so
# several runs (24h vs 48h, restricted vs full cohort) can be kept side by side:
#     python make_figures.py artifacts_24h
SRC = C.OUT_DIR
OUT = SRC / "paper"
COL_W, TWO_COL = 3.42, 7.0        # inches: single and double journal column


def set_source(path):
    """Repoint every reader at `path`."""
    global SRC, OUT
    SRC = Path(path)
    OUT = SRC / "paper"


def pick_runs():
    """Run folders, most-recently-modified last.

    Sorting alphabetically would pick run_fedavg_* over run_fedadam_* regardless
    of which was actually produced last.
    """
    return sorted(SRC.glob("run_*"), key=lambda p: p.stat().st_mtime)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def read(name):
    p = SRC / name
    return pd.read_csv(p) if p.exists() else None


def num(s):
    """'0.842 (0.821-0.863)' -> 0.842"""
    try:
        return float(str(s).split(" ")[0])
    except Exception:
        return np.nan


def ci(s):
    """'0.842 (0.821-0.863)' -> (0.821, 0.863)"""
    try:
        lo, hi = str(s).split("(")[1].rstrip(")").split("-")
        return float(lo), float(hi)
    except Exception:
        return (np.nan, np.nan)


# ===========================================================================
# FIGURE 1 — the headline
# ===========================================================================
def fig1_capacity():
    raw = read("exp1_capacity_raw.csv")
    if raw is None:
        print("  [skip] Fig 1: exp1_capacity_raw.csv not found")
        return
    g = (raw.groupby("level")
            .agg(n_param=("n_param", "first"),
                 gain=("fl_gain_auprc", "mean"), gain_sd=("fl_gain_auprc", "std"),
                 gap=("gap_auprc", "mean"), gap_sd=("gap_auprc", "std"),
                 auprc=("auprc", "mean"), auprc_sd=("auprc", "std"))
            .reset_index().sort_values("n_param"))
    n = raw.groupby("level").size().min()
    se = lambda sd: 1.96 * sd / np.sqrt(n)      # 95% CI of the mean

    from scipy import stats
    rho_g, p_g = stats.spearmanr(raw.n_param, raw.fl_gain_auprc)
    rho_d, p_d = stats.spearmanr(raw.n_param, raw.gap_auprc)

    fig, ax = plt.subplots(1, 2, figsize=(TWO_COL, 2.7))
    x = g.n_param / 1000

    ax[0].errorbar(x, g.gain, yerr=se(g.gain_sd), marker="o", color=BLUE,
                   capsize=3, markersize=5, label="Federated − local-only")
    ax[0].errorbar(x, g.gap, yerr=se(g.gap_sd), marker="s", ls="--", color=ORANGE,
                   capsize=3, markersize=5, label="Centralized − federated")
    ax[0].axhline(0, color=GREY, lw=0.8, ls=":")
    ax[0].set_xlabel("Model parameters (thousands)")
    ax[0].set_ylabel("Δ AUPRC")
    ax[0].set_title("A  Benefit and cost of federation")
    ax[0].legend(loc="upper left")
    ax[0].text(0.97, 0.05,
               f"gain: ρ={rho_g:.2f}, p={p_g:.3f}\ngap:  ρ={rho_d:.2f}, p={p_d:.2f}",
               transform=ax[0].transAxes, ha="right", va="bottom", fontsize=7.5,
               bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GREY, lw=0.5))

    ax[1].errorbar(x, g.auprc, yerr=se(g.auprc_sd), marker="o", color=BLUE,
                   capsize=3, markersize=5)
    ax[1].set_xlabel("Model parameters (thousands)")
    ax[1].set_ylabel("AUPRC (federated)")
    ax[1].set_title("B  Absolute performance")
    for _, r in g.iterrows():
        ax[1].annotate(r.level.split()[0], (r.n_param/1000, r.auprc),
                       textcoords="offset points", xytext=(0, 7),
                       ha="center", fontsize=7, color=GREY)
    # matplotlib's auto-ticker can pick unevenly spaced values here, which reads
    # as a distorted axis. Force a uniform grid.
    from matplotlib.ticker import MaxNLocator
    ax[1].yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    save(fig, "fig1_capacity")


# ===========================================================================
# FIGURE 2 — discrimination
# ===========================================================================
def fig2_curves():
    runs = pick_runs()
    if not runs:
        print("  [skip] Fig 2: no run_* directory (run fed_train.py)")
        return
    d = runs[-1]
    roc, pr = d / "roc_points.csv", d / "pr_points.csv"
    if not roc.exists():
        print("  [skip] Fig 2: curve files missing")
        return
    R, P = pd.read_csv(roc), pd.read_csv(pr)

    m = pd.read_csv(d / "metrics_formatted.csv")
    gl = m[m.scope == "global"]
    fed = gl[gl.method.str.contains("federated")].iloc[0]
    prev = fed.events / fed.N

    fig, ax = plt.subplots(1, 2, figsize=(TWO_COL, 3.0))
    ax[0].plot(R.fpr, R.tpr, color=BLUE)
    ax[0].plot([0, 1], [0, 1], ls=":", color=GREY, lw=0.9)
    ax[0].set_xlabel("1 − specificity")
    ax[0].set_ylabel("Sensitivity")
    ax[0].set_title("A  Receiver operating characteristic")
    ax[0].text(0.95, 0.08, f"AUROC {fed['AUROC (95% CI)']}",
               transform=ax[0].transAxes, ha="right", fontsize=8)
    ax[0].set_aspect("equal")

    ax[1].plot(P.recall, P.precision, color=BLUE)
    ax[1].axhline(prev, ls=":", color=GREY, lw=0.9)
    ax[1].text(0.98, prev + 0.02, f"prevalence {prev*100:.1f}%",
               ha="right", fontsize=7.5, color=GREY)
    ax[1].set_xlabel("Recall (sensitivity)")
    ax[1].set_ylabel("Precision (PPV)")
    ax[1].set_title("B  Precision–recall")
    ax[1].text(0.95, 0.9, f"AUPRC {fed['AUPRC (95% CI)']}",
               transform=ax[1].transAxes, ha="right", fontsize=8)
    ax[1].set_ylim(0, 1)
    save(fig, "fig2_roc_pr")


# ===========================================================================
# FIGURE 3 — calibration and clinical utility
# ===========================================================================
def fig3_calibration():
    runs = pick_runs()
    if not runs:
        print("  [skip] Fig 3: no run_* directory")
        return
    d = runs[-1]
    cal, dec = d / "calibration_curve.csv", d / "decision_curve.csv"
    if not cal.exists():
        print("  [skip] Fig 3: calibration/decision files missing")
        return
    Cv, D = pd.read_csv(cal), pd.read_csv(dec)

    fig, ax = plt.subplots(1, 2, figsize=(TWO_COL, 3.0))
    lim = max(Cv.mean_pred.max(), Cv.observed_rate.max()) * 1.1
    ax[0].plot([0, lim], [0, lim], ls=":", color=GREY, lw=0.9,
               label="Perfect calibration")
    ax[0].plot(Cv.mean_pred, Cv.observed_rate, marker="o", color=BLUE,
               markersize=4, label="Federated model")
    ax[0].set_xlabel("Predicted probability")
    ax[0].set_ylabel("Observed frequency")
    ax[0].set_title("A  Calibration")
    ax[0].legend(loc="upper left")
    ax[0].set_xlim(0, lim)
    ax[0].set_ylim(0, lim)

    ax[1].plot(D.threshold, D.net_benefit_model, color=BLUE, label="Model")
    ax[1].plot(D.threshold, D.net_benefit_treat_all, ls="--", color=ORANGE,
               label="Treat all")
    ax[1].plot(D.threshold, D.net_benefit_treat_none, ls=":", color=GREY,
               label="Treat none")
    ax[1].set_xlabel("Threshold probability")
    ax[1].set_ylabel("Net benefit")
    ax[1].set_title("B  Decision curve")
    ax[1].legend()
    ax[1].set_ylim(min(-0.02, D.net_benefit_model.min()),
                   D.net_benefit_model.max() * 1.25 + 1e-6)
    save(fig, "fig3_calibration_dca")


# ===========================================================================
# FIGURE 4 — per-client results
# ===========================================================================
def fig4b_gain_by_seed():
    """Federated minus local-only, per seed and capacity level.

    Fig 4 shows one seed at one capacity level, so it can look inconsistent with
    a claim like '22/25 runs favour federation'. This panel shows every run
    behind that claim, which is what the abstract actually refers to.
    """
    raw = read("exp1_capacity_raw.csv")
    if raw is None:
        print("  [skip] Fig 4b: exp1_capacity_raw.csv not found")
        return
    order = (raw.groupby("level").n_param.first().sort_values().index.tolist())
    fig, ax = plt.subplots(figsize=(COL_W * 1.5, 2.8))
    for i, lvl in enumerate(order):
        v = raw[raw.level == lvl].fl_gain_auprc.values
        ax.scatter(np.full(len(v), i) + np.linspace(-.12, .12, len(v)), v,
                   s=22, color=BLUE, zorder=3, alpha=.85)
        ax.hlines(v.mean(), i - .28, i + .28, color=ORANGE, lw=2, zorder=4)
    ax.axhline(0, color=GREY, lw=0.9, ls=":")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([l.split()[0] for l in order])
    ax.set_xlabel("Feature set (increasing model capacity)")
    ax.set_ylabel("Δ AUPRC (federated − local-only)")
    n_pos = int((raw.fl_gain_auprc > 0).sum())
    ax.set_title(f"Federated vs local-only across all runs "
                 f"({n_pos}/{len(raw)} favour federation)")
    ax.plot([], [], color=ORANGE, lw=2, label="Mean")
    ax.scatter([], [], s=22, color=BLUE, label="Individual seed")
    ax.legend(loc="upper left")
    save(fig, "fig4b_gain_all_runs")


def fig4_per_client():
    runs = pick_runs()
    if not runs:
        print("  [skip] Fig 4: no run_* directory")
        return
    m = pd.read_csv(runs[-1] / "metrics_formatted.csv")
    fed = m[(m.scope != "global") & m.method.str.contains("federated")]
    loc = m[m.method.str.contains("local-only")].set_index("scope")
    if fed.empty:
        print("  [skip] Fig 4: no per-client rows")
        return

    rows = []
    for _, r in fed.iterrows():
        if r.scope not in loc.index:
            continue
        lo, hi = ci(r["AUPRC (95% CI)"])
        rows.append({"client": r.scope, "events": int(r.events),
                     "fed": num(r["AUPRC (95% CI)"]), "lo": lo, "hi": hi,
                     "local": num(loc.loc[r.scope, "AUPRC (95% CI)"])})
    t = pd.DataFrame(rows).sort_values("events")

    short = {"Medical Intensive Care Unit (MICU)": "MICU",
             "Cardiac Vascular Intensive Care Unit (CVICU)": "CVICU",
             "Medical/Surgical Intensive Care Unit (MICU/SICU)": "MICU/SICU",
             "Surgical Intensive Care Unit (SICU)": "SICU",
             "Coronary Care Unit (CCU)": "CCU",
             "Trauma SICU (TSICU)": "TSICU",
             "Neuro Intermediate": "Neuro Int."}
    lab = [short.get(c, c[:14]) for c in t.client]
    yy = np.arange(len(t))

    fig, ax = plt.subplots(figsize=(COL_W * 1.5, 2.9))
    ax.errorbar(t.fed, yy, xerr=[t.fed - t.lo, t.hi - t.fed], fmt="o",
                color=BLUE, capsize=3, markersize=5, label="Federated (95% CI)")
    ax.plot(t.local, yy, marker="s", ls="none", color=ORANGE, markersize=5,
            label="Local-only")
    for i, r in enumerate(t.itertuples()):
        ax.annotate(f"{r.events} ev.", (max(r.hi, r.local), i),
                    textcoords="offset points", xytext=(5, 0),
                    va="center", fontsize=7, color=GREY)
    ax.set_yticks(yy)
    ax.set_yticklabels(lab)
    ax.set_xlabel("AUPRC")
    ax.set_title("Per-client performance, single run\n(ordered by outcome count)",
                 fontsize=9)
    ax.legend(loc="lower right")
    # leave room for the event-count annotations on the right
    right = max(t.hi.max(), t.local.max())
    ax.set_xlim(0, right + max(0.16, right * 0.30))
    save(fig, "fig4_per_client")


# ===========================================================================
# TABLES
# ===========================================================================
def to_latex(df, name, caption):
    try:
        body = df.to_latex(index=False, escape=True, column_format=None,
                           caption=caption, label=f"tab:{name}")
    except TypeError:
        body = df.to_latex(index=False, escape=True)
    (OUT / f"{name}.tex").write_text(body)


def tables():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = sorted(SRC.glob("run_*"))

    # Table 1 — cohort
    if runs and (runs[-1] / "table1.csv").exists():
        t1 = pd.read_csv(runs[-1] / "table1.csv")
        t1.to_csv(OUT / "table1_cohort.csv", index=False)
        to_latex(t1, "table1_cohort", "Cohort characteristics.")
        print("  table1_cohort.csv / .tex")

    # Table 2 — main results
    if runs and (runs[-1] / "metrics_formatted.csv").exists():
        m = pd.read_csv(runs[-1] / "metrics_formatted.csv")
        keep = ["method", "scope", "N", "events", "AUROC (95% CI)",
                "AUPRC (95% CI)", "Sensitivity (95% CI)",
                "Specificity (95% CI)", "Brier (95% CI)", "Calib. slope"]
        t2 = m[[c for c in keep if c in m.columns]]
        t2.to_csv(OUT / "table2_main_results.csv", index=False)
        to_latex(t2[t2.scope == "global"], "table2_main_results",
                 "Global discrimination and calibration, with 95\\% bootstrap CIs.")
        print("  table2_main_results.csv / .tex")

    # Table 3 — algorithms
    a, ap = read("exp2_algorithms.csv"), read("exp2_algorithms_paired.csv")
    if a is not None:
        if ap is not None:
            a = a.merge(ap[["arm", "mean_diff", "paired_p", "wins", "of"]],
                        left_on="algorithm", right_on="arm", how="left").drop(
                        columns=["arm"])
        a = a.sort_values("auprc_mean", ascending=False)
        a.to_csv(OUT / "table3_algorithms.csv", index=False)
        to_latex(a, "table3_algorithms",
                 "Aggregation algorithms, mean $\\pm$ SD across seeds, "
                 "paired against FedAvg.")
        print("  table3_algorithms.csv / .tex")

    # Table 4 — ablation
    e, epr = read("exp4_ablation.csv"), read("exp4_ablation_paired.csv")
    if e is not None:
        if epr is not None:
            e = e.merge(epr[["arm", "mean_diff", "paired_p", "wins", "of"]],
                        left_on="variant", right_on="arm", how="left").drop(
                        columns=["arm"])
        e = e.sort_values("auprc_mean", ascending=False)
        e.to_csv(OUT / "table4_ablation.csv", index=False)
        to_latex(e, "table4_ablation",
                 "Component ablation, paired against the full model.")
        print("  table4_ablation.csv / .tex")


# ===========================================================================
def summary():
    """The numbers most likely to be quoted in the abstract."""
    runs = sorted(SRC.glob("run_*"))
    lines = []
    if runs and (runs[-1] / "metrics_formatted.csv").exists():
        m = pd.read_csv(runs[-1] / "metrics_formatted.csv")
        gl = m[m.scope == "global"]
        for _, r in gl.iterrows():
            prev = r.events / r.N
            lines.append(
                f"{r.method}: AUROC {r['AUROC (95% CI)']}, "
                f"AUPRC {r['AUPRC (95% CI)']} "
                f"(prevalence {prev*100:.1f}%, "
                f"lift {num(r['AUPRC (95% CI)'])/prev:.1f}x), "
                f"Brier {r['Brier (95% CI)']}")
    raw = read("exp1_capacity_raw.csv")
    if raw is not None:
        from scipy import stats
        rho, p = stats.spearmanr(raw.n_param, raw.fl_gain_auprc)
        lines.append(f"Capacity vs FL gain: Spearman rho={rho:.3f}, p={p:.4f} "
                     f"(n={len(raw)} runs)")
        lines.append(f"Federated beat local-only in "
                     f"{int((raw.fl_gain_auprc>0).sum())}/{len(raw)} runs")
    txt = "\n".join(lines)
    (OUT / "abstract_numbers.txt").write_text(txt)
    print("\n--- numbers for the abstract ---")
    print(txt)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        set_source(args[0])

    if not SRC.exists():
        print(f"Results directory not found: {SRC.resolve()}")
        print("\nUsage: python make_figures.py [results_dir]")
        print("  e.g. python make_figures.py artifacts_24h")
        sys.exit(1)

    have_exp = list(SRC.glob("exp*.csv"))
    runs = pick_runs()
    print(f"Reading from {SRC.resolve()}")
    print(f"  experiment files : {len(have_exp)}")
    print(f"  run folders      : {[r.name for r in runs] or 'NONE'}")
    if runs:
        print(f"  using            : {runs[-1].name} (most recent)")
    if not have_exp and not runs:
        print("\nNothing to plot. Point this at the directory containing")
        print("exp*.csv and run_*/ , e.g.:")
        print("    python make_figures.py artifacts_24h")
        sys.exit(1)

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting to {OUT}\n")
    print("Figures:")
    fig1_capacity()
    fig2_curves()
    fig3_calibration()
    fig4_per_client()
    fig4b_gain_by_seed()
    print("\nTables:")
    tables()
    summary()
    print(f"\nAll outputs in {OUT}")
    print("PNG for drafts, PDF (vector) for submission.")


if __name__ == "__main__":
    main()