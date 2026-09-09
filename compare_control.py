"""
Compare the care-unit capacity series against the size-matched IID control.

This is the mechanism test. If federation's benefit grows with capacity BECAUSE
clients are heterogeneous, then removing heterogeneity while holding client sizes
fixed should weaken or abolish the relationship. If the relationship survives,
the mechanism is local sample size alone and the claim must be reworded.

Reads:  artifacts/exp1_capacity_raw.csv            (care units, heterogeneous)
        artifacts/exp1_capacity_shuffled_raw.csv   (IID control)
Writes: artifacts/paper/fig5_iid_control.{png,pdf}
        artifacts/paper/table5_control_comparison.csv

Run:  python compare_control.py [results_dir]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import config as C

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.4,
})
BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#666666"


def spearman(df, col):
    r, p = stats.spearmanr(df.n_param, df[col])
    return r, p


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else C.OUT_DIR
    out = src / "paper"
    out.mkdir(parents=True, exist_ok=True)

    f_het = src / "exp1_capacity_raw.csv"
    f_iid = src / "exp1_capacity_shuffled_raw.csv"
    if not f_het.exists():
        print(f"Missing {f_het} — run the care-unit E1 first.")
        sys.exit(1)
    if not f_iid.exists():
        print(f"Missing {f_iid}.")
        print("Run the control:  set PARTITION_SCHEME='shuffled' in config.py,")
        print("then  python partition.py  and  python run_experiments.py e1 --jobs=8")
        sys.exit(1)

    het = pd.read_csv(f_het)
    iid = pd.read_csv(f_iid)
    print(f"  care units (heterogeneous): {len(het)} runs")
    print(f"  shuffled (IID control)    : {len(iid)} runs\n")

    rows = []
    for name, d in (("Care units (heterogeneous)", het), ("Shuffled (IID control)", iid)):
        rg, pg = spearman(d, "fl_gain_auprc")
        rd, pd_ = spearman(d, "gap_auprc")
        rows.append({
            "partition": name,
            "n_runs": len(d),
            "rho_gain": round(rg, 3), "p_gain": round(pg, 4),
            "rho_gap": round(rd, 3), "p_gap": round(pd_, 4),
            "mean_gain": round(d.fl_gain_auprc.mean(), 4),
            "gain_lowest_capacity": round(
                d[d.n_param == d.n_param.min()].fl_gain_auprc.mean(), 4),
            "gain_highest_capacity": round(
                d[d.n_param == d.n_param.max()].fl_gain_auprc.mean(), 4),
            "wins": f"{int((d.fl_gain_auprc > 0).sum())}/{len(d)}",
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "table5_control_comparison.csv", index=False)
    pd.set_option("display.width", 220)
    print(tab.to_string(index=False))

    # --- figure -----------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for a, (d, name, col, mk, ls) in zip(
            [ax[0], ax[0]],
            [(het, "Care units (heterogeneous)", BLUE, "o", "-"),
             (iid, "Shuffled (IID control)", ORANGE, "s", "--")]):
        g = d.groupby("n_param").fl_gain_auprc.agg(["mean", "std", "count"])
        se = 1.96 * g["std"] / np.sqrt(g["count"])
        a.errorbar(g.index / 1000, g["mean"], yerr=se, marker=mk, ls=ls,
                   color=col, capsize=3, markersize=5, label=name)
    ax[0].axhline(0, color=GREY, lw=0.8, ls=":")
    ax[0].set_xlabel("Model parameters (thousands)")
    ax[0].set_ylabel("Δ AUPRC (federated − local-only)")
    ax[0].set_title("A  Federation's benefit vs capacity")
    ax[0].legend(loc="upper left")
    rg_h, pg_h = spearman(het, "fl_gain_auprc")
    rg_i, pg_i = spearman(iid, "fl_gain_auprc")
    ax[0].text(0.97, 0.04,
               f"heterogeneous: ρ={rg_h:.2f}, p={pg_h:.3f}\n"
               f"IID control:   ρ={rg_i:.2f}, p={pg_i:.3f}",
               transform=ax[0].transAxes, ha="right", va="bottom", fontsize=7.5,
               bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GREY, lw=0.5))

    # per-run distribution
    for i, (d, name, col) in enumerate(
            [(het, "Heterogeneous", BLUE), (iid, "IID control", ORANGE)]):
        x = np.full(len(d), i) + np.linspace(-0.16, 0.16, len(d))
        ax[1].scatter(x, d.fl_gain_auprc, s=16, color=col, alpha=0.8, zorder=3)
        ax[1].hlines(d.fl_gain_auprc.mean(), i - 0.3, i + 0.3,
                     color="black", lw=2, zorder=4)
    ax[1].axhline(0, color=GREY, lw=0.8, ls=":")
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["Care units", "Shuffled\n(IID)"])
    ax[1].set_ylabel("Δ AUPRC (federated − local-only)")
    ax[1].set_title("B  All runs, both partitions")

    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig5_iid_control.{ext}")
    plt.close(fig)
    print(f"\n  fig5_iid_control.png / .pdf")
    print(f"  table5_control_comparison.csv")

    # --- does the difference between partitions reach significance? -------
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    t, p = stats.mannwhitneyu(het.fl_gain_auprc, iid.fl_gain_auprc,
                              alternative="two-sided")
    print(f"  FL gain, heterogeneous vs IID: "
          f"{het.fl_gain_auprc.mean():+.4f} vs {iid.fl_gain_auprc.mean():+.4f}, "
          f"Mann-Whitney p={p:.4f}")

    weakened = abs(rg_i) < abs(rg_h) * 0.6 or pg_i > 0.05
    if weakened:
        print("\n  -> The capacity relationship WEAKENS when heterogeneity is")
        print("     removed while client sizes are held fixed. This supports the")
        print("     mechanism: capacity governs federation's value BECAUSE")
        print("     clients differ, not because of local sample size alone.")
        print("     Report as a designed control.")
    else:
        print("\n  -> The relationship PERSISTS under IID clients. Heterogeneity")
        print("     is then not the mechanism; the effect is driven by local")
        print("     sample size relative to capacity. That is still a finding,")
        print("     but the claim must be reworded — do not attribute it to")
        print("     heterogeneity.")
    print("\n  Note: with 25 runs per arm these tests have modest power. A")
    print("  non-significant difference means 'not demonstrated', not 'absent'.")


if __name__ == "__main__":
    main()