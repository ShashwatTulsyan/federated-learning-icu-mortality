"""
Saves every artifact a Q1 submission needs, into artifacts/run_<tag>/.

Written so that a reviewer asking "can you show X?" is always a file lookup,
never a re-run:

  predictions.parquet      per-stay y_true / y_pred (redraw any figure, run any
                           post-hoc test, without retraining)
  metrics_*.csv            point estimates + 95% bootstrap CIs, global & per-client
  roc_points.csv           ROC curve coordinates
  pr_points.csv            precision-recall curve coordinates
  calibration_curve.csv    reliability-diagram points
  decision_curve.csv       net benefit vs. treat-all / treat-none
  convergence.csv          per-round validation metrics
  statistical_tests.csv    DeLong + paired-bootstrap comparisons vs. baselines
  table1.csv               baseline characteristics (by client and by outcome)
  cohort_flow.csv          TRIPOD exclusion cascade
  heterogeneity.csv        per-client non-IID evidence
  config_snapshot.json     every hyperparameter used
  environment.json         library versions + GPU, for reproducibility
  model.pt                 final weights
"""
import json
import platform
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import torch

import config as C
import metrics as M


def run_dir(tag=None):
    tag = tag or f"{C.FL_ALGO}_{C.MODEL}_{C.PARTITION_SCHEME}" + \
                 ("_pers" if C.PERSONALIZE else "")
    d = C.OUT_DIR / f"run_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
def save_environment(d):
    def _v(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return "not installed"

    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5).stdout.strip() or None
    except Exception:
        git = None

    env = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _v("torch"),
        "numpy": _v("numpy"),
        "pandas": _v("pandas"),
        "sklearn": _v("sklearn"),
        "duckdb": _v("duckdb"),
        "scipy": _v("scipy"),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0))
                          if torch.cuda.is_available() else None,
        "mimic_version": "v3.1",
        "seed": C.SEED,
    }
    (d / "environment.json").write_text(json.dumps(env, indent=2))
    return env


def save_config(d):
    cfg = {k: v for k, v in vars(C).items()
           if k.isupper() and not k.startswith("_")}
    cfg = {k: (str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None)))
               else v) for k, v in cfg.items()}
    (d / "config_snapshot.json").write_text(json.dumps(cfg, indent=2, default=str))


# ---------------------------------------------------------------------------
def save_predictions(d, stay_ids, y, p, client, split):
    df = pd.DataFrame({"stay_id": stay_ids, "client": client, "split": split,
                       "y_true": y, "y_pred": p})
    path = d / "predictions.parquet"
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df.to_parquet(path, index=False)
    return path


def save_curves(d, y, p, prefix=""):
    pd.DataFrame(M.roc_points(y, p)).to_csv(d / f"{prefix}roc_points.csv", index=False)
    pd.DataFrame(M.pr_points(y, p)).to_csv(d / f"{prefix}pr_points.csv", index=False)
    pd.DataFrame(M.calibration_curve_points(y, p)).to_csv(
        d / f"{prefix}calibration_curve.csv", index=False)
    pd.DataFrame(M.decision_curve(y, p)).to_csv(
        d / f"{prefix}decision_curve.csv", index=False)


def save_metrics(d, rows, name="metrics.csv"):
    df = pd.DataFrame(rows)
    df.to_csv(d / name, index=False)

    # manuscript-ready formatted version
    fmt = []
    for r in rows:
        fmt.append({
            "method": r.get("method"), "scope": r.get("scope"),
            "N": r.get("n"), "events": r.get("n_pos"),
            "AUROC (95% CI)": M.fmt_ci(r, "auroc"),
            "AUPRC (95% CI)": M.fmt_ci(r, "auprc"),
            "Sensitivity (95% CI)": M.fmt_ci(r, "sensitivity"),
            "Specificity (95% CI)": M.fmt_ci(r, "specificity"),
            "F1 (95% CI)": M.fmt_ci(r, "f1"),
            "Brier (95% CI)": M.fmt_ci(r, "brier"),
            "Calib. slope": f"{r.get('calib_slope', float('nan')):.3f}",
            "ECE": f"{r.get('ece', float('nan')):.4f}",
        })
    pd.DataFrame(fmt).to_csv(d / name.replace(".csv", "_formatted.csv"), index=False)
    return df


def save_statistical_tests(d, y, preds_by_method, reference):
    """DeLong (AUROC) + paired bootstrap (AUPRC) for every method vs reference."""
    rows = []
    ref = preds_by_method[reference]
    for name, p in preds_by_method.items():
        if name == reference:
            continue
        a1, a2, z, pv = M.delong_test(y, p, ref)
        dprc, dlo, pv2 = M.bootstrap_auprc_test(y, p, ref, n_boot=500, seed=C.SEED)
        rows.append({
            "method": name, "reference": reference,
            "auroc_method": a1, "auroc_reference": a2, "auroc_diff": a1 - a2,
            "delong_z": z, "delong_p": pv,
            "auprc_diff": dprc, "auprc_boot_p": pv2,
        })
    df = pd.DataFrame(rows)
    df.to_csv(d / "statistical_tests.csv", index=False)
    return df


# ---------------------------------------------------------------------------
def save_table1(d, cohort, part=None):
    """Baseline characteristics: overall, by outcome, and by client."""
    def _summ(sub):
        return {
            "N": len(sub),
            "Age, median (IQR)": f"{sub.age.median():.0f} "
                                 f"({sub.age.quantile(.25):.0f}-{sub.age.quantile(.75):.0f})",
            "Male, n (%)": f"{(sub.gender=='M').sum()} ({(sub.gender=='M').mean()*100:.1f})",
            "ICU LOS h, median (IQR)": f"{sub.los_hours.median():.1f} "
                                       f"({sub.los_hours.quantile(.25):.1f}-"
                                       f"{sub.los_hours.quantile(.75):.1f})",
            "Emergency admission, n (%)":
                f"{sub.admission_type.fillna('').str.contains('EMER').sum()} "
                f"({sub.admission_type.fillna('').str.contains('EMER').mean()*100:.1f})",
            "In-hospital mortality, n (%)":
                f"{int(sub.label.sum())} ({sub.label.mean()*100:.1f})",
        }

    rows = {"Overall": _summ(cohort),
            "Survived": _summ(cohort[cohort.label == 0]),
            "Died": _summ(cohort[cohort.label == 1])}
    for unit in sorted(cohort.first_careunit.dropna().unique()):
        sub = cohort[cohort.first_careunit == unit]
        if len(sub) >= C.MIN_CLIENT_SIZE:
            rows[f"Client: {unit}"] = _summ(sub)

    t1 = pd.DataFrame(rows).T
    t1.index.name = "Group"
    t1.to_csv(d / "table1.csv")
    return t1


def save_cohort_flow(d, flow):
    """TRIPOD-style exclusion cascade. `flow` is a list of (step, n_remaining)."""
    df = pd.DataFrame(flow, columns=["step", "n_stays"])
    df["excluded"] = df.n_stays.shift(1) - df.n_stays
    df.to_csv(d / "cohort_flow.csv", index=False)
    return df


# ---------------------------------------------------------------------------
def finalize(d):
    print(f"\n{'='*62}\nArtifacts saved for submission -> {d}\n{'='*62}")
    for f in sorted(d.iterdir()):
        print(f"  {f.name:<32} {f.stat().st_size/1024:>8.1f} KB")
    print(f"{'='*62}")
