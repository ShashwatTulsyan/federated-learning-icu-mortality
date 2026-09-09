"""Verify every module is the current version and they agree with each other.

Run this after copying files over. It catches the most common failure mode:
updating some modules but not others, which surfaces later as a confusing
AttributeError deep inside a training run.
"""
import importlib
import sys

REQUIRED = {
    "config":            ["MIMIC_ROOT", "TS_FEATURES", "VASOPRESSOR_ITEMIDS",
                          "VENT_ITEMIDS", "MODEL", "USE_DELTA", "BIDIRECTIONAL",
                          "GPU_RESIDENT", "BOOTSTRAP_JOBS", "N_BOOTSTRAP",
                          "COMORBIDITY_SOURCE", "ALLOW_LEAKY_ICD", "CHARLSON",
                          "PARTITION_SCHEME", "AGG_WEIGHT", "PERSONAL_BRANCHES",
                          "FEDBN", "IMBALANCE", "CALIBRATE"],
    "metrics":           ["best_f1_threshold", "_f1_sweep", "point_metrics",
                          "bootstrap_ci", "full_report", "delong_test"],
    "models":            ["MortalityHybrid", "MortalityGRU", "MortalityMLP",
                          "build_model"],
    "fed_train":         ["compute_delta", "GPUBatcher", "build_loader",
                          "fit_normalizer", "evaluate_pooled", "run_federated",
                          "free_gpu", "agg_weight_for", "local_keys",
                          "pick_threshold", "fit_calibrator_from", "load_data",
                          "ServerAdam", "fedavg", "local_train", "raw_logits"],
    "tune":              ["SPACE", "fit_and_score", "apply", "run_jobs"],
    "run_experiments":   ["experiment_capacity", "experiment_algorithms",
                          "experiment_leakage", "experiment_ablation",
                          "load_partition", "run_jobs"],
    "check_partition":   [],
    "make_figures":      ["fig1_capacity", "set_source", "pick_runs"],
    "partition":         ["partition_careunit", "partition_dirichlet",
                          "partition_shuffled", "split_client",
                          "assert_no_patient_leakage", "cohort_fingerprint",
                          "check_clients"],
    "export":            ["run_dir", "save_metrics", "save_table1"],
}

def main():
    ok = True
    for mod, attrs in REQUIRED.items():
        try:
            m = importlib.import_module(mod)
        except Exception as e:
            print(f"  [FAIL] cannot import {mod}.py -> {e}")
            ok = False
            continue
        missing = [a for a in attrs if not hasattr(m, a)]
        if missing:
            print(f"  [FAIL] {mod}.py is OUT OF DATE -- missing: {missing}")
            ok = False
        else:
            print(f"  [ ok ] {mod}.py")

    # Cross-module symbol resolution. The failure this catches: one module
    # references a name that a STALE copy of another module does not define
    # (e.g. run_experiments doing `FT.free_gpu` against an old fed_train).
    # Import errors above would miss it, because the reference is at module
    # scope in the *other* file.
    print()
    try:
        import fed_train as _ft, run_experiments as _re, tune as _tn
        print("  [ ok ] cross-module references resolve "
              "(run_experiments + tune vs fed_train)")
    except AttributeError as e:
        print(f"  [FAIL] cross-module reference broken -> {e}")
        print("         One file is older than the others. Copy the whole set.")
        ok = False
    except Exception as e:
        print(f"  [warn] could not fully cross-check: {e}")

    # cross-module consistency
    try:
        import config as C
        import models, torch
        n_ts = len(C.TS_FEATURES)
        ch = 3 if C.USE_DELTA else 2
        mdl = models.build_model(C, n_ts, 50)
        x = torch.randn(2, C.OBS_WINDOW_H, n_ts)
        mk = torch.ones(2, C.OBS_WINDOW_H, n_ts * (2 if C.USE_DELTA else 1))
        st = torch.randn(2, 50)
        out = mdl(x, mk, st)
        print(f"  [ ok ] model builds and runs: MODEL={C.MODEL}, "
              f"{n_ts} features, {ch} channels -> {tuple(out.shape)}")
    except Exception as e:
        print(f"  [FAIL] model/config mismatch -> {e}")
        ok = False

    print()
    if ok:
        print("All modules are current and consistent. Safe to run.")
        sys.exit(0)
    print("Copy the flagged file(s) again from the latest set, then re-run this.")
    sys.exit(1)


if __name__ == "__main__":
    main()