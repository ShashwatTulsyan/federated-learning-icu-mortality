"""
True external validation: federate across eICU hospitals ONLY, then evaluate
zero-shot on MIMIC-IV -- a health system the federation never saw during
training. This is the experiment that actually tests generalization, as
opposed to combine_datasets.py's setup where MIMIC-IV is just another client
inside the same federation.

Comparison this produces:
  in-distribution : federated-on-eICU, tested on eICU's own held-out stays
  external        : federated-on-eICU, tested on ALL of MIMIC-IV (unseen)
  reference        : federated-on-MIMIC (paper 1's model), tested on MIMIC

The gap between in-distribution and external performance is the
"generalization gap" -- the number a reviewer will actually want to see for a
paper claiming real multi-institution federation.

Requires combine_datasets.py's cohort.parquet (has a `dataset` column marking
mimic/eicu rows) and timeseries.npz already built.

Run:  python external_validation.py
"""
import copy
import json

import numpy as np
import pandas as pd
import torch

import warnings
# Both filtered below are confirmed harmless, not silent-error suppression:
#   1. "All-NaN slice" -- fires inside FT.fit_normalizer when a rare lab was
#      never recorded in one client's training split; the code already
#      nan_to_num's the result to 0.0 immediately afterward.
#   2. "not writable" -- fires because X[idx]/static[idx] are numpy VIEWS
#      into the loaded .npz; harmless since this script never writes into
#      those tensors in place.
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message=".*is not writable.*")

import config as C
import fed_train as FT
import metrics as MET
import partition as P
from models import build_model

MAX_ROUNDS = 60
PATIENCE = 10
SEEDS = [42, 101, 2026, 7, 1234]


def build_eicu_only_partition(cohort):
    """eICU hospitals as clients; ALL MIMIC-IV rows withheld from training
    entirely (not just held out per-client -- excluded from the training
    universe altogether).

    `cohort` must already carry a valid `row` column (added once, in main(),
    mapping stay_id -> tensor row index). Do NOT recreate it here: an earlier
    version called .reset_index().rename(columns={"index": "row"}), which
    produced a SECOND column also named "row" holding pandas' positional
    index instead -- a duplicate-name collision that pandas does not error
    on, and that silently corrupts every downstream group/count.
    """
    eicu = cohort[(cohort.dataset == "eicu") & cohort.row.notna()].copy()
    eicu["row"] = eicu["row"].astype(np.int64)
    groups = eicu.groupby("first_careunit")["row"].apply(list)
    sizes = eicu.first_careunit.value_counts()
    valid = sizes[sizes >= C.MIN_CLIENT_SIZE].index
    return {u: np.array(sorted(groups[u]), dtype=np.int64) for u in valid}


def fit_one(X, mask, static, y, part, nf, ns, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    names = list(part)
    gm = build_model(C, nf, ns)
    gstate = copy.deepcopy(gm.state_dict())

    loaders = {}
    for n in names:
        idx = part[n]
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(idx))
        n_te = int(len(idx) * 0.15); n_va = int(len(idx) * 0.15)
        te, va, tr = idx[perm[:n_te]], idx[perm[n_te:n_te+n_va]], idx[perm[n_te+n_va:]]
        norm = FT.fit_normalizer(X, static, tr)
        loaders[n] = {
            "train": FT.build_loader(X, mask, static, y, tr, norm, C.BATCH_SIZE,
                                     True, weighted=True),
            "val": FT.build_loader(X, mask, static, y, va, norm, 1024, False),
            "n": len(tr), "pos_weight": FT.pos_weight_for(y, tr),
            "norm": norm,
        }

    best, best_state, patience = -1.0, None, 0
    for rnd in range(MAX_ROUNDS):
        states, ws = [], []
        for n in names:
            m = build_model(C, nf, ns); m.load_state_dict(gstate)
            FT.local_train(m, loaders[n]["train"], C.LOCAL_EPOCHS, C.LR,
                           pos_weight=loaders[n]["pos_weight"])
            states.append({k: v.detach().cpu() for k, v in m.state_dict().items()})
            ws.append(loaders[n]["n"])
        gstate = FT.fedavg(states, ws)

        gm.load_state_dict(gstate)
        zs, ys = [], []
        for n in names:
            z, yy = FT.raw_logits(gm, loaders[n]["val"])
            zs.append(z); ys.append(yy)
        yv = np.concatenate(ys)
        auprc = (MET.average_precision_score(yv, 1/(1+np.exp(-np.concatenate(zs))))
                 if len(np.unique(yv)) > 1 else np.nan)
        if np.isfinite(auprc) and auprc > best:
            best, best_state, patience = auprc, copy.deepcopy(gstate), 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    FT.free_gpu()
    return gm, best_state, loaders


def evaluate_on(model, state, X, mask, static, y, idx, norm=None):
    """Score `idx` with `model`. Explicitly places everything on FT.DEV,
    matching the convention used by FT.raw_logits() elsewhere in this
    codebase -- required because FT.raw_logits() mutates its model argument
    onto the GPU in place (model.to(DEV)) during training's per-round
    validation checks, so by the time this function runs the model may
    already live on cuda:0 while these freshly-built input tensors do not.
    """
    model.load_state_dict(state)
    model.to(FT.DEV).eval()
    if norm is None:
        norm = FT.fit_normalizer(X, static, idx)   # fallback: self-normalised
    Xn, Sn = FT.apply_normalizer(X[idx], static[idx], norm)
    Xn, Sn = Xn.to(FT.DEV), Sn.to(FT.DEV)
    mk = mask[idx].to(FT.DEV)
    with torch.no_grad():
        z = model(Xn, mk, Sn).cpu().numpy()
    p = 1 / (1 + np.exp(-z))
    return MET.point_metrics(y[idx].numpy(), p)


def load_tensors():
    """Load the data tensors WITHOUT requiring partition_<scheme>.json to
    exist. FT.load_data() insists on a partition file even though the caller
    might not need one -- this script builds its own eICU-only partition
    directly from cohort.parquet, so that dependency would be a needless
    crash if partition.py has not been run on this dataset."""
    d = np.load(C.TS_NPZ, allow_pickle=True)
    X = torch.from_numpy(d["X"]).float()
    mask = torch.from_numpy(d["mask"]).float()
    static = torch.from_numpy(d["static"]).float()
    y = torch.from_numpy(d["y"]).float()
    n_features, n_static = X.shape[2], static.shape[1]
    stay_ids = d["stay_id"]

    if C.MODEL == "hybrid":
        agg = pd.read_parquet(C.AGG_PQ)
        agg = agg.set_index("stay_id").reindex(stay_ids)
        drop = [c for c in ("label", "subject_id") if c in agg.columns]
        leaky = [c for c in agg.columns
                if c.startswith(("cci_curr_", "charlson_curr", "n_diagnoses_curr"))]
        if leaky and not getattr(C, "ALLOW_LEAKY_ICD", False):
            print(f"  [leakage guard] excluding {len(leaky)} "
                 f"current-admission ICD columns (assigned at discharge)")
            drop += leaky
        agg = agg.drop(columns=drop)
        static = torch.from_numpy(
            np.nan_to_num(agg.to_numpy(dtype=np.float32), nan=np.nan))
        n_static = static.shape[1]
        print(f"  hybrid: {n_features} time-series features + "
             f"{n_static} aggregate features")
    return X, mask, static, y, n_features, n_static, stay_ids


def main():
    print("Loading combined dataset ...")
    X, mask, static, y, nf, ns, stay_ids = load_tensors()
    cohort = pd.read_parquet(C.OUT_DIR / "cohort.parquet")
    if "dataset" not in cohort.columns:
        print("cohort.parquet has no 'dataset' column -- did you run "
             "combine_datasets.py? This script needs the combined cohort.")
        return

    row_of_stay = {int(s): i for i, s in enumerate(stay_ids)}
    cohort = cohort[cohort.stay_id.isin(row_of_stay)].copy()
    cohort["row"] = cohort.stay_id.map(row_of_stay).astype(np.int64)

    eicu_part = build_eicu_only_partition(cohort)
    mimic_rows = cohort.loc[cohort.dataset == "mimic", "row"].to_numpy()
    print(f"  eICU clients: {len(eicu_part)} hospitals, "
         f"{sum(len(v) for v in eicu_part.values()):,} stays")
    print(f"  MIMIC-IV held out ENTIRELY from training: {len(mimic_rows):,} stays")

    rows = []
    for seed in SEEDS:
        print(f"\n--- seed {seed} ---")
        model, state, loaders = fit_one(X, mask, static, y, eicu_part, nf, ns, seed)

        # in-distribution: eICU's own held-out validation stays, pooled across
        # hospitals (validation, not a fresh test split, for consistency with
        # how the rest of the suite reports client performance)
        zs, ys = [], []
        for n, L in loaders.items():
            m2 = build_model(C, nf, ns); m2.load_state_dict(state)
            z, yy = FT.raw_logits(m2, L["val"])
            zs.append(z); ys.append(yy)
        p_in = 1/(1+np.exp(-np.concatenate(zs)))
        m_in = MET.point_metrics(np.concatenate(ys), p_in)

        # external: ALL of MIMIC-IV, evaluated TWO ways with the SAME
        # eICU-trained weights, to separate scale mismatch from genuine
        # relational domain shift:
        #   avg_norm  -- MIMIC values scaled using eICU's average statistics
        #                (what the model saw during training)
        #   self_norm -- MIMIC values scaled using MIMIC's OWN statistics
        #                (label-free, so not leakage -- this is standard
        #                target-domain test-time renormalisation)
        # If self_norm scores much better, the raw gap above was substantially
        # a SCALE artifact, not evidence the model failed to learn transferable
        # clinical patterns. If both score similarly poorly, the gap is real.
        avg_norm = {k: np.mean([L["norm"][k] for L in loaders.values()], axis=0)
                   for k in loaders[list(loaders)[0]]["norm"]}
        mimic_self_norm = FT.fit_normalizer(X, static, mimic_rows)

        m_ext = evaluate_on(model, state, X, mask, static, y, mimic_rows, avg_norm)
        m_ext_self = evaluate_on(model, state, X, mask, static, y, mimic_rows,
                                 mimic_self_norm)

        rows.append({"seed": seed, "scope": "in_distribution_eicu",
                    "auroc": m_in["auroc"], "auprc": m_in["auprc"]})
        rows.append({"seed": seed, "scope": "external_mimic_eicu_norm",
                    "auroc": m_ext["auroc"], "auprc": m_ext["auprc"]})
        rows.append({"seed": seed, "scope": "external_mimic_self_norm",
                    "auroc": m_ext_self["auroc"], "auprc": m_ext_self["auprc"]})
        print(f"  in-distribution (eICU): AUROC {m_in['auroc']:.3f} "
             f"AUPRC {m_in['auprc']:.3f}")
        print(f"  external, eICU-scale (MIMIC, unseen): AUROC {m_ext['auroc']:.3f} "
             f"AUPRC {m_ext['auprc']:.3f}")
        print(f"  external, MIMIC-scale (same weights): AUROC "
             f"{m_ext_self['auroc']:.3f} AUPRC {m_ext_self['auprc']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(C.OUT_DIR / "external_validation_raw.csv", index=False)
    summary = df.groupby("scope")[["auroc", "auprc"]].agg(["mean", "std"])
    summary.to_csv(C.OUT_DIR / "external_validation_summary.csv")
    print("\n" + "=" * 60)
    print(summary.to_string())

    m = lambda s, c: df[df.scope == s][c].mean()
    gap_raw = m("in_distribution_eicu", "auroc") - m("external_mimic_eicu_norm", "auroc")
    gap_selfnorm = m("in_distribution_eicu", "auroc") - m("external_mimic_self_norm", "auroc")
    recovered = gap_raw - gap_selfnorm

    print(f"\nGeneralization gap, eICU-scale normalisation : {gap_raw:+.4f} AUROC")
    print(f"Generalization gap, MIMIC-scale normalisation: {gap_selfnorm:+.4f} AUROC")
    print(f"Gap closed by target-domain renormalisation  : {recovered:+.4f} AUROC "
         f"({recovered/max(gap_raw,1e-9)*100:.0f}% of the raw gap)")
    print()
    if recovered > 0.5 * gap_raw:
        print("MAJORITY of the apparent generalization gap is a SCALE artifact,")
        print("not evidence the model failed to learn transferable clinical")
        print("patterns. Report the self-norm gap as the primary result, and")
        print("state plainly that naive cross-database deployment (using the")
        print("source domain's normalisation) inflates the apparent gap.")
    else:
        print("Renormalising to MIMIC's own scale does NOT substantially close")
        print("the gap. This supports a genuine relational domain shift -- the")
        print("model learned patterns from eICU that do not transfer to MIMIC,")
        print("not merely a scale mismatch. Report the raw gap as the headline")
        print("result.")
    print("\nSaved -> external_validation_raw.csv / _summary.csv")


if __name__ == "__main__":
    main()