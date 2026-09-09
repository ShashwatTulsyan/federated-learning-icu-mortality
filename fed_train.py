"""
Step 3: Federated training + evaluation.

Implements FedAvg / FedProx / FedAdam directly rather than going through Flower's
Ray simulation. On a single GPU the clients run sequentially anyway, so the Ray
layer buys nothing and costs you VRAM fragmentation and much harder debugging.
The aggregation maths is identical.

Also runs the two baselines every reviewer will ask for:
  - centralized (all data pooled)  -> upper bound
  - local-only (each client alone) -> lower bound

Run:  python fed_train.py
"""
import copy
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss
from torch.utils.data import DataLoader, TensorDataset

import config as C
import export as EX
import metrics as MET
from models import build_model

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEV.type == "cuda":
    if getattr(C, "TF32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if getattr(C, "CUDNN_BENCHMARK", True):
        torch.backends.cudnn.benchmark = True


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
# Data
# ---------------------------------------------------------------------------
def free_gpu():
    """Release cached GPU blocks between runs.

    Each training run allocates fresh GPU-resident tensors. Across the hundreds
    of runs in a search or experiment sweep the allocator can fragment badly
    even though peak usage stays low, and a fragmented pool eventually fails to
    serve a contiguous request. Costs a few milliseconds.
    """
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_delta(mask, cap=None):
    """Hours since each feature was last actually measured (GRU-D style).

    Measurement FREQUENCY is itself prognostic in the ICU -- an unstable patient
    gets hourly gases, a stable one gets none. The mask says "missing"; the delta
    says "missing for nine hours", which carries far more information.
    """
    # cap at the observation window length, not a hardcoded 24 -- otherwise a
    # 48h run silently saturates every gap longer than 24h
    cap = float(cap if cap is not None else getattr(C, "OBS_WINDOW_H", 24))
    N, T, F = mask.shape
    delta = np.zeros((N, T, F), dtype=np.float32)
    for t in range(1, T):
        delta[:, t, :] = np.where(mask[:, t - 1, :] > 0, 1.0,
                                  delta[:, t - 1, :] + 1.0)
    return np.minimum(delta, cap) / cap


def load_data():
    if C.MODEL in ("gru", "hybrid"):
        d = np.load(C.TS_NPZ, allow_pickle=True)
        X = torch.from_numpy(d["X"]).float()
        mask = torch.from_numpy(d["mask"]).float()
        static = torch.from_numpy(d["static"]).float()
        y = torch.from_numpy(d["y"]).float()
        n_features, n_static = X.shape[2], static.shape[1]
        stay_ids = d["stay_id"]

        if getattr(C, "USE_DELTA", False):
            # Carry delta inside the mask tensor: X stays (N,T,F) so the
            # per-feature normaliser is unaffected, while the model receives
            # cat([X, mask]) = F + 2F = 3F channels.
            delta = torch.from_numpy(compute_delta(mask.numpy())).float()
            mask = torch.cat([mask, delta], dim=-1)          # (N, T, 2F)

        if C.MODEL == "hybrid":
            # replace the 3 static columns with the full aggregate feature table
            agg = pd.read_parquet(C.AGG_PQ)
            agg = agg.set_index("stay_id").reindex(stay_ids)
            drop = [c for c in ("label", "subject_id") if c in agg.columns]

            # LEAKAGE GUARD. With COMORBIDITY_SOURCE="both", the feature table
            # contains BOTH prior-admission (valid) and current-admission
            # (discharge-assigned, LEAKY) comorbidity columns. Loading every
            # column silently trains on the leaky ones and inflates every metric.
            # The "both" setting exists so run_experiments.py can COMPARE them;
            # ordinary training must never see the current-admission variant.
            leaky = [c for c in agg.columns
                     if c.startswith(("cci_curr_", "charlson_curr",
                                      "n_diagnoses_curr"))]
            if leaky and not getattr(C, "ALLOW_LEAKY_ICD", False):
                print(f"  [leakage guard] excluding {len(leaky)} "
                      f"current-admission ICD columns (assigned at discharge)")
                drop += leaky
            agg = agg.drop(columns=drop)
            tab = agg.to_numpy(dtype=np.float32)
            tab = np.nan_to_num(tab, nan=np.nan)   # per-client imputation later
            static = torch.from_numpy(tab)
            n_static = static.shape[1]
            print(f"  hybrid: {n_features} time-series features + "
                  f"{n_static} aggregate features")
    else:
        df = pd.read_parquet(C.AGG_PQ)
        feat_cols = [c for c in df.columns if c not in ("stay_id", "label")]
        X = torch.from_numpy(df[feat_cols].to_numpy(dtype=np.float32))
        mask = torch.zeros(len(df), 1)
        static = torch.zeros(len(df), 1)
        y = torch.from_numpy(df["label"].to_numpy(dtype=np.float32))
        n_features, n_static = X.shape[1], 1
        stay_ids = df["stay_id"].to_numpy()

    part = load_partition(C.OUT_DIR / f"partition_{C.PARTITION_SCHEME}.json",
                          stay_ids)
    return X, mask, static, y, part, n_features, n_static, stay_ids


def fit_normalizer(X, static, train_idx):
    """Compute imputation medians + z-score stats from ONE client's TRAIN split.

    This is the federated-correct way to normalise: a global mean/std would
    require pooling every client's raw data, which defeats the whole point. It
    also prevents test-set statistics leaking into training. Each client
    normalising with its own local statistics is exactly what a real hospital
    deployment would do.
    """
    tr = np.asarray(train_idx)
    if X.dim() == 3:                       # (N, T, F) time series
        flat = X[tr].reshape(-1, X.shape[2]).cpu().numpy()
    else:                                  # (N, F) tabular
        flat = X[tr].cpu().numpy()
    med = np.nanmedian(flat, axis=0)
    med = np.nan_to_num(med, nan=0.0)
    filled = np.where(np.isnan(flat), med[None, :], flat)
    mu = filled.mean(axis=0)
    sd = filled.std(axis=0) + 1e-6

    s_tr = static[tr].cpu().numpy()
    s_mu = np.nan_to_num(s_tr.mean(axis=0))
    s_sd = s_tr.std(axis=0) + 1e-6
    return {"med": med, "mu": mu, "sd": sd, "s_mu": s_mu, "s_sd": s_sd}


def apply_normalizer(X, static, norm):
    """Impute leading NaNs with the client's train median, then z-score."""
    Xn = X.clone()
    if Xn.dim() == 3:
        med = torch.tensor(norm["med"], dtype=torch.float32).view(1, 1, -1)
        mu = torch.tensor(norm["mu"], dtype=torch.float32).view(1, 1, -1)
        sd = torch.tensor(norm["sd"], dtype=torch.float32).view(1, 1, -1)
    else:
        med = torch.tensor(norm["med"], dtype=torch.float32).view(1, -1)
        mu = torch.tensor(norm["mu"], dtype=torch.float32).view(1, -1)
        sd = torch.tensor(norm["sd"], dtype=torch.float32).view(1, -1)
    Xn = torch.where(torch.isnan(Xn), med.expand_as(Xn), Xn)
    Xn = (Xn - mu) / sd

    Sn = static.clone()
    s_mu = torch.tensor(norm["s_mu"], dtype=torch.float32).view(1, -1)
    s_sd = torch.tensor(norm["s_sd"], dtype=torch.float32).view(1, -1)
    Sn = torch.nan_to_num((Sn - s_mu) / s_sd)
    return Xn, Sn


def evaluate_pooled(global_state, head_states, loaders, names, split,
                    n_features, n_static, calibrator=None, bn_states=None):
    """Evaluate across all clients and pool the predictions.

    Two reasons this replaces a single 'global model on a pooled loader':
      1. Each client normalises with its OWN local statistics, so there is no
         single normalised global tensor to evaluate on -- pooling predictions
         is the federated-correct equivalent.
      2. Under FedPer (PERSONALIZE=True) the global model's classifier head is
         NEVER trained -- only client heads are. Evaluating the global model
         would silently measure a randomly-initialised head.
    """
    zs, ys, sids, cls = [], [], [], []
    for n in names:
        m = build_model(C, n_features, n_static)
        m.load_state_dict(global_state)
        if bn_states is not None and n in bn_states:
            m.load_state_dict(bn_states[n], strict=False)
        if head_states is not None and head_states.get(n) is not None:
            m.load_state_dict(head_states[n], strict=False)
        z, yy = raw_logits(m, loaders[n][split])
        zs.append(z); ys.append(yy)
        sids.append(loaders[n][f"{split}_ids"])
        cls.append(np.full(len(z), n))
    z, y = np.concatenate(zs), np.concatenate(ys)
    evaluate_pooled.last_ids = np.concatenate(sids)
    evaluate_pooled.last_client = np.concatenate(cls)

    if calibrator is not None:
        p = calibrator.predict_proba(z.reshape(-1, 1))[:, 1]
    else:
        p = 1.0 / (1.0 + np.exp(-z))
    return _metrics(p, y), z, y


def fit_calibrator_from(z, y):
    if not C.CALIBRATE or len(np.unique(y)) < 2:
        return None
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(z.reshape(-1, 1), y)
    return lr


def _metrics(p, y):
    if len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "f1": float("nan"),
                "f1_at_0.5": float("nan"), "brier": float("nan"), "n": len(y)}
    ths = np.linspace(0.05, 0.95, 19)
    f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in ths]
    return {
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "f1": max(f1s),
        "f1_at_0.5": f1_score(y, (p >= 0.5).astype(int), zero_division=0),
        "brier": brier_score_loss(y, p),
        "n": len(y),
    }


MIN_VAL_EVENTS_FOR_THRESHOLD = 15


def pick_threshold(y_val, p_val, fallback=None):
    """Operating point from the validation split, with a small-sample guard.

    With only a few validation events the F1-maximising cut is essentially
    fitted to noise, and can land somewhere that classifies nothing as positive
    on test -- producing sensitivity = 0.000 and F1 = 0.000, which reads like a
    broken model rather than an unusable estimate. When events are too few we
    fall back to a supplied threshold (the pooled one) or, failing that, to the
    prevalence quantile, which is at least well defined.
    """
    y_val = np.asarray(y_val)
    n_pos = int(y_val.sum())
    if n_pos >= MIN_VAL_EVENTS_FOR_THRESHOLD:
        try:
            return MET.best_f1_threshold(y_val, p_val)[1]
        except Exception:
            pass
    if fallback is not None:
        return fallback
    prev = max(y_val.mean(), 1e-6)
    return float(np.quantile(np.asarray(p_val), 1 - prev))


def agg_weight_for(y, idx):
    """Aggregation weight for one client.

    FedAvg's n_k weighting implicitly assumes every sample carries equal
    information. For a rare binary outcome it does not: a client with many
    patients but few deaths contributes little about what death looks like.
    """
    idx = np.asarray(idx)
    n_pos = float((y[idx] == 1).sum())
    n_neg = float((y[idx] == 0).sum())
    mode = getattr(C, "AGG_WEIGHT", "samples")
    if mode == "events":
        return max(n_pos, 1.0)
    if mode == "effective":
        # harmonic mean of the two class counts = effective sample size for a
        # binary task; collapses toward 2*n_pos when positives are scarce
        return 2.0 * n_pos * n_neg / max(n_pos + n_neg, 1.0)
    return float(len(idx))


def local_keys(model):
    """Parameter names kept LOCAL (never aggregated), by branch prefix."""
    branches = list(getattr(C, "PERSONAL_BRANCHES", []))
    if not branches:
        return set()
    return {k for k in model.state_dict()
            if any(k.startswith(b + ".") or k == b for b in branches)}


def pos_weight_for(y, idx):
    """Positive-class weight for BCEWithLogitsLoss, or None if disabled."""
    if C.IMBALANCE != "pos_weight":
        return None
    n_pos = float((y[idx] == 1).sum())
    n_neg = float((y[idx] == 0).sum())
    return torch.tensor([n_neg / max(n_pos, 1.0)]).to(DEV)


class GPUBatcher:
    """Keeps one client's data resident in VRAM and yields batches by indexing.

    For models this small the bottleneck is host->device copies and DataLoader
    worker overhead, not compute. Holding the tensors on the GPU and slicing them
    removes both. ~250 MB per full dataset copy, so this is comfortable on 24 GB.
    """

    def __init__(self, X, mask, static, y, batch_size, shuffle,
                 weighted=False, device=None):
        dev = device or DEV
        self.X = X.to(dev, non_blocking=True)
        self.mask = mask.to(dev, non_blocking=True)
        self.static = static.to(dev, non_blocking=True)
        self.y = y.to(dev, non_blocking=True)
        self.bs = batch_size
        self.shuffle = shuffle
        self.n = len(self.y)
        self.drop_last = shuffle

        self.sample_w = None
        if weighted:
            yl = self.y
            n_pos = yl.sum().clamp(min=1)
            n_neg = (1 - yl).sum().clamp(min=1)
            self.sample_w = torch.where(yl == 1, 1.0 / n_pos, 1.0 / n_neg).double()

    def __len__(self):
        return max(1, self.n // self.bs if self.drop_last
                   else (self.n + self.bs - 1) // self.bs)

    def __iter__(self):
        if self.sample_w is not None:
            order = torch.multinomial(self.sample_w, self.n, replacement=True)
        elif self.shuffle:
            order = torch.randperm(self.n, device=self.X.device)
        else:
            order = torch.arange(self.n, device=self.X.device)

        end = (self.n // self.bs) * self.bs if self.drop_last else self.n
        for i in range(0, max(end, 0), self.bs):
            sel = order[i:i + self.bs]
            if len(sel) == 0:
                continue
            yield self.X[sel], self.mask[sel], self.static[sel], self.y[sel]


def build_loader(X, mask, static, y, idx, norm, batch_size, shuffle, weighted=False):
    """Slice out one client's rows, normalise with THAT client's stats, and
    (optionally) park the result in VRAM.

    Note this normalises only the client's own rows -- the earlier version
    normalised the full N x T x F tensor once per client, which wasted both time
    and memory."""
    weighted = weighted and C.IMBALANCE == "sampler"
    i = torch.as_tensor(np.asarray(idx), dtype=torch.long)
    Xs, Ms, Ss, ys = X[i], mask[i], static[i], y[i]
    Xs, Ss = apply_normalizer(Xs, Ss, norm)

    if C.GPU_RESIDENT and DEV.type == "cuda":
        return GPUBatcher(Xs, Ms, Ss, ys, batch_size, shuffle, weighted, DEV)

    ds = TensorDataset(Xs, Ms, Ss, ys)
    if weighted and shuffle:
        yl = ys.cpu().numpy()
        w = np.where(yl == 1, 1.0 / max(yl.sum(), 1), 1.0 / max((1 - yl).sum(), 1))
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.from_numpy(w).double(), num_samples=len(ys), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


def make_loader(X, mask, static, y, idx, batch_size, shuffle, weighted=False):
    weighted = weighted and C.IMBALANCE == "sampler"
    idx = torch.as_tensor(idx, dtype=torch.long)
    ds = TensorDataset(X[idx], mask[idx], static[idx], y[idx])
    if weighted and shuffle:
        # Class-balanced sampling: the federated equivalent of resampling for
        # imbalance, applied strictly locally so no cross-client info leaks.
        yl = y[idx].cpu().numpy()
        w = np.where(yl == 1, 1.0 / max(yl.sum(), 1), 1.0 / max((1 - yl).sum(), 1))
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.from_numpy(w).double(), num_samples=len(idx), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


# ---------------------------------------------------------------------------
# Local training / evaluation
# ---------------------------------------------------------------------------
def local_train(model, loader, epochs, lr, global_state=None, mu=0.0, pos_weight=None):
    model.to(DEV).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # NOTE: match by NAME, not by position. state_dict() contains buffers
    # (BatchNorm running_mean/var, num_batches_tracked) that parameters() does
    # not, so zipping the two silently misaligns and crashes on any model with
    # BatchNorm.
    prox_ref = None
    if mu > 0 and global_state is not None:
        prox_ref = {k: v.detach().clone().to(DEV).float()
                    for k, v in global_state.items()}

    for _ in range(epochs):
        for xb, mb, sb, yb in loader:
            xb, mb, sb, yb = xb.to(DEV), mb.to(DEV), sb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            loss = crit(model(xb, mb, sb), yb)

            if prox_ref is not None:
                # FedProx: mu/2 * ||w - w_global||^2 -- penalises drift away
                # from the global model, which is what stabilises training when
                # clients are strongly non-IID.
                prox = sum(((p - prox_ref[n]) ** 2).sum()
                           for n, p in model.named_parameters()
                           if n in prox_ref)
                loss = loss + (mu / 2.0) * prox

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return model


@torch.no_grad()
def raw_logits(model, loader):
    model.to(DEV).eval()
    out, ys = [], []
    for xb, mb, sb, yb in loader:
        out.append(model(xb.to(DEV), mb.to(DEV), sb.to(DEV)).cpu().numpy())
        ys.append(yb.cpu().numpy())
    return np.concatenate(out), np.concatenate(ys)


def fit_calibrator(model, val_loader):
    """Platt scaling: fit sigmoid(a*logit + b) on the validation split.
    Rank-preserving, so AUROC/AUPRC are untouched; only Brier improves."""
    if not C.CALIBRATE:
        return None
    z, y = raw_logits(model, val_loader)
    if len(np.unique(y)) < 2:
        return None
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(z.reshape(-1, 1), y)
    return lr


def evaluate(model, loader, calibrator=None):
    z, y = raw_logits(model, loader)
    if calibrator is not None:
        p = calibrator.predict_proba(z.reshape(-1, 1))[:, 1]
    else:
        p = 1.0 / (1.0 + np.exp(-z))
    return _metrics(p, y)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def fedavg(states, weights):
    total = float(sum(weights))
    out = copy.deepcopy(states[0])
    for k in out:
        if out[k].dtype.is_floating_point:
            out[k] = sum(s[k].float() * (w / total) for s, w in zip(states, weights))
        else:
            out[k] = states[0][k]
    return out


class ServerAdam:
    """FedAdam (Reddi et al.) -- server-side adaptive optimisation."""
    def __init__(self, state, lr, b1=0.9, b2=0.99, eps=1e-3):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: torch.zeros_like(v.float()) for k, v in state.items()}
        self.v = {k: torch.zeros_like(v.float()) for k, v in state.items()}

    def step(self, global_state, agg_state):
        new = copy.deepcopy(global_state)
        for k in global_state:
            if not global_state[k].dtype.is_floating_point:
                continue
            delta = agg_state[k].float() - global_state[k].float()
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * delta
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * delta ** 2
            new[k] = global_state[k].float() + self.lr * self.m[k] / (self.v[k].sqrt() + self.eps)
        return new


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def run_centralized(X, mask, static, y, part, n_features, n_static, stay_ids):
    print("\n=== Baseline: CENTRALIZED (pooled, not privacy-preserving) ===")
    tr = np.concatenate([v["train"] for v in part.values()])
    va = np.concatenate([v["val"] for v in part.values()])
    te = np.concatenate([v["test"] for v in part.values()])

    model = build_model(C, n_features, n_static)
    pw = pos_weight_for(y, tr)
    norm = fit_normalizer(X, static, tr)          # pooled train split
    tl = build_loader(X, mask, static, y, tr, norm, C.BATCH_SIZE, True, weighted=True)
    vl = build_loader(X, mask, static, y, va, norm, 1024, False)
    tel = build_loader(X, mask, static, y, te, norm, 1024, False)

    best, best_state, patience = -1, None, 0
    for ep in range(C.NUM_ROUNDS):
        local_train(model, tl, 1, C.LR, pos_weight=pw)
        m = evaluate(model, vl)
        if m["auprc"] > best:
            best, best_state, patience = m["auprc"], copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= C.EARLY_STOP_PATIENCE:
                break
    model.load_state_dict(best_state)
    cal = fit_calibrator(model, vl)
    _prob = lambda z: (cal.predict_proba(z.reshape(-1, 1))[:, 1] if cal is not None
                       else 1/(1+np.exp(-z)))
    zv, yv = raw_logits(model, vl)
    th_c = pick_threshold(yv, _prob(zv))
    zc, yc = raw_logits(model, tel)
    pc = _prob(zc)
    res = MET.full_report(yc, pc, n_boot=C.N_BOOTSTRAP, seed=C.SEED,
                          n_jobs=getattr(C,'BOOTSTRAP_JOBS',-1), threshold=th_c)
    run_centralized.test_preds = (yc, pc, stay_ids[te])
    print(f"  test: AUROC {res['auroc']:.4f} | AUPRC {res['auprc']:.4f} | "
          f"F1 {res['f1']:.4f} | Brier {res['brier']:.4f}")
    return res


def run_local_only(X, mask, static, y, part, n_features, n_static, stay_ids):
    print("\n=== Baseline: LOCAL-ONLY (each client trains alone) ===")
    out = {}
    for name, sp in part.items():
        model = build_model(C, n_features, n_static)
        tr = np.array(sp["train"])
        pw = pos_weight_for(y, tr)
        norm = fit_normalizer(X, static, tr)
        tl = build_loader(X, mask, static, y, tr, norm, C.BATCH_SIZE, True, weighted=True)
        vl = build_loader(X, mask, static, y, np.array(sp["val"]), norm, 1024, False)
        tel = build_loader(X, mask, static, y, np.array(sp["test"]), norm, 1024, False)
        local_train(model, tl, C.NUM_ROUNDS // 2, C.LR, pos_weight=pw)
        cal_l = fit_calibrator(model, vl)
        _prob = lambda z: (cal_l.predict_proba(z.reshape(-1, 1))[:, 1]
                           if cal_l is not None else 1/(1+np.exp(-z)))
        zv, yv = raw_logits(model, vl)
        th_l = pick_threshold(yv, _prob(zv))
        zc, yc = raw_logits(model, tel)
        out[name] = MET.full_report(yc, _prob(zc), n_boot=C.N_BOOTSTRAP,
                                    seed=C.SEED,
                                    n_jobs=getattr(C,'BOOTSTRAP_JOBS',-1),
                                    threshold=th_l)
        print(f"  {name:<12} AUROC {out[name]['auroc']:.4f} | AUPRC {out[name]['auprc']:.4f}")
    return out


# ---------------------------------------------------------------------------
# Federated training
# ---------------------------------------------------------------------------
def drift_diagnostics(global_state, client_states, weights, names):
    """Measure client drift -- why federation is losing to centralized.

    Two quantities, and they point at different fixes:

    * update NORM: how far each client moves from the global model in a round.
      Large norms mean local training is running away between syncs -- the fix
      is fewer local epochs / more rounds, or a stronger proximal term.

    * pairwise COSINE similarity between client updates: are clients pulling in
      the SAME direction or fighting each other? Near 1.0 means they broadly
      agree and averaging is nearly free. Near 0 or negative means the optimal
      model genuinely differs per client, averaging destroys signal, and no
      amount of extra synchronisation will fix it -- that calls for
      personalisation instead.
    """
    keys = [k for k in global_state
            if global_state[k].dtype.is_floating_point and "num_batches" not in k]
    deltas = []
    for sd in client_states:
        v = torch.cat([(sd[k].float() - global_state[k].float().cpu()).flatten()
                       for k in keys if k in sd])
        deltas.append(v)

    norms = [float(v.norm()) for v in deltas]
    cos = []
    for i in range(len(deltas)):
        for j in range(i + 1, len(deltas)):
            a, b = deltas[i], deltas[j]
            denom = (a.norm() * b.norm()).clamp(min=1e-12)
            cos.append(float((a @ b) / denom))

    return {
        "mean_update_norm": float(np.mean(norms)),
        "max_update_norm": float(np.max(norms)),
        "norm_ratio_max_min": float(np.max(norms) / max(np.min(norms), 1e-12)),
        "mean_pairwise_cosine": float(np.mean(cos)) if cos else float("nan"),
        "min_pairwise_cosine": float(np.min(cos)) if cos else float("nan"),
        "frac_negative_cosine": float(np.mean([c < 0 for c in cos])) if cos else 0.0,
    }


def interpret_drift(history):
    """Turn the drift numbers into a recommendation."""
    if not history:
        return
    late = history[len(history) // 2:]          # ignore the noisy warm-up
    cos = np.mean([h["mean_pairwise_cosine"] for h in late])
    ratio = np.mean([h["norm_ratio_max_min"] for h in late])
    neg = np.mean([h["frac_negative_cosine"] for h in late])

    print("\n  --- Client drift diagnosis (second half of training) ---")
    print(f"  mean pairwise cosine between client updates : {cos:+.3f}")
    print(f"  fraction of client pairs pulling oppositely : {neg:.1%}")
    print(f"  largest/smallest update norm ratio          : {ratio:.1f}x")

    if cos > 0.5:
        print("\n  -> Clients broadly AGREE on the update direction. Averaging is")
        print("     cheap, so the federated-centralized gap is mostly an")
        print("     optimisation issue: try MORE ROUNDS with FEWER LOCAL EPOCHS")
        print("     (NUM_ROUNDS=80, LOCAL_EPOCHS=1) before anything else.")
    elif cos > 0.15:
        print("\n  -> Partial agreement. Both fixes are worth trying; run the")
        print("     rounds/epochs change first since it is cheaper to justify,")
        print("     then personalisation.")
    else:
        print("\n  -> Clients DISAGREE on the update direction. The optimal model")
        print("     genuinely differs per care unit, so averaging is destroying")
        print("     signal and extra synchronisation will NOT recover it.")
        print("     PERSONALIZE=True (or a clustered/personalised FL variant) is")
        print("     the right fix; more rounds will mostly waste compute.")

    if ratio > 5:
        print(f"\n  Note: update norms differ {ratio:.0f}x across clients, so a few")
        print("  clients dominate each average. Consider weighting by event count")
        print("  rather than sample count, since positives carry the signal.")


def run_federated(X, mask, static, y, part, n_features, n_static, stay_ids):
    print(f"\n=== FEDERATED: {C.FL_ALGO.upper()} "
          f"(model={C.MODEL}, personalize={C.PERSONALIZE}) ===")
    names = list(part.keys())
    global_model = build_model(C, n_features, n_static)
    global_state = copy.deepcopy(global_model.state_dict())
    server_opt = ServerAdam(global_state, C.SERVER_LR) if C.FL_ALGO == "fedadam" else None

    # per-client loaders + class-imbalance weights, built once
    loaders = {}
    norms = {}
    for n in names:
        sp = part[n]
        tr = np.array(sp["train"])
        # normalisation fitted LOCALLY on this client's training data only
        norms[n] = fit_normalizer(X, static, tr)
        loaders[n] = {
            "train": build_loader(X, mask, static, y, tr, norms[n],
                                  C.BATCH_SIZE, True, weighted=True),
            "val": build_loader(X, mask, static, y, np.array(sp["val"]), norms[n],
                                1024, False),
            "test": build_loader(X, mask, static, y, np.array(sp["test"]), norms[n],
                                 1024, False),
            "n": agg_weight_for(y, tr),
            "n_samples": len(tr),
            "n_pos": int((y[np.asarray(tr)] == 1).sum()),
            "train_ids": stay_ids[tr],
            "val_ids": stay_ids[np.array(sp["val"])],
            "test_ids": stay_ids[np.array(sp["test"])],
            "pos_weight": pos_weight_for(y, tr),
        }

    # FedPer: each client keeps its own classifier head
    _probe = build_model(C, n_features, n_static)
    LOCAL_KEYS = local_keys(_probe)
    use_personal = C.PERSONALIZE or bool(LOCAL_KEYS)
    if LOCAL_KEYS:
        print(f"  branch-wise federation: keeping {len(LOCAL_KEYS)} params local "
              f"(branches={C.PERSONAL_BRANCHES})")
    if getattr(C, "AGG_WEIGHT", "samples") != "samples":
        print(f"  aggregation weighting: {C.AGG_WEIGHT}")
    head_states = {n: None for n in names} if use_personal else None
    bn_states = {}          # FedBN: per-client BatchNorm buffers

    best, best_state, patience, history = -1, None, 0, []
    drift_history = []
    for rnd in range(1, C.NUM_ROUNDS + 1):
        t0 = time.time()
        states, weights = [], []

        for n in names:
            local = build_model(C, n_features, n_static)
            local.load_state_dict(global_state)
            if C.FEDBN and n in bn_states:
                local.load_state_dict(bn_states[n], strict=False)
            if head_states is not None and head_states.get(n) is not None:
                local.load_state_dict(head_states[n], strict=False)

            local_train(
                local, loaders[n]["train"], C.LOCAL_EPOCHS, C.LR,
                global_state=global_state if C.FL_ALGO == "fedprox" else None,
                mu=C.PROX_MU if C.FL_ALGO == "fedprox" else 0.0,
                pos_weight=loaders[n]["pos_weight"],
            )
            sd = {k: v.detach().cpu() for k, v in local.state_dict().items()}
            if C.FEDBN:
                # FedBN: BatchNorm running stats stay on the client
                bn_keys = [k for k in sd if "running_mean" in k
                           or "running_var" in k or "num_batches_tracked" in k]
                for k in bn_keys:
                    bn_states.setdefault(n, {})[k] = sd[k]
                sd = {k: v for k, v in sd.items() if k not in bn_keys}
            keep_local = set(local.head_keys()) if C.PERSONALIZE else set()
            keep_local |= LOCAL_KEYS
            if keep_local:
                head_states[n] = {k: sd[k] for k in keep_local if k in sd}
                sd = {k: v for k, v in sd.items() if k not in keep_local}
            states.append(sd)
            weights.append(loaders[n]["n"])

        drift = drift_diagnostics(global_state, states, weights, names)
        drift_history.append({"round": rnd, **drift})

        agg = fedavg(states, weights)
        # restore any keys deliberately excluded from aggregation
        # (personalized heads and/or local BatchNorm buffers)
        merged = copy.deepcopy(global_state)
        merged.update(agg)
        agg = merged

        global_state = server_opt.step(global_state, agg) if server_opt else agg

        m, _, _ = evaluate_pooled(global_state, head_states, loaders, names,
                                  "val", n_features, n_static, bn_states=bn_states)
        history.append({"round": rnd, **m})
        print(f"  round {rnd:>3}/{C.NUM_ROUNDS} | val AUROC {m['auroc']:.4f} "
              f"AUPRC {m['auprc']:.4f} | cos {drift['mean_pairwise_cosine']:+.2f} "
              f"| {time.time()-t0:.1f}s")

        if m["auprc"] > best:
            best, best_state, patience = m["auprc"], copy.deepcopy(global_state), 0
        else:
            patience += 1
            if patience >= C.EARLY_STOP_PATIENCE:
                print(f"  early stop at round {rnd}")
                break

    global_model.load_state_dict(best_state)

    # Pooled evaluation: predictions gathered from every client (each using its
    # own local normaliser, and its own head under FedPer), then concatenated.
    _, z_val, y_val = evaluate_pooled(best_state, head_states, loaders, names,
                                      "val", n_features, n_static, bn_states=bn_states)
    global_cal = fit_calibrator_from(z_val, y_val)
    p_val = (global_cal.predict_proba(z_val.reshape(-1, 1))[:, 1]
             if global_cal is not None else 1/(1+np.exp(-z_val)))
    _, val_threshold = MET.best_f1_threshold(y_val, p_val)
    print(f"  operating threshold chosen on VALIDATION: {val_threshold:.4f}")
    global_res, z_test, y_test = evaluate_pooled(
        best_state, head_states, loaders, names, "test", n_features, n_static,
        global_cal, bn_states=bn_states)
    p_test = (global_cal.predict_proba(z_test.reshape(-1, 1))[:, 1]
              if global_cal is not None else 1/(1+np.exp(-z_test)))
    test_ids = evaluate_pooled.last_ids
    test_client = evaluate_pooled.last_client

    # publication-grade metrics with 95% bootstrap CIs
    global_res = MET.full_report(y_test, p_test, n_boot=C.N_BOOTSTRAP, seed=C.SEED,
                                 n_jobs=getattr(C,'BOOTSTRAP_JOBS',-1),
                                 threshold=val_threshold)

    d = EX.run_dir()
    EX.save_predictions(d, test_ids, y_test, p_test, test_client, "test")
    EX.save_curves(d, y_test, p_test)
    run_federated.test_preds = (y_test, p_test)
    print(f"\n  GLOBAL test: AUROC {global_res['auroc']:.4f} | "
          f"AUPRC {global_res['auprc']:.4f} | F1 {global_res['f1']:.4f} | "
          f"Brier {global_res['brier']:.4f}")

    per_client = {}
    for n in names:
        m = build_model(C, n_features, n_static)
        m.load_state_dict(best_state)
        if C.FEDBN and n in bn_states:
            m.load_state_dict(bn_states[n], strict=False)
        if head_states is not None and head_states.get(n) is not None:
            m.load_state_dict(head_states[n], strict=False)
        zv, yv = raw_logits(m, loaders[n]["val"])
        cal_c = fit_calibrator(m, loaders[n]["val"])
        _prob = lambda z: (cal_c.predict_proba(z.reshape(-1, 1))[:, 1]
                           if cal_c is not None else 1/(1+np.exp(-z)))
        th_c = pick_threshold(yv, _prob(zv), fallback=val_threshold)
        if int(np.asarray(yv).sum()) < MIN_VAL_EVENTS_FOR_THRESHOLD:
            print(f"      [note] {n}: only {int(np.asarray(yv).sum())} validation "
                  f"events -- using the pooled threshold instead of a local one")
        zc, yc = raw_logits(m, loaders[n]["test"])
        per_client[n] = MET.full_report(yc, _prob(zc), n_boot=C.N_BOOTSTRAP,
                                        seed=C.SEED,
                                        n_jobs=getattr(C,'BOOTSTRAP_JOBS',-1),
                                        threshold=th_c)
        print(f"    {n:<12} AUROC {per_client[n]['auroc']:.4f} | "
              f"AUPRC {per_client[n]['auprc']:.4f}")

    pd.DataFrame(history).to_csv(d / "convergence.csv", index=False)
    pd.DataFrame(drift_history).to_csv(d / "client_drift.csv", index=False)
    interpret_drift(drift_history)
    torch.save({"global_state": best_state,
                "head_states": head_states,
                "bn_states": bn_states,
                "config": {k: v for k, v in vars(C).items() if k.isupper()},
                "n_features": n_features, "n_static": n_static},
               d / "model.pt")
    return global_res, per_client


# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    if getattr(C, "OBS_WINDOW_H", 24) != 24:
        print(f"[NOTE] OBS_WINDOW_H={C.OBS_WINDOW_H}. Hyperparameters in config.py")
        print("       were selected at a 24h window; consider re-running tune.py.")
    print(f"Device: {DEV}")
    if DEV.type == "cuda":
        print(f"  TF32={getattr(C,'TF32',True)} "
              f"cudnn.benchmark={getattr(C,'CUDNN_BENCHMARK',True)} "
              f"compile={getattr(C,'COMPILE',False)}")
    if DEV.type == "cuda":
        cap = torch.cuda.get_device_capability(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"(sm_{cap[0]}{cap[1]}, {vram:.0f} GB) | "
              f"GPU_RESIDENT={C.GPU_RESIDENT}")

    X, mask, static, y, part, n_features, n_static, stay_ids = load_data()
    print(f"Loaded {len(y):,} stays | mortality {y.mean()*100:.2f}% | "
          f"{len(part)} clients | features={n_features}")

    d = EX.run_dir()
    EX.save_config(d)
    env = EX.save_environment(d)
    print(f"  artifacts -> {d}")

    fed_global, fed_client = run_federated(X, mask, static, y, part,
                                           n_features, n_static, stay_ids)
    central = run_centralized(X, mask, static, y, part, n_features, n_static, stay_ids)
    local = run_local_only(X, mask, static, y, part, n_features, n_static, stay_ids)

    rows = [{"method": "centralized (upper bound)", "scope": "global", **central},
            {"method": f"federated {C.FL_ALGO}", "scope": "global", **fed_global}]
    for n in fed_client:
        rows.append({"method": f"federated {C.FL_ALGO}", "scope": n, **fed_client[n]})
        rows.append({"method": "local-only (lower bound)", "scope": n, **local[n]})

    df = EX.save_metrics(d, rows)

    # Table 1 + cohort flow (from the saved cohort)
    try:
        cohort = pd.read_parquet(C.COHORT_PQ)
        EX.save_table1(d, cohort)
        flow_src = C.OUT_DIR / "cohort_flow.csv"
        if flow_src.exists():
            pd.read_csv(flow_src).to_csv(d / "cohort_flow.csv", index=False)
        het = C.OUT_DIR / f"heterogeneity_{C.PARTITION_SCHEME}.csv"
        if het.exists():
            pd.read_csv(het).to_csv(d / "heterogeneity.csv", index=False)
    except Exception as e:
        print(f"  [warn] could not build table1/flow: {e}")

    # statistical comparison: federated vs centralized on a common test set
    try:
        y_f, p_f = run_federated.test_preds
        y_c, p_c, ids_c = run_centralized.test_preds
        fed_df = pd.DataFrame({"id": evaluate_pooled.last_ids, "y": y_f, "p": p_f})
        cen_df = pd.DataFrame({"id": ids_c, "y": y_c, "p": p_c})
        mrg = fed_df.merge(cen_df, on="id", suffixes=("_fed", "_cen"))
        if len(mrg) > 50:
            EX.save_statistical_tests(
                d, mrg.y_fed.to_numpy(),
                {f"federated_{C.FL_ALGO}": mrg.p_fed.to_numpy(),
                 "centralized": mrg.p_cen.to_numpy()},
                reference="centralized")
            print(f"  statistical tests on {len(mrg):,} shared test stays")
    except Exception as e:
        print(f"  [warn] statistical tests skipped: {e}")

    EX.finalize(d)
    print("\nHeadline results:")
    print(df[df.scope == "global"][
        ["method", "n", "n_pos", "auroc", "auprc", "f1", "brier"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()