"""
Live demonstration of federated learning.

  python demo.py predict      RECOMMENDED. Load the trained model and run it
                              on held-out patients. Seconds, cannot fail.
  python demo.py train        watch federated training happen, round by round
  python demo.py compare      federated vs local-only vs centralized, live
  python demo.py app          browser app: interactive risk calculator

DATA GOVERNANCE
---------------
The MIMIC-IV data use agreement prohibits sharing patient-level data with people
who are not PhysioNet-credentialed. This demo therefore shows only AGGREGATE
quantities (client sizes, mortality rates, metrics) and SYNTHETIC patients in the
interactive calculator. No real patient record is ever displayed.

Runs entirely offline. No internet required.
"""
import copy
import json
import sys
import time

import numpy as np
import torch

import config as C
import fed_train as FT
import metrics as MET
from models import build_model

# ---------------------------------------------------------------------------
# Console presentation
# ---------------------------------------------------------------------------
W = 78
BAR = "█"
DIM = "░"


def rule(ch="="):
    print(ch * W)


def title(t, sub=""):
    print()
    rule()
    print(f"  {t}")
    if sub:
        print(f"  {sub}")
    rule()


def bar(frac, width=30, lo=0.0, hi=1.0):
    frac = 0.0 if not np.isfinite(frac) else (frac - lo) / max(hi - lo, 1e-9)
    frac = min(max(frac, 0.0), 1.0)
    n = int(round(frac * width))
    return BAR * n + DIM * (width - n)


def pause(s=0.35):
    time.sleep(s)


# ---------------------------------------------------------------------------
def load_everything():
    X, mask, static, y, part, nf, ns, stay_ids = FT.load_data()
    return X, mask, static, y, part, nf, ns


def client_table(part, y):
    print(f"\n  {'Client (ICU care unit)':<34}{'Patients':>10}{'Deaths':>9}"
          f"{'Rate':>8}")
    print("  " + "-" * (W - 4))
    rows = []
    for name, sp in part.items():
        tr = np.array(sp["train"])
        n, ev = len(tr), int(y[tr].sum())
        rows.append((name, n, ev, ev / n))
    for name, n, ev, r in sorted(rows, key=lambda x: -x[1]):
        short = name.replace(" Intensive Care Unit", "").replace(
            "Medical/Surgical", "Med/Surg")[:32]
        print(f"  {short:<34}{n:>10,}{ev:>9}{r*100:>7.1f}%")
    rates = [r for *_, r in rows]
    print("  " + "-" * (W - 4))
    print(f"  {'TOTAL':<34}{sum(r[1] for r in rows):>10,}"
          f"{sum(r[2] for r in rows):>9}"
          f"{sum(r[2] for r in rows)/sum(r[1] for r in rows)*100:>7.1f}%")
    print(f"\n  Mortality ranges {min(rates)*100:.1f}% to {max(rates)*100:.1f}% "
          f"across units — a {max(rates)/min(rates):.1f}-fold spread.")
    print("  This is what makes the problem non-IID: each 'hospital' sees a")
    print("  different patient population.")


# ---------------------------------------------------------------------------
def load_trained_model():
    """Load the model produced by fed_train.py, rebuilding its architecture
    from the config saved alongside the weights."""
    runs = sorted(C.OUT_DIR.glob("run_*"), key=lambda p: p.stat().st_mtime)
    for d in reversed(runs):
        f = d / "model.pt"
        if not f.exists():
            continue
        ck = torch.load(f, map_location="cpu", weights_only=False)
        saved = ck.get("config", {})
        for k in ("MODEL", "HIDDEN", "DROPOUT", "USE_DELTA", "BIDIRECTIONAL",
                  "PERSONAL_BRANCHES", "OBS_WINDOW_H"):
            if k in saved:
                setattr(C, k, saved[k])
        nf, ns = ck["n_features"], ck["n_static"]
        model = build_model(C, nf, ns)
        model.load_state_dict(ck["global_state"], strict=False)
        model.eval()
        return model, ck, d
    return None, None, None


def demo_predict():
    """Run the TRAINED model on held-out patients. No training, seconds to run."""
    title("TRAINED FEDERATED MODEL — PREDICTION",
          "Held-out patients the model has never seen")

    model, ck, run_dir = load_trained_model()
    if model is None:
        print("\n  No trained model found in artifacts/run_*/model.pt")
        print("  Run `python fed_train.py` first, or use `python demo.py train`.")
        sys.exit(1)

    algo = run_dir.name.replace("run_", "").split("_")[0]
    n_param = sum(p.numel() for p in model.parameters())
    print(f"\n  Loaded: {run_dir.name}")
    print(f"  Aggregation: {algo.upper()} | {n_param:,} parameters")
    print(f"  Trained across 7 ICU care units without centralising any record.")

    X, mask, static, y, part, nf, ns = load_everything()
    names = list(part)

    input("\n  [Enter] to run inference on the held-out test set ...")

    title("INFERENCE")
    all_p, all_y, per_unit = [], [], []
    for n in names:
        sp = part[n]
        tr, te = np.array(sp["train"]), np.array(sp["test"])
        norm = FT.fit_normalizer(X, static, tr)
        loader = FT.build_loader(X, mask, static, y, te, norm, 1024, False)
        t0 = time.time()
        z, yy = FT.raw_logits(model, loader)
        p = 1 / (1 + np.exp(-z))
        all_p.append(p); all_y.append(yy)
        short = n.replace(" Intensive Care Unit", "")[:26]
        auprc = (MET.average_precision_score(yy, p)
                 if len(np.unique(yy)) > 1 else np.nan)
        per_unit.append((short, len(yy), int(yy.sum()), auprc))
        print(f"  {short:<28} {len(yy):>5} patients scored in "
              f"{(time.time()-t0)*1000:>5.0f} ms")
        pause(0.15)

    p = np.concatenate(all_p); yv = np.concatenate(all_y)
    prev = yv.mean()

    title("HOW WELL DID IT DO?")
    m = MET.point_metrics(yv, p)
    print(f"\n  Patients scored          {len(yv):>8,}")
    print(f"  Deaths in this set       {int(yv.sum()):>8,}  ({prev*100:.1f}%)")
    print()
    print(f"  AUROC   {m['auroc']:.3f}   {bar(m['auroc'], 30, 0.5, 1.0)}")
    print(f"  AUPRC   {m['auprc']:.3f}   {bar(m['auprc'], 30, 0.0, 0.7)}")
    print(f"          {'':5}   random guessing would score {prev:.3f}")
    print(f"          {'':5}   -> {m['auprc']/prev:.1f}x better than chance")
    print()
    print(f"  Sensitivity {m['sensitivity']:.3f}   of patients who died, this "
          f"fraction was flagged")
    print(f"  Specificity {m['specificity']:.3f}   of patients who survived, "
          f"this fraction was not")
    print(f"  Brier score {m['brier']:.3f}   lower is better; measures whether "
          f"the")
    print(f"  {'':22} probabilities are honest, not just ranked")

    # risk-decile table: aggregate only, no patient rows
    title("DOES A HIGHER SCORE MEAN A HIGHER RISK?")
    print("\n  Patients grouped into ten bands by predicted risk.")
    print("  If the model works, observed mortality should rise down the table.\n")
    print(f"  {'Predicted risk band':<24}{'Patients':>10}{'Died':>8}"
          f"{'Observed':>10}")
    print("  " + "-" * (W - 4))
    q = np.quantile(p, np.linspace(0, 1, 11))
    for i in range(10):
        sel = (p >= q[i]) & (p <= q[i+1] if i == 9 else p < q[i+1])
        if sel.sum() == 0:
            continue
        obs = yv[sel].mean()
        print(f"  {q[i]*100:>5.1f}% - {q[i+1]*100:>5.1f}%      "
              f"{int(sel.sum()):>10,}{int(yv[sel].sum()):>8}"
              f"{obs*100:>9.1f}%  {bar(obs, 18, 0, 0.6)}")
    lo, hi = yv[p <= q[1]].mean(), yv[p >= q[9]].mean()
    print("  " + "-" * (W - 4))
    print(f"\n  Lowest-risk tenth:  {lo*100:.1f}% died")
    print(f"  Highest-risk tenth: {hi*100:.1f}% died")
    if lo > 0:
        print(f"  The model separates them by a factor of {hi/lo:.0f}.")

    title("PER CARE UNIT")
    print(f"\n  {'Unit':<28}{'Patients':>10}{'Deaths':>9}{'AUPRC':>9}")
    print("  " + "-" * (W - 4))
    for short, n_, ev, ap in per_unit:
        note = "   (too few deaths to judge)" if ev < 15 else ""
        print(f"  {short:<28}{n_:>10,}{ev:>9}"
              f"{ap:>9.3f}{note}")
    print("\n  Units with very few deaths give unreliable estimates — reported")
    print("  for completeness, not interpretation.")

    print()
    rule()
    print("  No patient record was displayed. Only aggregate results, which is")
    print("  what the data use agreement permits.")
    rule()


def demo_train(rounds=15, quick=True):
    title("LIVE FEDERATED LEARNING",
          "Seven ICU care units train a shared model without sharing patients")

    print("\n  Loading data ...")
    X, mask, static, y, part, nf, ns = load_everything()
    names = list(part)
    print(f"  {len(y):,} ICU stays | {len(names)} clients | "
          f"{nf} time-series features")

    client_table(part, y)

    input("\n  [Enter] to begin federated training ...")

    if quick:
        C.HIDDEN, C.LOCAL_EPOCHS, C.BATCH_SIZE = 96, 1, 256

    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    gm = build_model(C, nf, ns)
    gstate = copy.deepcopy(gm.state_dict())
    n_param = sum(p.numel() for p in gm.parameters())

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
            "n": len(tr),
            "pos_weight": FT.pos_weight_for(y, tr),
        }

    title("TRAINING", f"{n_param:,} parameters | {rounds} communication rounds")
    print("\n  Each round: every unit trains privately on its own patients,")
    print("  sends only model WEIGHTS to the server, and receives the average.")
    print("  No patient data ever leaves a unit.\n")

    history = []
    for rnd in range(1, rounds + 1):
        t0 = time.time()
        print(f"  Round {rnd:>2}/{rounds}")
        states, ws = [], []
        for n in names:
            m = build_model(C, nf, ns)
            m.load_state_dict(gstate)
            FT.local_train(m, loaders[n]["train"], C.LOCAL_EPOCHS, C.LR,
                           pos_weight=loaders[n]["pos_weight"])
            sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
            states.append(sd)
            ws.append(loaders[n]["n"])
            short = n.replace(" Intensive Care Unit", "")[:26]
            print(f"      {short:<28} trained locally on "
                  f"{loaders[n]['n']:>5,} patients   [weights sent]")
            pause(0.05)

        gstate = FT.fedavg(states, ws)
        print(f"      {'SERVER':<28} averaged {len(states)} models "
              f"-> new global model")

        gm.load_state_dict(gstate)
        zs, ys = [], []
        for n in names:
            z, yy = FT.raw_logits(gm, loaders[n]["val"])
            zs.append(z); ys.append(yy)
        yv = np.concatenate(ys)
        pv = 1 / (1 + np.exp(-np.concatenate(zs)))
        auroc = MET.roc_auc_score(yv, pv)
        auprc = MET.average_precision_score(yv, pv)
        history.append((rnd, auroc, auprc))
        print(f"      {'':28} AUROC {auroc:.3f}  {bar(auroc, 24, 0.5, 0.95)}")
        print(f"      {'':28} AUPRC {auprc:.3f}  {bar(auprc, 24, 0.0, 0.6)}"
              f"   ({time.time()-t0:.1f}s)\n")

    title("RESULT")
    print(f"\n  Round   AUROC   AUPRC")
    for rnd, a, p in history:
        mark = "  <- best" if p == max(h[2] for h in history) else ""
        print(f"  {rnd:>5}   {a:.3f}   {p:.3f}{mark}")
    best = max(history, key=lambda h: h[2])
    prev = float(yv.mean())
    print(f"\n  Best: AUROC {best[1]:.3f}, AUPRC {best[2]:.3f}")
    print(f"  Mortality rate is {prev*100:.1f}%, so random guessing scores "
          f"{prev:.3f}.")
    print(f"  The model is {best[2]/prev:.1f}x better than chance.")
    print("\n  Crucially: the server never saw a single patient record.")

    torch.save({"state": gstate, "nf": nf, "ns": ns,
                "hidden": C.HIDDEN, "model": C.MODEL},
               C.OUT_DIR / "demo_model.pt")
    print(f"\n  Model saved for the interactive app "
          f"({C.OUT_DIR / 'demo_model.pt'})")


# ---------------------------------------------------------------------------
def demo_compare(rounds=12):
    title("WHY FEDERATE?",
          "Three ways to train, compared live")

    X, mask, static, y, part, nf, ns = load_everything()
    names = list(part)
    C.HIDDEN, C.LOCAL_EPOCHS, C.BATCH_SIZE = 96, 1, 256

    print("\n  1. LOCAL-ONLY    each unit trains alone, no sharing")
    print("  2. FEDERATED     units share model weights only")
    print("  3. CENTRALIZED   all patient data pooled (privacy violated)")
    input("\n  [Enter] to run all three ...")

    def evaluate_on_val(model, loaders):
        zs, ys = [], []
        for n in names:
            z, yy = FT.raw_logits(model, loaders[n]["val"])
            zs.append(z); ys.append(yy)
        yv = np.concatenate(ys)
        pv = 1 / (1 + np.exp(-np.concatenate(zs)))
        return MET.average_precision_score(yv, pv)

    loaders, norms = {}, {}
    for n in names:
        sp = part[n]
        tr = np.array(sp["train"])
        norms[n] = FT.fit_normalizer(X, static, tr)
        loaders[n] = {
            "train": FT.build_loader(X, mask, static, y, tr, norms[n],
                                     C.BATCH_SIZE, True, weighted=True),
            "val": FT.build_loader(X, mask, static, y, np.array(sp["val"]),
                                   norms[n], 1024, False),
            "n": len(tr), "pos_weight": FT.pos_weight_for(y, tr),
        }

    # ---- 1. local-only -------------------------------------------------
    print("\n  [1/3] LOCAL-ONLY — each unit alone")
    local_scores = []
    for n in names:
        torch.manual_seed(C.SEED)
        m = build_model(C, nf, ns)
        for _ in range(rounds):
            FT.local_train(m, loaders[n]["train"], 1, C.LR,
                           pos_weight=loaders[n]["pos_weight"])
        z, yy = FT.raw_logits(m, loaders[n]["val"])
        s = (MET.average_precision_score(yy, 1/(1+np.exp(-z)))
             if len(np.unique(yy)) > 1 else np.nan)
        local_scores.append(s)
        short = n.replace(" Intensive Care Unit", "")[:26]
        print(f"      {short:<28} AUPRC {s:.3f}  {bar(s, 20, 0, .6)}")
    local_mean = float(np.nanmean(local_scores))
    print(f"      {'MEAN':<28} AUPRC {local_mean:.3f}")

    # ---- 2. federated --------------------------------------------------
    print("\n  [2/3] FEDERATED — weights shared, data not")
    torch.manual_seed(C.SEED)
    gm = build_model(C, nf, ns)
    gstate = copy.deepcopy(gm.state_dict())
    for rnd in range(1, rounds + 1):
        states, ws = [], []
        for n in names:
            m = build_model(C, nf, ns); m.load_state_dict(gstate)
            FT.local_train(m, loaders[n]["train"], 1, C.LR,
                           pos_weight=loaders[n]["pos_weight"])
            states.append({k: v.detach().cpu() for k, v in m.state_dict().items()})
            ws.append(loaders[n]["n"])
        gstate = FT.fedavg(states, ws)
        gm.load_state_dict(gstate)
        s = evaluate_on_val(gm, loaders)
        print(f"      round {rnd:>2}  AUPRC {s:.3f}  {bar(s, 20, 0, .6)}")
    fed = s

    # ---- 3. centralized ------------------------------------------------
    print("\n  [3/3] CENTRALIZED — all data pooled (not privacy-preserving)")
    tr_all = np.concatenate([part[n]["train"] for n in names])
    norm = FT.fit_normalizer(X, static, tr_all)
    tl = FT.build_loader(X, mask, static, y, tr_all, norm,
                         C.BATCH_SIZE, True, weighted=True)
    torch.manual_seed(C.SEED)
    cm = build_model(C, nf, ns)
    pw = FT.pos_weight_for(y, tr_all)
    for rnd in range(1, rounds + 1):
        FT.local_train(cm, tl, 1, C.LR, pos_weight=pw)
        s = evaluate_on_val(cm, loaders)
        print(f"      epoch {rnd:>2}  AUPRC {s:.3f}  {bar(s, 20, 0, .6)}")
    cent = s

    title("COMPARISON")
    print()
    for label, v, note in (
            ("Local-only (no sharing)", local_mean, "each unit alone"),
            ("Federated (weights only)", fed, "privacy preserved"),
            ("Centralized (data pooled)", cent, "privacy violated")):
        print(f"  {label:<30} AUPRC {v:.3f}  {bar(v, 26, 0, .6)}  {note}")
    print()
    print(f"  Federation gains {fed-local_mean:+.3f} over training alone,")
    print(f"  and gives up {cent-fed:+.3f} versus pooling all the data.")
    print("\n  That trade — most of the benefit, none of the data sharing —")
    print("  is the reason federated learning exists.")


# ---------------------------------------------------------------------------
def demo_app(port=8000):
    """Browser app: interactive risk calculator on SYNTHETIC patients."""
    import http.server
    import socketserver
    import urllib.parse

    # prefer the real trained model; fall back to one made by `demo.py train`
    model, ck, run_dir = load_trained_model()
    if model is not None:
        nf, ns = ck["n_features"], ck["n_static"]
        source = run_dir.name
    else:
        path = C.OUT_DIR / "demo_model.pt"
        if not path.exists():
            print("No model found. Run `python fed_train.py` "
                  "or `python demo.py train` first.")
            sys.exit(1)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        C.HIDDEN, C.MODEL = ck["hidden"], ck["model"]
        model = build_model(C, ck["nf"], ck["ns"])
        model.load_state_dict(ck["state"])
        model.eval()
        nf, ns = ck["nf"], ck["ns"]
        source = "demo_model.pt"

    FEATS = [("Heart rate", "bpm", 60, 140, 85),
             ("Systolic BP", "mmHg", 70, 180, 120),
             ("Respiratory rate", "/min", 8, 40, 18),
             ("SpO2", "%", 80, 100, 97),
             ("Temperature", "°C", 34, 41, 37.0),
             ("Lactate", "mmol/L", 0.5, 15, 1.5),
             ("Creatinine", "mg/dL", 0.3, 10, 1.0),
             ("White cell count", "K/uL", 1, 40, 9)]

    def predict(vals):
        """Map slider values to a model input. SYNTHETIC patient only."""
        rng = np.random.default_rng(0)
        x = torch.zeros(1, C.OBS_WINDOW_H, nf)
        # z-score each entered value against its plausible range, then write it
        # into the corresponding channel across the window
        for i, ((_, _, lo, hi, _), v) in enumerate(zip(FEATS, vals)):
            if i < nf:
                z = (v - (lo + hi) / 2) / ((hi - lo) / 4)
                x[0, :, i] = float(z)
        mk = torch.ones(1, C.OBS_WINDOW_H, nf)
        st = torch.zeros(1, ns)
        with torch.no_grad():
            return float(torch.sigmoid(model(x, mk, st)).item())

    HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>ICU Mortality Risk — Federated Model</title><style>
*{box-sizing:border-box} body{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
margin:0;background:#0f1720;color:#e6edf3}
.wrap{max-width:940px;margin:0 auto;padding:28px}
h1{font-size:23px;margin:0 0 4px} .sub{color:#8b98a5;margin-bottom:22px;font-size:14px}
.grid{display:grid;grid-template-columns:1fr 320px;gap:26px}
@media(max-width:820px){.grid{grid-template-columns:1fr}}
.card{background:#161f2b;border:1px solid #263243;border-radius:10px;padding:18px}
.row{margin-bottom:15px}
.lab{display:flex;justify-content:space-between;font-size:13px;margin-bottom:5px}
.val{color:#58a6ff;font-variant-numeric:tabular-nums}
input[type=range]{width:100%;accent-color:#58a6ff}
.risk{font-size:52px;font-weight:600;text-align:center;margin:6px 0;
font-variant-numeric:tabular-nums}
.meter{height:12px;background:#263243;border-radius:6px;overflow:hidden;margin:14px 0}
.fill{height:100%;transition:width .25s,background .25s}
.tag{text-align:center;font-size:14px;letter-spacing:.4px;margin-bottom:14px}
.note{font-size:12px;color:#8b98a5;border-top:1px solid #263243;padding-top:12px;
margin-top:14px}
button{background:#1f6feb;color:#fff;border:0;padding:9px 14px;border-radius:7px;
cursor:pointer;font-size:13px;margin:3px 3px 0 0}
button.alt{background:#30363d}
.warn{background:#2d1a10;border:1px solid #5c3a1e;color:#e8b88a;padding:11px;
border-radius:8px;font-size:12.5px;margin-bottom:18px}
</style></head><body><div class="wrap">
<h1>ICU Mortality Risk Prediction</h1>
<div class="sub">Model trained by federated learning across 7 ICU care units &mdash;
no patient data was centralised</div>
<div class="warn"><b>Demonstration only.</b> These are synthetic values you set
yourself, not real patient data. This tool is not for clinical use.</div>
<div class="grid"><div class="card">__SLIDERS__
<div style="margin-top:16px">
<button onclick="preset('stable')">Stable patient</button>
<button onclick="preset('deteriorating')" class="alt">Deteriorating</button>
<button onclick="preset('critical')" class="alt">Critical</button>
</div></div>
<div class="card">
<div style="text-align:center;font-size:13px;color:#8b98a5">
Predicted in-hospital mortality risk</div>
<div class="risk" id="risk">&mdash;</div>
<div class="meter"><div class="fill" id="fill" style="width:0%"></div></div>
<div class="tag" id="tag">&nbsp;</div>
<div class="note">
<b>How to read this.</b> The ICU baseline mortality rate is about 10%. A score
well above that marks a patient the model considers high risk.<br><br>
<b>How it was trained.</b> Each care unit trained on its own patients and shared
only model weights. The server averaged those weights. No record ever left a unit.
</div></div></div></div>
<script>
const F=__FEATS__;
function preset(k){const p={stable:[75,125,16,98,36.8,1.0,0.9,7],
deteriorating:[110,95,26,92,38.5,3.5,1.8,16],
critical:[132,78,32,86,39.4,7.0,3.4,24]}[k];
F.forEach((f,i)=>{document.getElementById('s'+i).value=p[i];});upd();}
function upd(){const v=F.map((f,i)=>{const x=+document.getElementById('s'+i).value;
document.getElementById('v'+i).textContent=x+' '+f[1];return x;});
fetch('/predict?v='+v.join(',')).then(r=>r.json()).then(d=>{
const pct=(d.risk*100);document.getElementById('risk').textContent=pct.toFixed(1)+'%';
const f=document.getElementById('fill');f.style.width=Math.min(pct*3,100)+'%';
let c,t;if(pct<8){c='#3fb950';t='LOWER THAN AVERAGE';}
else if(pct<20){c='#d29922';t='ELEVATED';}
else if(pct<40){c='#db6d28';t='HIGH';}else{c='#f85149';t='VERY HIGH';}
f.style.background=c;document.getElementById('tag').textContent=t;
document.getElementById('tag').style.color=c;});}
F.forEach((f,i)=>document.getElementById('s'+i).addEventListener('input',upd));
upd();
</script></body></html>"""

    sliders = ""
    for i, (nm, unit, lo, hi, dflt) in enumerate(FEATS):
        step = "0.1" if isinstance(dflt, float) else "1"
        sliders += (f'<div class="row"><div class="lab"><span>{nm}</span>'
                    f'<span class="val" id="v{i}">{dflt} {unit}</span></div>'
                    f'<input type="range" id="s{i}" min="{lo}" max="{hi}" '
                    f'step="{step}" value="{dflt}"></div>')
    page = HTML.replace("__SLIDERS__", sliders).replace(
        "__FEATS__", json.dumps([[f[0], f[1]] for f in FEATS]))

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            if u.path == "/predict":
                q = urllib.parse.parse_qs(u.query)
                vals = [float(x) for x in q.get("v", ["0"])[0].split(",")]
                r = predict(vals)
                body = json.dumps({"risk": r}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            else:
                body = page.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    title("INTERACTIVE DEMONSTRATION")
    print(f"\n  Model: {source}")
    print(f"\n  Open this in a browser:   http://localhost:{port}")
    print("\n  Move the sliders to build a synthetic patient and watch the")
    print("  risk score respond. Preset buttons show a stable, a deteriorating")
    print("  and a critical patient.")
    print("\n  No real patient data is used or displayed.")
    print("\n  Ctrl+C to stop.\n")
    with socketserver.TCPServer(("", port), H) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  stopped.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "predict"
    if cmd == "predict":
        demo_predict()
    elif cmd == "train":
        demo_train()
    elif cmd == "compare":
        demo_compare()
    elif cmd == "app":
        p = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        demo_app(p)
    else:
        print(__doc__)
