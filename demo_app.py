"""
Case review — the trained federated model on real held-out patients.

Loads the model produced by fed_train.py and runs it on the held-out TEST split:
patients the model has never seen, using their actual first-24h physiology as
recorded in MIMIC-IV. For each case it shows the real trajectory, the model's
predicted risk, and the true outcome.

    python demo_app.py            # http://localhost:8000
    python demo_app.py 8080       # different port

AUDIENCE: this displays patient-level MIMIC-IV data and must only be shown to
PhysioNet-credentialed viewers. For a general audience use `demo.py predict`,
which reports aggregate results only.
"""
import hashlib
import json
import sys
import urllib.parse
import http.server
import socketserver

import numpy as np
import torch

import config as C
import fed_train as FT
import metrics as MET
from models import build_model

# Variables worth plotting, in clinical reading order
PLOT = [
    ("heart_rate", "Heart rate", "bpm", 40, 160),
    ("map", "Mean arterial pressure", "mmHg", 40, 130),
    ("sbp", "Systolic BP", "mmHg", 60, 190),
    ("resp_rate", "Respiratory rate", "/min", 5, 45),
    ("spo2", "SpO2", "%", 75, 100),
    ("temp_c", "Temperature", "°C", 33, 41),
    ("lactate", "Lactate", "mmol/L", 0, 15),
    ("creatinine", "Creatinine", "mg/dL", 0, 8),
    ("wbc", "White cell count", "K/uL", 0, 40),
    ("platelets", "Platelets", "K/uL", 0, 500),
    ("bun", "Urea nitrogen", "mg/dL", 0, 150),
    ("urine", "Urine output", "mL/h", 0, 400),
]
INTERVENTIONS = [("vasopressor", "Vasopressors"), ("ventilation", "Mechanical ventilation")]


class Store:
    """Loads everything once and precomputes predictions for the test split."""

    def __init__(self):
        runs = sorted(C.OUT_DIR.glob("run_*"), key=lambda p: p.stat().st_mtime)
        ck = None
        for d in reversed(runs):
            if (d / "model.pt").exists():
                ck = torch.load(d / "model.pt", map_location="cpu",
                                weights_only=False)
                self.run = d.name
                break
        if ck is None:
            raise SystemExit("No trained model. Run fed_train.py first.")

        for k in ("MODEL", "HIDDEN", "DROPOUT", "USE_DELTA", "BIDIRECTIONAL",
                  "PERSONAL_BRANCHES", "OBS_WINDOW_H"):
            if k in ck.get("config", {}):
                setattr(C, k, ck["config"][k])
        self.algo = self.run.replace("run_", "").split("_")[0]

        d = np.load(C.TS_NPZ, allow_pickle=True)
        self.Xraw = d["X"]                     # (N, T, F) raw clinical scale
        self.names = [str(x) for x in d["feature_names"]]
        self.stay_id = d["stay_id"]

        X, mask, static, y, part, nf, ns, _ = FT.load_data()
        self.X, self.mask, self.static, self.y = X, mask, static, y
        self.part = part
        self.model = build_model(C, nf, ns)
        self.model.load_state_dict(ck["global_state"], strict=False)
        self.model.eval()
        self.n_param = sum(p.numel() for p in self.model.parameters())

        # per-client normaliser, then predictions for that client's test split
        self.cases = []
        for cname, sp in part.items():
            tr, te = np.array(sp["train"]), np.array(sp["test"])
            norm = FT.fit_normalizer(X, static, tr)
            self.__dict__.setdefault("_norms", {})[cname] = norm
            Xn, Sn = FT.apply_normalizer(X[te], static[te], norm)
            with torch.no_grad():
                z = self.model(Xn, mask[te], Sn).numpy()
            p = 1 / (1 + np.exp(-z))
            for i, row in enumerate(te):
                self.cases.append({
                    "row": int(row), "client": cname,
                    "risk": float(p[i]), "died": int(y[row].item()),
                })
        self.cases.sort(key=lambda c: -c["risk"])
        self.by_row = {c["row"]: c for c in self.cases}
        self.by_stay = {int(self.stay_id[c["row"]]): c for c in self.cases}
        for rank, c in enumerate(self.cases):
            c["rank"] = rank

        yv = np.array([c["died"] for c in self.cases])
        pv = np.array([c["risk"] for c in self.cases])
        self.metrics = MET.point_metrics(yv, pv)
        self.prevalence = float(yv.mean())
        # operating threshold: the F1-optimal cut, as used in the paper
        self.threshold = float(self.metrics["threshold"])

    # ---- verification support -------------------------------------------
    def provenance(self):
        def h(path):
            try:
                d = hashlib.sha256()
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        d.update(chunk)
                return d.hexdigest()[:16]
            except Exception:
                return "n/a"

        import datetime as _dt
        mp = C.OUT_DIR / self.run / "model.pt"
        wsum = float(sum(float(v.abs().sum())
                         for v in self.model.state_dict().values()
                         if v.dtype.is_floating_point))
        return {
            "model_path": str(mp), "model_sha256": h(mp),
            "model_mtime": _dt.datetime.fromtimestamp(
                mp.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "data_path": str(C.TS_NPZ), "data_sha256": h(C.TS_NPZ),
            "data_mtime": _dt.datetime.fromtimestamp(
                C.TS_NPZ.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "n_param": self.n_param,
            "weight_checksum": round(wsum, 4),
        }

    def predict_row(self, row, edits=None):
        """Score one stay after editing its raw trajectory point by point.

        `edits` maps feature name -> a list of length OBS_WINDOW_H, one entry
        per hour, each either a new value (float) or None to leave that hour
        unchanged. Several features can be edited at once and are applied
        together in a single forward pass -- this is what lets the demo show
        the combined effect of several simultaneous changes, not just one at a
        time. The edit is applied to the RAW clinical values, then the
        client's own normaliser is applied, exactly as in training. This is a
        genuine forward pass, not a lookup or interpolation between cached
        answers.
        """
        c = self.by_row[row]
        norm = self._norms[c["client"]]
        x = self.X[row:row + 1].clone()
        T = x.shape[1]
        n_edited_points = 0
        if edits:
            for key, series in edits.items():
                if key not in self.names or not isinstance(series, (list, tuple)):
                    continue
                j = self.names.index(key)
                for i, v in enumerate(series):
                    if v is None or i >= T:
                        continue
                    x[0, i, j] = float(v)
                    n_edited_points += 1
        Xn, Sn = FT.apply_normalizer(x, self.static[row:row + 1], norm)
        with torch.no_grad():
            z = self.model(Xn, self.mask[row:row + 1], Sn).item()
        return float(1 / (1 + np.exp(-z))), n_edited_points

    def shuffled_scores(self, n=400, seed=0):
        """Re-score a sample with RANDOMISED weights.

        If the displayed numbers came from a pre-computed file, destroying the
        weights would change nothing. It collapses performance to chance, which
        demonstrates the predictions are produced by these weights.
        """
        import copy as _copy
        g = torch.Generator().manual_seed(seed)
        backup = _copy.deepcopy(self.model.state_dict())
        sd = self.model.state_dict()
        for k, v in sd.items():
            if v.dtype.is_floating_point:
                # scale to the trained weights' own magnitude; overly large
                # random weights saturate the GRU and produce NaN logits
                sc = (v.std().item() if v.numel() > 1 else abs(v.item())) + 1e-6
                sd[k] = torch.randn(v.shape, generator=g) * float(min(sc, 0.15))
        self.model.load_state_dict(sd)
        rng = np.random.default_rng(seed)
        sample = rng.choice(len(self.cases), min(n, len(self.cases)),
                            replace=False)
        ps, ys = [], []
        for i in sample:
            c = self.cases[int(i)]
            norm = self._norms[c["client"]]
            r = c["row"]
            Xn, Sn = FT.apply_normalizer(self.X[r:r+1], self.static[r:r+1], norm)
            with torch.no_grad():
                z = self.model(Xn, self.mask[r:r+1], Sn).item()
            ps.append(1 / (1 + np.exp(-z))); ys.append(c["died"])
        self.model.load_state_dict(backup)          # always restore

        ys, ps = np.array(ys), np.array(ps)
        trained = np.array([self.cases[int(i)]["risk"] for i in sample])
        # a destroyed network can emit non-finite logits; treat those as
        # uninformative rather than failing the demonstration
        ps = np.nan_to_num(ps, nan=0.5, posinf=1.0, neginf=0.0)
        try:
            a_sh = float(MET.roc_auc_score(ys, ps))
        except Exception:
            a_sh = 0.5
        return {
            "n": int(len(ys)),
            "auroc_trained": float(MET.roc_auc_score(ys, trained)),
            "auroc_shuffled": a_sh,
        }

    def series(self, row):
        """Raw clinical trajectory for one stay, NaNs preserved as gaps."""
        out = {}
        for key, label, unit, lo, hi in PLOT:
            if key not in self.names:
                continue
            j = self.names.index(key)
            v = self.Xraw[row, :, j]
            if np.all(np.isnan(v)):
                continue
            out[key] = {
                "label": label, "unit": unit, "lo": lo, "hi": hi,
                "values": [None if np.isnan(x) else round(float(x), 1) for x in v],
            }
        iv = {}
        for key, label in INTERVENTIONS:
            if key in self.names:
                j = self.names.index(key)
                v = self.Xraw[row, :, j]
                iv[key] = {"label": label,
                           "on": bool(np.nanmax(np.nan_to_num(v)) > 0)}
        return out, iv


HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Case review — federated ICU mortality model</title><style>
*{box-sizing:border-box;margin:0;padding:0}
body{font:14px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
background:#0d1117;color:#c9d1d9}
header{border-bottom:1px solid #21262d;padding:14px 24px;display:flex;
align-items:baseline;gap:18px;background:#010409}
header h1{font-size:16px;font-weight:600;color:#e6edf3}
header .meta{font-size:12px;color:#7d8590}
.dua{background:#1c1408;border-bottom:1px solid #3d2c0e;color:#d8a657;
padding:7px 24px;font-size:11.5px}
.layout{display:grid;grid-template-columns:280px 1fr;height:calc(100vh - 84px)}
.side{border-right:1px solid #21262d;overflow-y:auto;background:#0d1117}
.side h2{font-size:11px;text-transform:uppercase;letter-spacing:.7px;
color:#7d8590;padding:14px 16px 8px;font-weight:600}
.filters{padding:0 12px 12px;display:flex;flex-wrap:wrap;gap:5px}
.filters button{background:#161b22;border:1px solid #30363d;color:#c9d1d9;
padding:5px 9px;border-radius:5px;font-size:11.5px;cursor:pointer}
.filters button:hover{border-color:#58a6ff}
.case{padding:9px 16px;border-bottom:1px solid #161b22;cursor:pointer;
display:flex;justify-content:space-between;align-items:center;gap:8px}
.case:hover{background:#161b22}
.case.sel{background:#0d2d5e;border-left:3px solid #58a6ff;padding-left:13px}
.case .id{font-size:12px;color:#8b949e;font-variant-numeric:tabular-nums}
.case .unit{font-size:10.5px;color:#6e7681}
.case .r{font-size:13px;font-variant-numeric:tabular-nums;font-weight:600}
.main{overflow-y:auto;padding:20px 24px}
.hdr{display:flex;justify-content:space-between;align-items:flex-start;
margin-bottom:18px;gap:20px;flex-wrap:wrap}
.hdr h2{font-size:17px;color:#e6edf3;font-weight:600}
.hdr .sub{font-size:12px;color:#7d8590;margin-top:3px}
.verdict{display:flex;gap:12px;align-items:stretch}
.box{background:#161b22;border:1px solid #30363d;border-radius:8px;
padding:12px 16px;min-width:150px}
.box .k{font-size:10.5px;text-transform:uppercase;letter-spacing:.6px;
color:#7d8590;margin-bottom:5px}
.box .v{font-size:26px;font-weight:600;font-variant-numeric:tabular-nums}
.box .n{font-size:11px;color:#7d8590;margin-top:3px}
.charts{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));
gap:12px}
.chart{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:10px 12px}
.chart.edited{border-color:#7a4a1e;background:#1c150d}
.chart .t{font-size:11.5px;color:#8b949e;display:flex;justify-content:space-between;
margin-bottom:4px}
.chart .t b{color:#c9d1d9;font-weight:600}
svg{display:block;width:100%;height:74px}
.iv{display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap}
.pill{font-size:11.5px;padding:4px 10px;border-radius:20px;border:1px solid #30363d;
color:#7d8590}
.pill.on{background:#3d1d1d;border-color:#8b3232;color:#ff9d9d}
.empty{color:#6e7681;text-align:center;padding:70px 20px;font-size:13px}
.tally{font-size:11.5px;color:#7d8590;padding:10px 16px;border-top:1px solid #21262d}
.whatif{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 16px;
margin-bottom:14px}
.whatif .k{font-size:10.5px;text-transform:uppercase;letter-spacing:.6px;color:#7d8590;
margin-bottom:8px}
.whatif button{background:#21262d;border:1px solid #30363d;color:#c9d1d9;padding:5px 10px;
border-radius:5px;font-size:11.5px;cursor:pointer;margin:0 5px 5px 0}
.whatif button:hover{border-color:#58a6ff}
.wi{font-size:12.5px;margin-top:8px;font-variant-numeric:tabular-nums}
</style></head><body>
<header><h1>Case review</h1>
<span class="meta" id="meta"></span></header>
<div class="dua">MIMIC-IV patient-level data — for PhysioNet-credentialed viewers only.
Research demonstration; not for clinical use.</div>
<div class="layout">
<div class="side">
  <h2>Select cases</h2>
  <div class="filters">
    <button onclick="f('highrisk')">Highest risk</button>
    <button onclick="f('lowrisk')">Lowest risk</button>
    <button onclick="f('died')">Died</button>
    <button onclick="f('survived')">Survived</button>
    <button onclick="f('correct')">Model correct</button>
    <button onclick="f('missed')">Model missed</button>
    <button onclick="f('falsealarm')">False alarm</button>
    <button onclick="f('random')">Random</button>
  </div>
  <div style="padding:0 12px 12px">
    <input id="sid" placeholder="look up a stay_id" style="width:100%;
    background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:6px 9px;
    border-radius:5px;font-size:12px">
    <div id="sidmsg" style="font-size:11px;color:#7d8590;margin-top:4px"></div>
  </div>
  <div id="list"></div>
  <div class="tally" id="tally"></div>
  <h2>Verify this is live</h2>
  <div class="filters">
    <button onclick="verify()">Show file hashes</button>
    <button onclick="shuffle()">Destroy the weights</button>
  </div>
  <div id="verify" style="padding:0 16px 18px;font-size:11px;color:#7d8590;
  font-family:ui-monospace,Menlo,Consolas,monospace;word-break:break-all"></div>
</div>
<div class="main" id="main">
  <div class="empty">Choose a case on the left.<br><br>
  Each is a patient held out of training. The model sees only their first
  24&nbsp;hours; the recorded outcome is shown for comparison.</div>
</div></div>
<script>
let CASES=[],SEL=null,TH=0.5;
// Per-patient chart-editing state, reset each time a new case is opened.
let META={},ORIG={},EDIT={},CURROW=null,DRAG=null;
fetch('/meta').then(r=>r.json()).then(d=>{
  TH=d.threshold;
  document.getElementById('meta').textContent=
    d.run+' · '+d.algo.toUpperCase()+' · '+d.n_param.toLocaleString()+' parameters · '
    +d.n.toLocaleString()+' held-out patients · AUROC '+d.auroc.toFixed(3)
    +' · AUPRC '+d.auprc.toFixed(3)+' (prevalence '+(d.prevalence*100).toFixed(1)+'%)';
  document.getElementById('tally').innerHTML=
    'Alert threshold '+(TH*100).toFixed(0)+'%<br>Sensitivity '
    +(d.sens*100).toFixed(0)+'% · Specificity '+(d.spec*100).toFixed(0)+'%';
  f('highrisk');
});
function f(kind){fetch('/cases?kind='+kind).then(r=>r.json()).then(d=>{
  CASES=d;const L=document.getElementById('list');L.innerHTML='';
  d.forEach(c=>{const e=document.createElement('div');e.className='case';
    e.onclick=()=>open_(c.row,e);
    const col=c.risk>=TH?'#f85149':(c.risk>TH/2?'#d29922':'#3fb950');
    e.innerHTML='<div><div class="id">stay '+c.row_id+'</div>'
      +'<div class="unit">'+c.unit+'</div></div>'
      +'<div class="r" style="color:'+col+'">'+(c.risk*100).toFixed(1)+'%</div>';
    L.appendChild(e);});});}
function open_(row,el){
  document.querySelectorAll('.case').forEach(x=>x.classList.remove('sel'));
  if(el)el.classList.add('sel');
  fetch('/case?row='+row).then(r=>r.json()).then(d=>{
    if(d.error){document.getElementById('main').innerHTML=
      '<div class="empty">'+d.error+'</div>';return;}
    const risk=d.risk*100, flagged=d.risk>=TH;
    const col=flagged?'#f85149':(d.risk>TH/2?'#d29922':'#3fb950');
    const outCol=d.died?'#f85149':'#3fb950';
    const correct=(flagged&&d.died)||(!flagged&&!d.died);
    let h='<div class="hdr"><div><h2>Stay '+d.row_id+'</h2>'
      +'<div class="sub">'+d.unit+' · first '+d.hours+' hours of the ICU stay</div></div>'
      +'<div class="verdict">'
      +'<div class="box"><div class="k">Predicted risk</div>'
      +'<div class="v" style="color:'+col+'">'+risk.toFixed(1)+'%</div>'
      +'<div class="n">'+(flagged?'above':'below')+' the '+(TH*100).toFixed(0)+'% alert threshold</div></div>'
      +'<div class="box"><div class="k">Recorded outcome</div>'
      +'<div class="v" style="color:'+outCol+'">'+(d.died?'Died':'Survived')+'</div>'
      +'<div class="n">'+(correct?'model agreed':'model disagreed')+'</div></div>'
      +'</div></div>';
    META={};ORIG={};EDIT={};CURROW=d.row;
    h+='<div class="whatif"><div class="k">Live recomputation &mdash; drag any '
      +'point on a chart below to edit that reading, on as many charts as you '
      +'like, then recompute. Every drag is a real value change fed back '
      +'through the model.</div>'
      +'<button onclick="recompute()">Recompute risk</button>'
      +'<button onclick="resetAll()" style="background:#21262d">Reset all changes</button>'
      +'<div class="wi" id="wi">&nbsp;</div></div>';
    h+='<div class="iv">';
    for(const k in d.iv){h+='<span class="pill'+(d.iv[k].on?' on':'')+'">'
      +d.iv[k].label+': '+(d.iv[k].on?'yes':'no')+'</span>';}
    h+='</div><div class="charts">';
    for(const k in d.series){const s=d.series[k];h+=chart(s,k);}
    h+='</div>';
    document.getElementById('main').innerHTML=h;});}
function diffEdits(){
  const edits={};
  for(const key in EDIT){
    if(EDIT[key].some((v,i)=>v!==ORIG[key][i]))edits[key]=EDIT[key];
  }
  return edits;
}
function recompute(){
  const el=document.getElementById('wi');
  const edits=diffEdits();
  const nFeat=Object.keys(edits).length;
  if(nFeat===0){el.textContent='Drag a point on any chart first, then recompute.';return;}
  el.textContent='recomputing ...';
  fetch('/recompute',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({row:CURROW,edits:edits})}).then(r=>r.json()).then(d=>{
    if(d.error){el.textContent=d.error;return;}
    const b=d.base*100,n=d.risk*100,delta=n-b;
    const col=delta<0?'#3fb950':(delta>0?'#f85149':'#8b949e');
    el.innerHTML='Edited '+d.n_points_edited+' point'+(d.n_points_edited==1?'':'s')
      +' across '+d.n_features_edited+' variable'+(d.n_features_edited==1?'':'s')
      +'.<br>Recorded '+b.toFixed(1)+'% &rarr; <b style="color:'+col+'">'
      +n.toFixed(1)+'%</b> <span style="color:'+col+'">('+(delta>0?'+':'')
      +delta.toFixed(1)+' points)</span> &mdash; computed just now by the model.';});}
function resetChart(key){
  EDIT[key]=ORIG[key].slice();
  const box=document.getElementById('box-'+key);
  if(box)box.outerHTML=renderChart(key);
  document.getElementById('wi').textContent=
    'Chart reset. Drag another point, or recompute.';}
function resetAll(){
  for(const key in ORIG){
    EDIT[key]=ORIG[key].slice();
    const box=document.getElementById('box-'+key);
    if(box)box.outerHTML=renderChart(key);
  }
  document.getElementById('wi').innerHTML='&nbsp;';}
function verify(){fetch('/verify').then(r=>r.json()).then(d=>{
  document.getElementById('verify').innerHTML=
   'model&nbsp; '+d.model_path+'<br>sha256 '+d.model_sha256
   +'<br>saved&nbsp; '+d.model_mtime
   +'<br>weights sum '+d.weight_checksum+' over '+d.n_param.toLocaleString()+' params'
   +'<br><br>data&nbsp;&nbsp; '+d.data_path+'<br>sha256 '+d.data_sha256
   +'<br>built&nbsp; '+d.data_mtime
   +'<br><br>Verify the hashes on disk with certutil or sha256sum.';});}
function shuffle(){
  document.getElementById('verify').innerHTML='re-scoring with random weights ...';
  fetch('/shuffle').then(r=>r.json()).then(d=>{
    document.getElementById('verify').innerHTML=
     'Re-scored '+d.n+' patients twice:<br><br>'
     +'trained weights&nbsp; AUROC '+d.auroc_trained.toFixed(3)+'<br>'
     +'random weights&nbsp;&nbsp; AUROC '+d.auroc_shuffled.toFixed(3)+'<br><br>'
     +'A stored table of numbers would not change. The trained weights have '
     +'been restored.';});}
document.getElementById('sid').addEventListener('keydown',e=>{
  if(e.key!=='Enter')return;
  fetch('/lookup?stay_id='+e.target.value).then(r=>r.json()).then(d=>{
    const m=document.getElementById('sidmsg');
    if(d.error){m.textContent=d.error;m.style.color='#d29922';}
    else{m.textContent='';open_(d.row,null);}});});
// META/ORIG/EDIT are declared once, at the top of the script (see below), and
// reset for each new patient in open_(). ORIG holds the values as recorded;
// EDIT holds the working copy the user is dragging; a point differs from ORIG
// exactly when the user has changed it.
function X_(key,i){const m=META[key];return m.P+i*(m.W-2*m.P)/Math.max(m.n-1,1);}
function Y_(key,y){const m=META[key];
  return m.H-m.P-((y-m.lo)/Math.max(m.hi-m.lo,1e-9))*(m.H-2*m.P);}

function chart(s,key){
  META[key]={lo:s.lo,hi:s.hi,W:300,H:74,P:6,n:s.values.length,
             label:s.label,unit:s.unit};
  ORIG[key]=s.values.slice();
  EDIT[key]=s.values.slice();
  // widen lo/hi to fit any out-of-range recorded values so the line is never
  // clipped
  const obsVals=s.values.filter(v=>v!==null);
  if(obsVals.length){
    META[key].lo=Math.min(s.lo,...obsVals);
    META[key].hi=Math.max(s.hi,...obsVals);
  }
  return renderChart(key);
}

function renderChart(key){
  const m=META[key],v=EDIT[key];
  const idx=v.map((y,i)=>i).filter(i=>v[i]!==null);
  let d='',started=false;
  idx.forEach(i=>{d+=(started?'L':'M')+X_(key,i).toFixed(1)+' '+Y_(key,v[i]).toFixed(1)+' ';started=true;});
  const edited=idx.some(i=>v[i]!==ORIG[key][i]);
  const dots=idx.map(i=>{
    const ch=v[i]!==ORIG[key][i];
    return '<circle id="pt-'+key+'-'+i+'" cx="'+X_(key,i).toFixed(1)+'" cy="'
      +Y_(key,v[i]).toFixed(1)+'" r="'+(ch?3.2:2.4)+'" fill="'
      +(ch?'#f0883e':'#58a6ff')+'" style="cursor:ns-resize" '
      +'onpointerdown="startDrag(event,\''+key+'\','+i+')"/>';
  }).join('');
  const last=idx.length?v[idx[idx.length-1]]:null;
  const lastCh=idx.length&&v[idx[idx.length-1]]!==ORIG[key][idx[idx.length-1]];
  return '<div class="chart'+(edited?' edited':'')+'" id="box-'+key+'">'
    +'<div class="t"><b>'+m.label+'</b><span>'
    +(last===null?'—':'<span'+(lastCh?' style="color:#f0883e"':'')+'>'
      +last.toFixed(1)+'</span> '+m.unit)
    +(edited?' <a onclick="resetChart(\''+key+'\')" title="reset this chart" '
      +'style="cursor:pointer;color:#7d8590;margin-left:4px">&#8635;</a>':'')
    +'</span></div>'
    +'<svg id="svg-'+key+'" viewBox="0 0 '+m.W+' '+m.H+'" preserveAspectRatio="none" '
    +'style="touch-action:none">'
    +'<path id="path-'+key+'" d="'+d+'" fill="none" stroke="'
    +(edited?'#f0883e':'#58a6ff')+'" stroke-width="1.6"/>'+dots+'</svg>'
    +'<div class="t" style="margin-top:2px"><span>'+idx.length+' measurements'
    +(edited?' &mdash; edited':'')+'</span>'
    +'<span>'+m.lo.toFixed(0)+'–'+m.hi.toFixed(0)+'</span></div></div>';
}

// Drag handling. While dragging we only patch the moved circle and the path
// (cheap, smooth); the full chart -- colours, reset icon, header value -- is
// re-rendered once on release.
function startDrag(ev,key,i){ev.preventDefault();DRAG={key,i};document.body.style.userSelect='none';}
document.addEventListener('pointermove',e=>{
  if(!DRAG)return;
  const svg=document.getElementById('svg-'+DRAG.key);
  if(!svg)return;
  const r=svg.getBoundingClientRect(), m=META[DRAG.key];
  const py=Math.min(Math.max((e.clientY-r.top)*(m.H/r.height),m.P),m.H-m.P);
  const val=m.lo+((m.H-m.P-py)/(m.H-2*m.P))*(m.hi-m.lo);
  EDIT[DRAG.key][DRAG.i]=Math.round(val*10)/10;
  const c=document.getElementById('pt-'+DRAG.key+'-'+DRAG.i);
  if(c)c.setAttribute('cy',Y_(DRAG.key,EDIT[DRAG.key][DRAG.i]).toFixed(1));
  const p=document.getElementById('path-'+DRAG.key);
  if(p){
    const idx=EDIT[DRAG.key].map((y,ii)=>ii).filter(ii=>EDIT[DRAG.key][ii]!==null);
    let d='',started=false;
    idx.forEach(ii=>{d+=(started?'L':'M')+X_(DRAG.key,ii).toFixed(1)+' '
      +Y_(DRAG.key,EDIT[DRAG.key][ii]).toFixed(1)+' ';started=true;});
    p.setAttribute('d',d);
  }
});
document.addEventListener('pointerup',()=>{
  if(DRAG){const k=DRAG.key;DRAG=null;
    const box=document.getElementById('box-'+k);
    if(box)box.outerHTML=renderChart(k);}
  document.body.style.userSelect='';});
</script></body></html>"""


def run(port=8000):
    print("Loading model and scoring the held-out test split ...")
    S = Store()
    print(f"  {S.run} | {S.algo.upper()} | {S.n_param:,} parameters")
    print(f"  {len(S.cases):,} held-out patients scored")
    print(f"  AUROC {S.metrics['auroc']:.3f}  AUPRC {S.metrics['auprc']:.3f}"
          f"  (prevalence {S.prevalence*100:.1f}%)")
    print(f"  stay_id range {int(S.stay_id.min()):,}-{int(S.stay_id.max()):,} "
          f"— real MIMIC-IV identifiers")

    rng = np.random.default_rng(0)

    def select(kind, k=40):
        cs = S.cases
        th = S.threshold
        if kind == "highrisk":
            return cs[:k]
        if kind == "lowrisk":
            return cs[-k:][::-1]
        if kind == "died":
            return [c for c in cs if c["died"]][:k]
        if kind == "survived":
            return [c for c in cs if not c["died"]][:k]
        if kind == "correct":
            return [c for c in cs if (c["risk"] >= th) == bool(c["died"])][:k]
        if kind == "missed":      # died but not flagged
            return [c for c in cs if c["died"] and c["risk"] < th][:k]
        if kind == "falsealarm":  # flagged but survived
            return [c for c in cs if not c["died"] and c["risk"] >= th][:k]
        idx = rng.choice(len(cs), min(k, len(cs)), replace=False)
        return [cs[i] for i in idx]

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj, ctype="application/json"):
            body = (obj if isinstance(obj, bytes)
                    else json.dumps(obj).encode())
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                self._route()
            except Exception as e:
                # A malformed request must not print a traceback in front of an
                # audience, nor take the server down.
                self._send({"error": f"{type(e).__name__}: {e}"})

        def _route(self):
            u = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(u.query)

            def as_int(name):
                v = (q.get(name) or [""])[0]
                if not v or not v.lstrip("-").isdigit():
                    raise ValueError(f"'{name}' must be a number, got '{v}'")
                return int(v)
            if u.path == "/meta":
                self._send({"run": S.run, "algo": S.algo, "n_param": S.n_param,
                            "n": len(S.cases), "auroc": S.metrics["auroc"],
                            "auprc": S.metrics["auprc"],
                            "sens": S.metrics["sensitivity"],
                            "spec": S.metrics["specificity"],
                            "prevalence": S.prevalence,
                            "threshold": S.threshold})
            elif u.path == "/cases":
                sel = select(q.get("kind", ["highrisk"])[0])
                self._send([{"row": c["row"],
                             "row_id": int(S.stay_id[c["row"]]),
                             "unit": c["client"].replace(
                                 " Intensive Care Unit", "")[:26],
                             "risk": c["risk"], "died": c["died"]} for c in sel])
            elif u.path == "/case":
                row = as_int("row")
                c = S.by_row.get(row)
                if c is None:
                    self._send({"error": f"row {row} is not in the test split"})
                    return
                ser, iv = S.series(row)
                self._send({"row": row,
                            "row_id": int(S.stay_id[row]),
                            "unit": c["client"].replace(
                                " Intensive Care Unit", ""),
                            "hours": C.OBS_WINDOW_H,
                            "risk": c["risk"], "died": c["died"],
                            "series": ser, "iv": iv})
            elif u.path == "/verify":
                self._send(S.provenance())
            elif u.path == "/shuffle":
                self._send(S.shuffled_scores())
            elif u.path == "/lookup":
                try:
                    sid = as_int("stay_id")
                except ValueError:
                    self._send({"error": "enter a numeric stay_id"}); return
                c = S.by_stay.get(sid)
                if c is None:
                    self._send({"error": f"stay {sid} is not in the held-out "
                                         f"test split"})
                else:
                    self._send({"row": c["row"]})
            else:
                self._send(HTML.encode(), "text/html; charset=utf-8")

        def do_POST(self):
            try:
                self._route_post()
            except Exception as e:
                self._send({"error": f"{type(e).__name__}: {e}"})

        def _route_post(self):
            u = urllib.parse.urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                self._send({"error": "malformed request body"}); return

            if u.path == "/recompute":
                # `edits`: {feature_name: [24 values-or-null, one per hour]}.
                # Several features can be edited together and are scored in ONE
                # forward pass, so the panel reports the COMBINED effect of
                # every change made on the charts, not one change at a time.
                try:
                    row = int(payload.get("row"))
                except (TypeError, ValueError):
                    self._send({"error": "row must be a number"}); return
                if row not in S.by_row:
                    self._send({"error": "unknown patient"}); return
                edits = payload.get("edits") or {}
                if not isinstance(edits, dict):
                    self._send({"error": "edits must be an object"}); return
                clean = {k: v for k, v in edits.items()
                        if k in S.names and isinstance(v, list)}
                risk, n_points = S.predict_row(row, clean)
                self._send({"base": S.by_row[row]["risk"], "risk": risk,
                            "n_features_edited": len(clean),
                            "n_points_edited": n_points})
            else:
                self._send({"error": "not found"})

    # A previous run can leave the port held (Windows in particular keeps it in
    # TIME_WAIT). Reuse the address, and if it is genuinely occupied by another
    # process, move to the next free port rather than crashing.
    class Server(socketserver.TCPServer):
        allow_reuse_address = True

    httpd = None
    for cand in range(port, port + 12):
        try:
            httpd = Server(("", cand), H)
            if cand != port:
                print(f"  (port {port} was busy — using {cand} instead)")
            port = cand
            break
        except OSError:
            continue
    if httpd is None:
        print(f"\n  Could not bind any port in {port}-{port+11}.")
        print("  Close the other demo window, or pass a port explicitly:")
        print("      python demo_app.py 9000")
        sys.exit(1)

    print(f"\n  Data:  {C.TS_NPZ}")
    print(f"         real MIMIC-IV stays; IDs shown are genuine stay_id values")
    print(f"\n  Open:  http://localhost:{port}")
    print("  Ctrl+C to stop.\n")
    with httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("stopped.")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)