"""
Central configuration for the federated MIMIC-IV mortality prediction pipeline.
Edit MIMIC_ROOT to point at your unzipped MIMIC-IV v3.1 download.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
# Point this at the folder that directly contains hosp/ and icu/.
# Windows: use a raw string, e.g. Path(r"D:\fl_ly\physionet.org\files\mimiciv\3.1")
# The layout inside (.csv.gz, extracted .csv, or nested folders) does not matter.
MIMIC_ROOT = Path(r"D:\fl_ly\physionet.org\files\mimiciv\3.1")   # <-- CHANGE ME
HOSP = MIMIC_ROOT / "hosp"
ICU = MIMIC_ROOT / "icu"

OUT_DIR = Path("./artifacts_combined")
OUT_DIR.mkdir(exist_ok=True, parents=True)


# ----------------------------------------------------------------------------
# Table locator
# ----------------------------------------------------------------------------
# MIMIC-IV arrives in several layouts depending on how it was downloaded and
# whether it was extracted. All of these are handled:
#     hosp/admissions.csv.gz                  (canonical PhysioNet download)
#     hosp/admissions.csv                     (extracted, flat)
#     hosp/admissions.csv/admissions.csv      (extracted by tools that create a
#                                              folder per archive -- common on
#                                              Windows with 7-Zip/WinRAR)
#     hosp/admissions/admissions.csv
# Compressed files are NOT slower here -- DuckDB streams gzip fine -- so there is
# no need to extract, and no need to re-compress if you already have.
_TABLE_CACHE = {}


def find_table(module, name):
    """Locate a MIMIC table file. `module` is 'hosp' or 'icu'. Returns Path|None."""
    key = (module, name)
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]

    base = MIMIC_ROOT / module
    candidates = [
        base / f"{name}.csv.gz",
        base / f"{name}.csv",
        base / f"{name}.csv" / f"{name}.csv",
        base / f"{name}.csv.gz" / f"{name}.csv.gz",
        base / f"{name}.csv.gz" / f"{name}.csv",
        base / name / f"{name}.csv.gz",
        base / name / f"{name}.csv",
    ]
    found = next((c for c in candidates if c.is_file()), None)

    if found is None and base.is_dir():
        # last resort: search the module folder (prefer .gz, then shallowest)
        hits = [p for p in list(base.rglob(f"{name}.csv.gz"))
                        + list(base.rglob(f"{name}.csv")) if p.is_file()]
        if hits:
            found = sorted(hits, key=lambda p: (not p.suffix == ".gz", len(p.parts)))[0]

    _TABLE_CACHE[key] = found
    return found


def csv_reader(module, name, types=None, ignore_errors=False):
    """DuckDB read_csv_auto(...) expression for a table, layout-agnostic.

    Sets compression only for .gz, uses forward slashes (Windows backslashes are
    escape characters inside SQL string literals), and lets callers pin column
    types -- DuckDB infers from a sample, and columns like `valuenum` that are
    empty in the first few thousand rows can otherwise be inferred as VARCHAR.
    """
    path = find_table(module, name)
    if path is None:
        raise FileNotFoundError(
            f"Could not find '{name}' under {MIMIC_ROOT / module}. "
            f"Run `python validate_dataset.py` to see what was found.")
    p = path.as_posix()
    opts = []
    if path.suffix == ".gz":
        opts.append("compression='gzip'")
    if types:
        t = ", ".join(f"'{k}': '{v}'" for k, v in types.items())
        opts.append(f"types={{{t}}}")
    if ignore_errors:
        # ONLY for a known-truncated file being handled deliberately by
        # recover_truncated.py. Never enable this to make an unexplained parse
        # error go away -- it discards rows silently.
        opts.append("ignore_errors=true")
    return f"read_csv_auto('{p}'" + ("," + ",".join(opts) if opts else "") + ")"

COHORT_PQ = OUT_DIR / "cohort.parquet"
EVENTS_PQ = OUT_DIR / "events_long.parquet"
TS_NPZ = OUT_DIR / "timeseries.npz"
AGG_PQ = OUT_DIR / "features_agg.parquet"

# ----------------------------------------------------------------------------
# Cohort definition
# ----------------------------------------------------------------------------
OBS_WINDOW_H = 24        # hours of data used as model input (24 or 48)
MIN_AGE = 18
MIN_LOS_H = 24           # require full observation window to be observable
FIRST_STAY_ONLY = True   # one ICU stay per hospital admission

# Leakage guard: drop stays where death occurred at/before the end of the
# observation window. Predicting death for an already-dead patient is not a
# prediction task, and leaving these in inflates every metric.
DROP_DEATH_IN_WINDOW = True

# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
# name -> (source_table, [itemids], (valid_min, valid_max))
# Values outside the valid range are treated as charting errors and dropped.
CHART_FEATURES = {
    "heart_rate":   ([220045],           (10, 300)),
    "sbp":          ([220050, 220179],   (20, 300)),
    "dbp":          ([220051, 220180],   (5, 200)),
    "map":          ([220052, 220181],   (10, 250)),
    "resp_rate":    ([220210],           (0, 70)),
    "spo2":         ([220277],           (30, 100)),
    "temp_f":       ([223761],           (70, 120)),   # converted to C below
    "temp_c":       ([223762],           (25, 45)),
    "gcs_eye":      ([220739],           (0, 5)),
    "gcs_verbal":   ([223900],           (0, 5)),
    "gcs_motor":    ([223901],           (0, 6)),
    # --- added: respiratory support settings ---
    "fio2":         ([223835],           (21, 100)),
    "peep":         ([220339],           (0, 40)),
    "tidal_volume": ([224685],           (0, 2000)),
}

# Vasopressor administration (inputevents). Being on pressors is one of the
# single strongest mortality signals in the ICU and was missing entirely.
VASOPRESSOR_ITEMIDS = {
    "norepinephrine": [221906],
    "epinephrine":    [221289],
    "dopamine":       [221662],
    "dobutamine":     [221653],
    "vasopressin":    [222315],
    "phenylephrine":  [221749],
}

# Mechanical ventilation (procedureevents) -- also a major mortality signal.
VENT_ITEMIDS = [225792, 225794]     # invasive, non-invasive

# ---------------------------------------------------------------------------
# Comorbidities (Charlson, Quan et al. 2005 coding)
# ---------------------------------------------------------------------------
# WARNING -- LABEL LEAKAGE. Read this before enabling.
#
# ICD codes in MIMIC-IV are BILLING codes, assigned at hospital discharge by
# coders reading the signed notes ("Diagnoses are billed on hospital discharge"
# -- MIMIC-IV documentation, diagnoses_icd). They are NOT available during the
# first 24h of an ICU stay.
#
# Using the CURRENT admission's codes to predict that admission's outcome is
# same-admission label leakage: a patient coded for "cardiac arrest" or "sepsis"
# on day 5 is trivially identifiable as high risk on day 1. Models trained on
# MIMIC-IV ICD codes ALONE reach AUROC 0.97-0.98 for in-hospital mortality
# (Ludwig et al., medRxiv 2025), and ~40% of published MIMIC prediction models
# contain this flaw.
#
#   "current"  -- LEAKY. Current admission's codes. Do not use for publication.
#   "prior"    -- SAFE. Codes from the patient's PREVIOUS completed admissions
#                 only (dischtime < ICU intime). Genuinely known at prediction
#                 time; this is what "comorbidity burden on admission" means.
#   "none"     -- no comorbidity features at all.
#   "both"     -- extract BOTH, prefixed cci_prior_* and cci_curr_*, so the
#                 leakage experiment can compare them without re-extracting.
#                 run_experiments.py needs this.
COMORBIDITY_SOURCE = "both"

# Escape hatch used ONLY by run_experiments.py's E3, which deliberately trains a
# leaky arm in order to measure the inflation. Never set this True for a model
# whose numbers you intend to report.
ALLOW_LEAKY_ICD = False
USE_COMORBIDITIES = COMORBIDITY_SOURCE != "none"

CHARLSON = {
    # condition: (weight, icd9 prefixes, icd10 prefixes)
    "myocardial_infarction":  (1, ["410", "412"], ["I21", "I22", "I252"]),
    "congestive_heart_failure": (1, ["428", "4254", "4255", "4256", "4257",
                                     "4258", "4259", "39891", "40201", "40211",
                                     "40291", "40401", "40403", "40411", "40413",
                                     "40491", "40493"],
                                 ["I099", "I110", "I130", "I132", "I255", "I420",
                                  "I425", "I426", "I427", "I428", "I429", "I43",
                                  "I50", "P290"]),
    "peripheral_vascular":    (1, ["440", "441", "0930", "4373", "4471", "5571",
                                   "5579", "V434"],
                               ["I70", "I71", "I731", "I738", "I739", "I771",
                                "I790", "I792", "K551", "K558", "K559"]),
    "cerebrovascular":        (1, ["430", "431", "432", "433", "434", "435",
                                   "436", "437", "438", "36234"],
                               ["G45", "G46", "H340", "I60", "I61", "I62", "I63",
                                "I64", "I65", "I66", "I67", "I68", "I69"]),
    "dementia":               (1, ["290", "2941", "3312"],
                               ["F00", "F01", "F02", "F03", "F051", "G30", "G311"]),
    "chronic_pulmonary":      (1, ["490", "491", "492", "493", "494", "495",
                                   "496", "500", "501", "502", "503", "504",
                                   "505", "4168", "4169", "5064", "5081", "5088"],
                               ["I278", "I279", "J40", "J41", "J42", "J43", "J44",
                                "J45", "J46", "J47", "J60", "J61", "J62", "J63",
                                "J64", "J65", "J66", "J67", "J684", "J701", "J703"]),
    "rheumatic":              (1, ["7100", "7101", "7102", "7103", "7104",
                                   "7140", "7141", "7142", "7148", "725", "4465"],
                               ["M05", "M06", "M315", "M32", "M33", "M34",
                                "M351", "M353", "M360"]),
    "peptic_ulcer":           (1, ["531", "532", "533", "534"],
                               ["K25", "K26", "K27", "K28"]),
    "mild_liver_disease":     (1, ["570", "571", "5733", "5734", "5738", "5739",
                                   "V427", "07022", "07023", "07032", "07033",
                                   "07044", "07054", "0706", "0709"],
                               ["B18", "K700", "K701", "K702", "K703", "K709",
                                "K713", "K714", "K715", "K717", "K73", "K74",
                                "K760", "K762", "K763", "K764", "K768", "K769",
                                "Z944"]),
    "diabetes_uncomplicated": (1, ["2500", "2501", "2502", "2503", "2508", "2509"],
                               ["E100", "E101", "E106", "E108", "E109", "E110",
                                "E111", "E116", "E118", "E119", "E120", "E121",
                                "E126", "E128", "E129", "E130", "E131", "E136",
                                "E138", "E139", "E140", "E141", "E146", "E148",
                                "E149"]),
    "diabetes_complicated":   (2, ["2504", "2505", "2506", "2507"],
                               ["E102", "E103", "E104", "E105", "E107", "E112",
                                "E113", "E114", "E115", "E117", "E122", "E123",
                                "E124", "E125", "E127", "E132", "E133", "E134",
                                "E135", "E137", "E142", "E143", "E144", "E145",
                                "E147"]),
    "hemiplegia":             (2, ["342", "343", "3341", "3440", "3441", "3442",
                                   "3443", "3444", "3445", "3446", "3449"],
                               ["G041", "G114", "G801", "G802", "G81", "G82",
                                "G830", "G831", "G832", "G833", "G834", "G839"]),
    "renal_disease":          (2, ["582", "585", "586", "5830", "5831", "5832",
                                   "5834", "5836", "5837", "5880", "V420",
                                   "V451", "V56", "40301", "40311", "40391",
                                   "40402", "40403", "40412", "40413", "40492",
                                   "40493"],
                               ["I120", "I131", "N032", "N033", "N034", "N035",
                                "N036", "N037", "N052", "N053", "N054", "N055",
                                "N056", "N057", "N18", "N19", "N250", "Z490",
                                "Z491", "Z492", "Z940", "Z992"]),
    "malignancy":             (2, ["140", "141", "142", "143", "144", "145",
                                   "146", "147", "148", "149", "150", "151",
                                   "152", "153", "154", "155", "156", "157",
                                   "158", "159", "160", "161", "162", "163",
                                   "164", "165", "170", "171", "172", "174",
                                   "175", "176", "179", "180", "181", "182",
                                   "183", "184", "185", "186", "187", "188",
                                   "189", "190", "191", "192", "193", "194",
                                   "195", "200", "201", "202", "203", "204",
                                   "205", "206", "207", "208", "2386"],
                               ["C0", "C1", "C2", "C30", "C31", "C32", "C33",
                                "C34", "C37", "C38", "C39", "C40", "C41", "C43",
                                "C45", "C46", "C47", "C48", "C49", "C5", "C6",
                                "C70", "C71", "C72", "C73", "C74", "C75", "C76",
                                "C81", "C82", "C83", "C84", "C85", "C88", "C90",
                                "C91", "C92", "C93", "C94", "C95", "C96", "C97"]),
    "severe_liver_disease":   (3, ["4560", "4561", "4562", "5722", "5723",
                                   "5724", "5728"],
                               ["I850", "I859", "I864", "I982", "K704", "K711",
                                "K721", "K729", "K765", "K766", "K767"]),
    "metastatic_cancer":      (6, ["196", "197", "198", "199"],
                               ["C77", "C78", "C79", "C80"]),
    "aids_hiv":               (6, ["042", "043", "044"],
                               ["B20", "B21", "B22", "B24"]),
}

LAB_FEATURES = {
    # original set
    "creatinine":   ([50912], (0, 30)),
    "potassium":    ([50971], (1, 12)),
    "sodium":       ([50983], (90, 200)),
    "chloride":     ([50902], (60, 160)),
    "bicarbonate":  ([50882], (5, 60)),
    "hematocrit":   ([51221], (5, 70)),
    "wbc":          ([51301], (0, 200)),
    "glucose":      ([50931], (10, 1500)),
    "magnesium":    ([50960], (0, 10)),
    "calcium":      ([50893], (2, 20)),
    "lactate":      ([50813], (0, 40)),
    # --- added: organ-failure markers that drive the SOFA/APACHE scores ---
    "platelets":    ([51265], (0, 2000)),    # coagulation / SOFA
    "bilirubin":    ([50885], (0, 60)),      # liver / SOFA
    "bun":          ([51006], (0, 300)),     # renal, strong mortality signal
    "inr":          ([51237], (0, 20)),      # coagulopathy
    "ptt":          ([51275], (0, 200)),
    "albumin":      ([50862], (0, 10)),      # nutrition/inflammation
    "alt":          ([50861], (0, 10000)),
    "ast":          ([50878], (0, 10000)),
    "hemoglobin":   ([51222], (2, 25)),
    "anion_gap":    ([50868], (0, 60)),
    "ph":           ([50820], (6.5, 8.0)),   # blood gas
    "po2":          ([50821], (10, 700)),
    "pco2":         ([50818], (5, 200)),
    "base_excess":  ([50802], (-40, 40)),
    "phosphate":    ([50970], (0, 20)),
    "troponin":     ([51003], (0, 50)),
}

# Urine output itemids (outputevents). Summed per hour rather than averaged.
URINE_ITEMIDS = [
    226559, 226560, 226561, 226584, 226563, 226564,
    226565, 226567, 226557, 226558, 227488, 227489,
]

# Final ordered feature list used by the time-series model
TS_FEATURES = [
    # vitals
    "heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2", "temp_c",
    "gcs_eye", "gcs_verbal", "gcs_motor",
    "fio2", "peep", "tidal_volume",
    # labs
    "creatinine", "potassium", "sodium", "chloride", "bicarbonate",
    "hematocrit", "wbc", "glucose", "magnesium", "calcium", "lactate",
    "platelets", "bilirubin", "bun", "inr", "ptt", "albumin", "alt", "ast",
    "hemoglobin", "anion_gap", "ph", "po2", "pco2", "base_excess",
    "phosphate", "troponin",
    # outputs and interventions
    "urine",
    "vasopressor", "ventilation",
]

STATIC_FEATURES = ["age", "gender_m", "admission_emergency"]

# ----------------------------------------------------------------------------
# Partitioning (Section 5 of the design doc)
# ----------------------------------------------------------------------------
PARTITION_SCHEME = "careunit"   # "careunit" | "shuffled" | "year_group" | "dirichlet"
#   careunit  -- real ICU units; genuinely non-IID (mortality 2.3-15.3%)
#   shuffled  -- IID CONTROL: identical client sizes, randomised membership.
#                Isolates heterogeneity by holding client size fixed, which
#                dirichlet does not.
MIN_CLIENT_SIZE = 500           # merge/drop clients smaller than this
DIRICHLET_ALPHA = 0.3           # only used when PARTITION_SCHEME == "dirichlet"
DIRICHLET_N_CLIENTS = 8

TEST_FRAC = 0.15                # per-client held-out test
VAL_FRAC = 0.15                 # per-client validation
SEED = 42

# ----------------------------------------------------------------------------
# Federated training
# ----------------------------------------------------------------------------
FL_ALGO = "fedprox"      # controlled comparison (compare_algorithms.py, 5 seeds)
                         # ranked FedAdam > FedAvg = FedProx. FedAdam at
                         # SERVER_LR=0.01 won 5/5 seeds against every other arm
                         # (+0.0103 AUPRC vs FedAvg, Cohen's d = 1.47).
MODEL = "hybrid"         # "hybrid" (RECOMMENDED) | "gru" | "mlp"
                         #   hybrid = GRU over the time series PLUS an MLP over
                         #   the aggregate features, fused before the head. The
                         #   two views are complementary: the GRU sees trajectory,
                         #   the tabular branch sees extremes and counts that a
                         #   24-step sequence can wash out.

# Append a "time since last measurement" channel per feature (GRU-D style).
# In the ICU, HOW OFTEN something is measured is itself informative -- an unstable
# patient gets hourly labs, a stable one gets daily. The mask alone says "missing";
# the delta says "missing for 9 hours", which is a much stronger signal.
# Measured empirically: NO benefit, because the aggregate branch already carries
# a per-feature measurement COUNT, which encodes the same information. Left here
# because it may help if you ever drop the tabular branch. Costs ~14% more
# parameters for nothing when MODEL="hybrid".
USE_DELTA = False

# Bidirectional GRU. The 24h window is fixed and fully observed at prediction
# time, so there is no leakage in reading it backwards as well as forwards.
# Measured empirically: slightly WORSE (-0.003 AUROC) and doubles the GRU
# parameters. A 24-step sequence apparently does not need a backward pass to be
# read well. Left configurable for the ablation table.
BIDIRECTIONAL = False
PERSONALIZE = False      # FedPer: keep the classifier head local per client

# FedBN (Li et al., ICLR 2021): keep BatchNorm statistics local instead of
# averaging them. Averaging BN running_mean/var across non-IID clients mixes
# incompatible feature distributions and is a known failure mode. Only affects
# MODEL="mlp" -- the GRU has no BatchNorm.
FEDBN = True

# ---------------------------------------------------------------------------
# Branch-wise federation  (proposed method)
# ---------------------------------------------------------------------------
# Standard personalisation is all-or-nothing: share the body, localise the head.
# But the two branches of the hybrid model see structurally different information:
#
#   gru / seq_norm  -- physiology (HR, lactate, pressors). The mapping from
#                      deranged physiology to death is broadly UNIVERSAL across
#                      ICUs, so this branch benefits from pooling everyone's data.
#   tab             -- comorbidities, demographics, admission type. These are
#                      strongly UNIT-SPECIFIC: CVICU is cardiac surgery, Neuro
#                      Intermediate is neurology. Averaging this branch across
#                      units blends incompatible case-mixes.
#
# So federate the first and localise the second. Empirically motivated by the
# observation that the federated-centralized gap widened sharply when
# unit-specific comorbidity features were added.
#
# Set to [] to disable (recovers standard full federation).
# ["tab", "head"] federates only the temporal branch.
# ["head"] is equivalent to classic FedPer.
PERSONAL_BRANCHES = []

# ---------------------------------------------------------------------------
# Aggregation weighting
# ---------------------------------------------------------------------------
# FedAvg weights each client by n_k (sample count). With a rare outcome that is
# the wrong currency: the gradient signal comes from the POSITIVES, and your
# clients differ ~20x in event count (25 to 493 positives) while differing only
# ~3x in sample count. A client with 3,227 samples and 25 deaths should not
# outweigh one with 1,763 samples and 236 deaths.
#
#   "samples"   -- classic FedAvg, weight by n_k
#   "events"    -- weight by number of positive cases
#   "effective" -- weight by the harmonic mean of positives and negatives,
#                  i.e. the effective sample size for a binary task:
#                      n_eff = 2 * n_pos * n_neg / (n_pos + n_neg)
#                  This reduces to n_k/2 when balanced and to ~2*n_pos when
#                  positives are scarce, so it degrades gracefully.
AGG_WEIGHT = "samples"

# Class imbalance. Pick ONE -- applying both a balanced sampler and a weighted
# loss double-corrects, which over-predicts the positive class and wrecks
# calibration (Brier score). Applied strictly locally per client either way.
#   "pos_weight" : weight the positive class in the loss  (recommended)
#   "sampler"    : class-balanced resampling of the local training set
#   "none"       : no correction (rely on AUPRC only)
IMBALANCE = "pos_weight"

# Post-hoc Platt scaling on each client's validation split. Any imbalance
# correction leaves probabilities badly calibrated; this fixes Brier/calibration
# without touching discrimination (AUROC/AUPRC are rank-based and unchanged).
CALIBRATE = True

NUM_ROUNDS = 40
LOCAL_EPOCHS = 2
BATCH_SIZE = 256         # 24 GB VRAM -- room for far more than the old 128
LR = 0.002065
PROX_MU = 0.01           # FedProx proximal strength
SERVER_LR = 0.01         # best of {0.003, 0.01, 0.03} under controlled comparison
HIDDEN = 128
DROPOUT = 0.3

EARLY_STOP_PATIENCE = 8  # rounds without global val AUPRC improvement

# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
# Bootstrap replicates for 95% CIs on every metric. 1000 is the usual reporting
# standard; drop to 200 for quick iteration, raise to 2000 for the final run.
N_BOOTSTRAP = 2000       # 24 threads available; 2000 is the stricter reporting standard

# ----------------------------------------------------------------------------
# Hardware
# ----------------------------------------------------------------------------
# Tuned for: RTX A5000 (24 GB, Ampere sm_86), Xeon W-2265 (12c/24t), 128 GB RAM.
DUCKDB_THREADS = 20            # leave a few cores for the OS
DUCKDB_MEMORY = "96GB"

# Keep the whole dataset resident in VRAM instead of streaming batches from CPU.
# The tensors are tiny (~250 MB for 24h x 22 features x ~50k stays) and the models
# are small, so host->device copies dominate runtime otherwise. Expect a large
# speedup. Set False only if you extend the feature set enough to exhaust 24 GB.
GPU_RESIDENT = True

# --- throughput knobs -------------------------------------------------------
# TF32 matmuls: Ampere (A5000) executes these on tensor cores at ~8x FP32 rate,
# with precision far beyond what a 200k-parameter GRU on noisy clinical data
# needs. Free speedup.
TF32 = True

# cuDNN autotuner. Our input shapes are fixed (batch x 24 x 44), so it picks the
# best kernels once and reuses them.
CUDNN_BENCHMARK = True

# torch.compile with CUDA graphs. A small GRU is *kernel-launch bound* -- 24
# sequential timesteps of tiny kernels means the GPU idles between launches.
# CUDA graphs replay the whole step as one submission and can cut round time
# substantially. Costs 30-60s of compilation up front, so it only pays off on
# long runs: leave False for a single 40-round fit, set True for run_ablation.py
# (which trains 8+ models).
COMPILE = False

# Parallel workers for bootstrap CIs. -1 uses every core. This, not the GPU, was
# the real bottleneck: 2000 replicates x ~16 evaluations single-threaded took
# over two hours, versus ~25 seconds for the actual federated training.
BOOTSTRAP_JOBS = -1