# Federated Learning for ICU Mortality Prediction

Federated learning for 24-hour ICU mortality prediction, trained across 78
real institutional clients (9 MIMIC-IV care units + 69 eICU-CRD hospitals)
without pooling patient data.

## ⚠️ Data and model availability

**This repository contains code and final results, including graphs and result CSV files.** No patient data, no derived
tensors (`.npz`/`.parquet`), and no trained model weights (`.pt`) are
included, per PhysioNet's own policy that derived models trained on
MIMIC-IV/eICU-CRD carry the same access restriction as the source data
(see `physionet.org/content/mimiciv/3.1/`, "Sharing MIMIC data" section).

- **To reproduce results**, obtain independent PhysioNet credentialing for
  [MIMIC-IV](https://physionet.org/content/mimiciv/) and
  [eICU-CRD](https://physionet.org/content/eicu-crd/), then run this code
  against your own local copies.
- **To use the trained model**, request access via our PhysioNet project:
  `[link once submitted]`.

## Project description

Federated learning (FL) is the standard technical response to a real
constraint: clinical data cannot be pooled across institutions, but
prediction models generally improve with more diverse training data.
Whether FL actually delivers on that promise is contested in the
literature — published estimates of its benefit range from "barely better
than training locally" to "clearly and substantially better," and most of
that evidence comes from heterogeneity *simulated* within a single health
system (e.g. splitting one hospital's data by department) rather than
genuine multi-institution federation.

This project tests three specific questions using a combined federation of
**MIMIC-IV** (one U.S. academic health system, 9 ICU care units) and
**eICU-CRD** (69 real, independent U.S. hospitals) — 78 clients total,
176,199 ICU stays:

1. **Does FL's benefit over local-only training scale with model capacity**
   under real, not simulated, institutional heterogeneity?
2. **Does the optimal aggregation algorithm depend on heterogeneity
   severity** — i.e., is there one universally "best" FL algorithm, or does
   the answer change with how different the clients actually are?
3. **Does a model trained via FL on one set of hospitals transport to a
   health system it has never seen**, and if it appears not to, is that a
   genuine failure to learn generalizable clinical patterns or a
   correctable artifact of how inputs are scaled?

## Key findings

| Question | Finding |
|---|---|
| Capacity vs. FL benefit | Federation's advantage over local-only training increases with model capacity (Spearman ρ = 0.714, p = 0.0001); federation won 25/25 controlled runs |
| Algorithm choice under real heterogeneity | FedProx (μ=0.01) significantly outperforms plain FedAvg (p = 0.002, 10 seeds, Bonferroni-corrected) — the *opposite* ranking from what is typically reported on simulated single-system heterogeneity |
| Cross-database generalization | An eICU-only-trained model showed an apparent 0.258 AUROC drop on unseen MIMIC-IV (0.828 → 0.570, unstable across seeds); re-scaling MIMIC-IV inputs using its own statistics — no retraining, same weights — recovered performance to 0.831, statistically indistinguishable from in-distribution |

## Headline model performance

Final model: hybrid GRU + MLP, federated with **FedProx (μ=0.01)** across
78 clients — selected per the algorithm-comparison finding above, not
simply the best single run (see note below).

| | AUROC | AUPRC | Brier |
|---|---|---|---|
| **Federated (FedProx, μ=0.01)** | 0.847 (0.838–0.855) | 0.441 (0.420–0.461) | 0.069 |
| Centralized (pooled, non-privacy-preserving upper bound) | 0.870 (0.862–0.878) | 0.487 (0.466–0.508) | 0.066 |

Prevalence 9.8% (2,189 / 22,436 held-out stays); AUPRC represents a
4.5-fold lift over the base rate.

**Single-run comparison across all three algorithms** (for context — the
statistically supported comparison is the 10-seed result above, not this
one-off run):

| Algorithm | AUROC | AUPRC |
|---|---|---|
| FedAvg | 0.847 (0.838–0.855) | 0.442 (0.422–0.463) |
| FedProx (μ=0.01) | 0.847 (0.838–0.855) | 0.441 (0.420–0.461) |
| FedAdam | 0.843 (0.834–0.851) | 0.431 (0.411–0.452) |

On a single run, FedAvg and FedProx are statistically indistinguishable
(overlapping CIs) and FedAdam is clearly worst — consistent with, not
contradicting, the 10-seed finding: FedProx's advantage (+0.0026 AUPRC) is
real but small, and only becomes statistically detectable when averaged
across multiple seeds, which is exactly why that controlled experiment
exists rather than relying on any single run.

## Repository structure

**Core pipeline**
```
config.py                  All paths and hyperparameters — edit before running
preprocess.py               MIMIC-IV extraction (DuckDB -> cohort + tensors)
preprocess_eicu.py          eICU-CRD extraction, same output schema
combine_datasets.py         Merges both into one federated dataset
partition.py                 Client partitioning + heterogeneity report
check_partition.py          Partition integrity verification
models.py                   Hybrid GRU + MLP architecture
fed_train.py                 Federated training (FedAvg / FedProx / FedAdam)
metrics.py                  Bootstrap CIs, calibration, decision curves
```

**Experiments**
```
tune.py                     Validation-only hyperparameter search
analyze_tuning.py           Search-result analysis
run_experiments.py          E1 capacity / E2 algorithms / E3 leakage / E4 ablation
external_validation.py      Cross-database generalization test (eICU -> MIMIC)
compare_control.py          IID-control comparison (isolates heterogeneity)
make_figures.py             Publication figure/table generation
```

**Utilities** (situational — see comments in each file; not part of the
main run sequence for a clean data download)
```
validate_dataset.py         Pre-flight check of MIMIC-IV source files
diagnose_csv.py             Detects truncated/corrupted MIMIC-IV downloads
diagnose_eicu.py            Detects truncated/corrupted eICU-CRD downloads
recover_truncated.py        Recovery procedure for a truncated chartevents download
verify_subsample.py         Statistical check that a recovered cohort is representative
check_files.py              Verifies all modules are internally consistent
```

**Demonstration**
```
demo.py                     Aggregate-only demo — no patient data displayed
demo_app.py                 Case-review app — displays real patient trajectories;
                             for PhysioNet-credentialed viewers only
```

**Robustness diagnostics** (exploratory checks, not part of the reported
experiment suite — see "Model robustness" below)
```
trend_check.py               Tests whether the model responds to a declining vs.
                             steady vital-sign trajectory
```

## Federated algorithms implemented

| Algorithm | Mechanism | Result on this federation |
|---|---|---|
| **FedAvg** | Sample-size-weighted parameter averaging (McMahan et al., 2017) | Strong baseline |
| **FedProx** | Adds a proximal penalty (μ/2)·‖w − w_t‖² to each client's local loss, discouraging drift from the global model (Li et al., 2020) | **Best**, μ=0.01, significant at 10 seeds |
| **FedAdam** | Server-side adaptive optimization in place of simple averaging (Reddi et al., 2021) | Consistently worst |

Set via `config.py`: `FL_ALGO = "fedavg" | "fedprox" | "fedadam"`, with
`PROX_MU` or `SERVER_LR` set accordingly.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Edit `config.py`: set `MIMIC_ROOT` and `EICU_ROOT` to your own local,
credentialed copies of the source data.

## Reproducing the results

```bash
# 1. Verify source data integrity
python validate_dataset.py
python diagnose_eicu.py

# 2. Preprocess and combine
python preprocess.py
python preprocess_eicu.py
python combine_datasets.py

# 3. Point config.py at the combined dataset, then partition
python partition.py
python check_partition.py

# 4. (Optional) hyperparameter search — validation only, test set untouched
python tune.py --jobs=4
python analyze_tuning.py

# 5. Run the four experiments + external validation
python run_experiments.py --jobs=4
python external_validation.py

# 6. Train the final headline model and generate figures
#    (set FL_ALGO="fedprox", PROX_MU=0.01 in config.py first)
python fed_train.py
python make_figures.py artifacts_combined
```

Multi-seed experiments use fixed seeds `[42, 101, 2026, 7, 1234]` (the
algorithm comparison uses 10 seeds: add `[11, 77, 314, 2718, 1618]`).
Hyperparameters are selected strictly on the validation split; the test
split is read exactly once for the final reported model.

## Methodological safeguards

- **Patient-level grouped splits**, enforced by a runtime assertion — no
  patient's stays appear on both sides of any train/val/test split
- **No discharge-time information leakage**: current-admission diagnosis
  codes (assigned only at hospital discharge) are excluded from all
  reported models; their inflationary effect is separately quantified
  (Experiment 3)
- **Per-client normalization**, computed independently from each client's
  own training split only — a federated-correctness requirement, since a
  global statistic would require pooling raw values centrally
- **Validation-only hyperparameter and threshold selection** — the test
  set is read exactly once, for final reporting
- **2,000-replicate stratified bootstrap** for all confidence intervals
- **Joint discrimination and calibration reporting** (AUROC, AUPRC, Brier
  score, calibration slope, decision-curve analysis) — most comparable
  studies in this literature report discrimination only

## Model robustness checks

Beyond the reported experiments, the trained model was probed directly for
two known failure modes in clinical deep learning, and the findings are
disclosed rather than omitted:

- **Physiologically inconsistent / out-of-distribution inputs** can
  produce non-monotonic risk estimates — a general property of neural
  networks evaluated outside their training distribution, not specific to
  this implementation.
- **Sensitivity to acute deterioration trends** was tested directly
  (`trend_check.py`) by comparing model response to a rapidly declining
  vs. a steadily abnormal vital-sign trajectory. Results were mixed and
  dataset-dependent across development iterations — this is stated as an
  unconfirmed capability, not a validated one. See `MODEL_CARD.md` for
  full discussion and recommended usage constraints.

## Data availability

This repository does not distribute MIMIC-IV or eICU-CRD patient-level
data. No patient identifiers, per-patient predictions, or derived tensors
are included.

## Citation

```
[citation once published]
```

Source databases:
```
Johnson AEW, Bulgarelli L, Pollard T, et al. MIMIC-IV.
    Scientific Data. 2023.
Pollard TJ, Johnson AEW, Raffa JD, Celi LA, Mark RG, Badawi O.
    The eICU Collaborative Research Database. Scientific Data. 2018.
```

## License

The code in this repository is released under the MIT License.

The datasets used in this project are not included in this repository and remain subject to the terms and conditions of their respective PhysioNet data use agreements. The MIT License applies only to the original code contained in this repository.
