"""
The alpha-sweep ablation: FedAvg vs FedProx across heterogeneity levels.

This is the experiment that addresses the contradiction in the literature --
FedProx was best on MIMIC-IV (care-unit split) but worst on eICU (hospital
split), and nobody has characterised where the crossover is. Sweeping the
Dirichlet alpha lets you plot performance vs. heterogeneity and show *when*
the proximal term starts paying for itself.

Run:  python run_ablation.py
Out:  artifacts/ablation_alpha_sweep.csv
"""
import json

import numpy as np
import pandas as pd
import torch

import config as C
import partition as P
import fed_train as F

ALPHAS = [0.1, 0.3, 1.0, 10.0]
ALGOS = ["fedavg", "fedprox"]


def main():
    cohort = pd.read_parquet(C.COHORT_PQ).reset_index(drop=True)
    labels = cohort.label.to_numpy()
    rows = []

    for alpha in ALPHAS:
        clients = P.check_clients(
            P.partition_dirichlet(cohort, C.DIRICHLET_N_CLIENTS, alpha, C.SEED),
            f"dirichlet(alpha={alpha})")
        rep = P.heterogeneity_report(cohort, clients)
        js = rep["JS_div_vs_global"].mean()

        groups = cohort.subject_id.to_numpy()
        part = {}
        for i, (name, idx) in enumerate(clients.items()):
            tr, va, te = P.split_client(idx, labels, C.SEED + i, groups=groups)
            part[name] = {"train": tr.tolist(), "val": va.tolist(), "test": te.tolist()}

        path = C.OUT_DIR / "partition_dirichlet.json"
        P.assert_no_patient_leakage(part, groups)
        with open(path, "w") as f:
            json.dump(part, f)

        for algo in ALGOS:
            C.FL_ALGO = algo
            C.PARTITION_SCHEME = "dirichlet"
            F.C = C

            torch.manual_seed(C.SEED)
            np.random.seed(C.SEED)
            X, mask, static, y, part_loaded, nf, ns = F.load_data()
            g, _ = F.run_federated(X, mask, static, y, part_loaded, nf, ns)

            rows.append({
                "alpha": alpha, "algo": algo, "mean_JS": js,
                "auroc": g["auroc"], "auprc": g["auprc"],
                "f1": g["f1"], "brier": g["brier"],
            })
            print(f"[alpha={alpha} {algo}] AUROC {g['auroc']:.4f} AUPRC {g['auprc']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(C.OUT_DIR / "ablation_alpha_sweep.csv", index=False)
    print("\n" + df.to_string(index=False))

    piv = df.pivot(index="alpha", columns="algo", values="auprc")
    piv["fedprox_gain"] = piv["fedprox"] - piv["fedavg"]
    print("\nFedProx advantage vs. heterogeneity (this is your headline figure):")
    print(piv.to_string())


if __name__ == "__main__":
    main()
