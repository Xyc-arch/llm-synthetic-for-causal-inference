#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_generate import generate_dataset

SEEDS = [1, 2, 3, 4, 5]
SAMPLE_SIZES = [100, 200, 500]

BASE_OUT_DIR = os.path.join(PROJECT_ROOT, "positivity", "data")
os.makedirs(BASE_OUT_DIR, exist_ok=True)

all_summary = {}

for n in SAMPLE_SIZES:
    out_dir = os.path.join(BASE_OUT_DIR, f"n{n}")
    os.makedirs(out_dir, exist_ok=True)

    summary = {}

    for seed in SEEDS:
        data, ate_true, y1_true, y0_true = generate_dataset(
            n=n,
            seed=seed,
            rct=False,
            truth=True,
        )

        out_csv = os.path.join(out_dir, f"data_{seed}.csv")
        data.drop(columns=["pA", "pY"]).to_csv(out_csv, index=False)

        summary[f"data_{seed}"] = {
            "seed": seed,
            "n": n,
            "treated": int((data["A"] == 1).sum()),
            "control": int((data["A"] == 0).sum()),
            "outcome_1": int((data["Y"] == 1).sum()),
            "outcome_0": int((data["Y"] == 0).sum()),
            "true_ate_mc": float(ate_true),
            "true_y1_mc": float(y1_true),
            "true_y0_mc": float(y0_true),
            "min_true_pA": float(data["pA"].min()),
            "max_true_pA": float(data["pA"].max()),
            "pA_q01": float(np.quantile(data["pA"], 0.01)),
            "pA_q05": float(np.quantile(data["pA"], 0.05)),
            "pA_median": float(np.median(data["pA"])),
            "pA_q95": float(np.quantile(data["pA"], 0.95)),
            "pA_q99": float(np.quantile(data["pA"], 0.99)),
            "count_true_pA_lt_0.001": int((data["pA"] < 0.001).sum()),
            "count_true_pA_gt_0.999": int((data["pA"] > 0.999).sum()),
            "count_true_pA_lt_0.01": int((data["pA"] < 0.01).sum()),
            "count_true_pA_gt_0.99": int((data["pA"] > 0.99).sum()),
        }

    summary_path = os.path.join(out_dir, "dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    all_summary[f"n{n}"] = summary

    print(f"Saved n={n} datasets to {out_dir}")
    print(f"Saved summary to {summary_path}")

all_summary_path = os.path.join(BASE_OUT_DIR, "all_dataset_summary.json")
with open(all_summary_path, "w") as f:
    json.dump(all_summary, f, indent=4)

print(f"Saved all summaries to {all_summary_path}")