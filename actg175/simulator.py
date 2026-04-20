#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = "/home/ubuntu/syn_causal"
ACTG_DIR = os.path.join(PROJECT_ROOT, "actg175")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algs.aipw_continuous import estimate_aipw_continuous_df
from algs.ipw import estimate_ipw_df
from algs.outcome_regression_continuous import (
    estimate_outcome_regression_continuous_df,
)
from algs.tmle_continuous import estimate_tmle_continuous_df

# =========================
# Paths
# =========================

SYN_SOURCES = {
    "llm": os.path.join(ACTG_DIR, "llm_data", "syn_hybrid.csv"),
    "ctgan": os.path.join(ACTG_DIR, "ctgan_data", "syn_hybrid.csv"),
}

OUT_DIR = os.path.join(ACTG_DIR, "results")
OUT_JSON = os.path.join(OUT_DIR, "simulation_engine_synth_only.json")

# =========================
# Experiment setup
# =========================

COVARIATES = [
    "age", "wtkg", "hemo", "homo", "drugs", "karnof",
    "oprior", "z30", "zprior", "preanti", "race", "gender",
    "str2", "strat", "symptom", "cd40", "cd80"
]
OUTCOME_COL = "cd420"
TREATMENT_COL = "A"

SAMPLE_SIZES = list(range(100, 1001, 100))
N_SYN_REPS = 20

ESTIMATORS = {
    "ipw": estimate_ipw_df,
    "tmle_continuous": estimate_tmle_continuous_df,
    "aipw_continuous": estimate_aipw_continuous_df,
    "outcome_regression_continuous": estimate_outcome_regression_continuous_df,
}

# Common synthetic reference truth for all estimators within a source
SYNTHETIC_REFERENCE_ESTIMATOR = "tmle_continuous"


# =========================
# Utilities
# =========================

def evaluate_estimator(df, estimator_name):
    fn = ESTIMATORS[estimator_name]
    kwargs = {
        "data": df.copy(),
        "covariates": COVARIATES,
        "outcome_col": OUTCOME_COL,
        "treatment_col": TREATMENT_COL,
        "random_state": 42,
    }
    return float(fn(**kwargs))


def sample_without_replacement(df, n, seed):
    if n > len(df):
        raise ValueError(f"Requested sample size {n} exceeds pool size {len(df)}.")
    rng = np.random.default_rng(seed)
    idx = rng.choice(df.index.to_numpy(), size=n, replace=False)
    return df.loc[idx].reset_index(drop=True)


def summarize_against_truth(estimates, truth_value):
    arr = np.asarray(estimates, dtype=float)
    errors = arr - truth_value
    return {
        "n_reps": int(len(arr)),
        "estimates": arr.tolist(),
        "mean_estimate": float(np.mean(arr)),
        "bias": float(np.mean(arr) - truth_value),
        "abs_bias": float(abs(np.mean(arr) - truth_value)),
        "var": float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "mse": float(np.mean(errors ** 2)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "truth": float(truth_value),
    }


# =========================
# Main
# =========================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    results = {
        "config": {
            "project_root": PROJECT_ROOT,
            "actg_dir": ACTG_DIR,
            "synthetic_sources": SYN_SOURCES,
            "out_dir": OUT_DIR,
            "sample_sizes": SAMPLE_SIZES,
            "n_syn_reps_per_source": N_SYN_REPS,
            "estimators": list(ESTIMATORS.keys()),
            "covariates": COVARIATES,
            "outcome_col": OUTCOME_COL,
            "treatment_col": TREATMENT_COL,
            "synthetic_reference_estimator": SYNTHETIC_REFERENCE_ESTIMATOR,
        },
        "synthetic_engine": {},
    }

    for source_name, source_path in SYN_SOURCES.items():
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Missing synthetic hybrid file: {source_path}")

        print(f"Evaluating synthetic engine for source: {source_name}")
        syn_pool = pd.read_csv(source_path)

        needed = COVARIATES + [OUTCOME_COL, TREATMENT_COL]
        missing = [c for c in needed if c not in syn_pool.columns]
        if missing:
            raise ValueError(f"{source_name} synthetic pool missing required columns: {missing}")

        syn_pool = syn_pool[needed].dropna().reset_index(drop=True)

        usable_sample_sizes = [n for n in SAMPLE_SIZES if n <= len(syn_pool)]
        if len(usable_sample_sizes) == 0:
            raise ValueError(f"No sample sizes are usable for {source_name}; pool size={len(syn_pool)}")

        syn_reference_truth = evaluate_estimator(syn_pool, SYNTHETIC_REFERENCE_ESTIMATOR)

        results["synthetic_engine"][source_name] = {
            "source_file": source_path,
            "n_full_pool": int(len(syn_pool)),
            "reference_truth": {
                "estimator_used": SYNTHETIC_REFERENCE_ESTIMATOR,
                "estimate_on_full_hybrid_pool": float(syn_reference_truth),
            },
            "by_sample_size": {},
        }

        for n in usable_sample_sizes:
            print(f"  n={n}")
            results["synthetic_engine"][source_name]["by_sample_size"][str(n)] = {}

            syn_rep_collect = {e: [] for e in ESTIMATORS}

            for rep in range(1, N_SYN_REPS + 1):
                rep_df = sample_without_replacement(
                    syn_pool,
                    n=n,
                    seed=100000 + 1000 * (1 if source_name == "llm" else 2) + 100 * n + rep,
                )

                for est_name in ESTIMATORS:
                    rep_est = evaluate_estimator(rep_df, est_name)
                    syn_rep_collect[est_name].append(rep_est)

            for est_name in ESTIMATORS:
                summary_internal = summarize_against_truth(
                    estimates=syn_rep_collect[est_name],
                    truth_value=syn_reference_truth,
                )

                results["synthetic_engine"][source_name]["by_sample_size"][str(n)][est_name] = {
                    "against_synthetic_reference_truth": summary_internal,
                }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved results to {OUT_JSON}")


if __name__ == "__main__":
    main()