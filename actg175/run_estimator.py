#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTG_DIR = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algs.aipw import estimate_aipw_df
from algs.ipw import estimate_ipw_df
from algs.outcome_regression import estimate_outcome_regression_df
from algs.tmle_continuous import estimate_tmle_continuous_df

COVARIATES = [
    "age", "wtkg", "hemo", "homo", "drugs", "karnof",
    "oprior", "z30", "zprior", "preanti", "race", "gender",
    "str2", "strat", "symptom", "cd40", "cd80"
]
CONT_VARS = ["age", "wtkg", "karnof", "preanti", "cd40", "cd80"]

OUTCOME_COL = "cd420"
TREATMENT_COL = "A"

SEEDS = [1, 2, 3, 4, 5]
SUBSAMPLE_N = 1000

# Use cleaned ACTG data, not raw actg175.csv
ORIGINAL_FILE = os.path.join(ACTG_DIR, "data", "actg175_clean.csv")

DATASETS = {
    "actg_original": ORIGINAL_FILE,
    "llm_syn_clean": os.path.join(ACTG_DIR, "llm_data", "syn_clean.csv"),
    "llm_syn_hybrid": os.path.join(ACTG_DIR, "llm_data", "syn_hybrid.csv"),
    "ctgan_syn_clean": os.path.join(ACTG_DIR, "ctgan_data", "syn_clean.csv"),
    "ctgan_syn_hybrid": os.path.join(ACTG_DIR, "ctgan_data", "syn_hybrid.csv"),
}


def load_original_actg():
    if not os.path.exists(ORIGINAL_FILE):
        raise FileNotFoundError(f"Missing original ACTG cleaned file: {ORIGINAL_FILE}")

    df = pd.read_csv(ORIGINAL_FILE)

    needed = COVARIATES + [TREATMENT_COL, OUTCOME_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in cleaned ACTG file: {missing}")

    df = df[needed].dropna().copy()

    for c in CONT_VARS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    discrete_cols = [c for c in COVARIATES if c not in CONT_VARS] + [TREATMENT_COL]
    for c in discrete_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").round().astype(int)

    df = df.dropna().reset_index(drop=True)
    return df


def sanitize_synthetic(df, ref_df):
    needed = COVARIATES + [TREATMENT_COL, OUTCOME_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Synthetic dataset missing required columns: {missing}")

    df = df[needed].copy()

    for c in CONT_VARS + [OUTCOME_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    discrete_cols = [c for c in COVARIATES if c not in CONT_VARS] + [TREATMENT_COL]
    for c in discrete_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    # snap discrete vars to observed support
    for c in discrete_cols:
        df[c] = df[c].round()
        observed = sorted(ref_df[c].dropna().unique().tolist())
        df[c] = df[c].apply(
            lambda x: np.nan if pd.isna(x) else min(observed, key=lambda z: abs(z - x))
        )

    # clip continuous vars to observed range
    for c in CONT_VARS:
        lo = float(ref_df[c].min())
        hi = float(ref_df[c].max())
        df[c] = df[c].clip(lo, hi)

    # clip outcome to observed range
    y_lo = float(ref_df[OUTCOME_COL].min())
    y_hi = float(ref_df[OUTCOME_COL].max())
    df[OUTCOME_COL] = df[OUTCOME_COL].clip(y_lo, y_hi)

    df = df.dropna().copy()

    for c in discrete_cols:
        df[c] = df[c].astype(int)

    return df.reset_index(drop=True)


def evaluate_dataset(path, estimator_fn, estimator_name, original_df):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset file: {path}")

    df = pd.read_csv(path)

    if path != ORIGINAL_FILE:
        df = sanitize_synthetic(df, original_df)
    else:
        df = original_df.copy()

    n = len(df)
    subsample_n = min(SUBSAMPLE_N, n)
    estimates = []

    for seed in SEEDS:
        sub = df.sample(n=subsample_n, random_state=seed).copy()
        est = estimator_fn(
            sub,
            covariates=COVARIATES,
            outcome_col=OUTCOME_COL,
            treatment_col=TREATMENT_COL,
            random_state=42,
        )
        estimates.append(float(est))

    estimates = np.array(estimates, dtype=float)

    return {
        "file": path,
        "estimator": estimator_name,
        "n_full": int(n),
        "subsample_n": int(subsample_n),
        "n_reps": len(SEEDS),
        "seeds": SEEDS,
        "estimates": estimates.tolist(),
        "mean": float(np.mean(estimates)),
        "std": float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0,
        "var": float(np.var(estimates, ddof=1)) if len(estimates) > 1 else 0.0,
        "min": float(np.min(estimates)),
        "max": float(np.max(estimates)),
    }


def main():
    original_df = load_original_actg()

    results = {
        "original_file": ORIGINAL_FILE,
        "subsample_n": SUBSAMPLE_N,
        "seeds": SEEDS,
        "datasets": {},
    }

    estimators = {
        "aipw": estimate_aipw_df,
        "ipw": estimate_ipw_df,
        "outcome_regression": estimate_outcome_regression_df,
        "tmle_continuous": estimate_tmle_continuous_df,
    }

    for dataset_name, path in DATASETS.items():
        results["datasets"][dataset_name] = {}
        for est_name, est_fn in estimators.items():
            print(f"Running {est_name} on {dataset_name}")
            results["datasets"][dataset_name][est_name] = evaluate_dataset(
                path=path,
                estimator_fn=est_fn,
                estimator_name=est_name,
                original_df=original_df,
            )

    results_dir = os.path.join(ACTG_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "actg_estimators.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()