#!/usr/bin/env python3
import os
import sys
import json
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTG_DIR = os.path.join(PROJECT_ROOT, "actg175")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algs.aipw_continuous import estimate_aipw_continuous_df
from algs.ipw import estimate_ipw_df
from algs.outcome_regression_continuous import estimate_outcome_regression_continuous_df
from algs.tmle_continuous import estimate_tmle_continuous_df

SYN_SOURCES = {
    "llm": os.path.join(ACTG_DIR, "llm_data", "syn_hybrid.csv"),
    "ctgan": os.path.join(ACTG_DIR, "ctgan_data", "syn_hybrid.csv"),
}

COVARIATES = [
    "age", "wtkg", "hemo", "homo", "drugs", "karnof",
    "oprior", "z30", "zprior", "preanti", "race", "gender",
    "str2", "strat", "symptom", "cd40", "cd80"
]
OUTCOME_COL = "cd420"
TREATMENT_COL = "A"

ESTIMATORS = {
    "ipw": estimate_ipw_df,
    "tmle_continuous": estimate_tmle_continuous_df,
    "aipw_continuous": estimate_aipw_continuous_df,
    "outcome_regression_continuous": estimate_outcome_regression_continuous_df,
}

needed = COVARIATES + [OUTCOME_COL, TREATMENT_COL]


def evaluate_estimator(df, estimator_name):
    fn = ESTIMATORS[estimator_name]
    return float(
        fn(
            data=df.copy(),
            covariates=COVARIATES,
            outcome_col=OUTCOME_COL,
            treatment_col=TREATMENT_COL,
            random_state=42,
        )
    )


def main():
    results = {}

    for source_name, source_path in SYN_SOURCES.items():
        print(f"\n=== {source_name} ===")
        df = pd.read_csv(source_path)

        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"{source_name} missing columns: {missing}")

        df = df[needed].dropna().reset_index(drop=True)

        print(f"full usable pool size: {len(df)}")

        estimates = {}
        for est_name in ESTIMATORS:
            print(f"Running {est_name} on full {source_name} pool...")
            estimates[est_name] = evaluate_estimator(df, est_name)

        tmle_ref = estimates["tmle_continuous"]

        rows = []
        for est_name, est in estimates.items():
            rows.append({
                "source": source_name,
                "estimator": est_name,
                "full_pool_estimate": est,
                "difference_from_full_pool_tmle": est - tmle_ref,
                "absolute_difference_from_full_pool_tmle": abs(est - tmle_ref),
            })

        out_df = pd.DataFrame(rows)
        print(out_df.to_string(index=False))

        results[source_name] = {
            "source_file": source_path,
            "n_full_pool": len(df),
            "estimates": estimates,
            "tmle_reference": tmle_ref,
            "differences_from_tmle": {
                est_name: est - tmle_ref for est_name, est in estimates.items()
            },
            "absolute_differences_from_tmle": {
                est_name: abs(est - tmle_ref) for est_name, est in estimates.items()
            },
        }

    out_path = os.path.join(ACTG_DIR, "results", "full_sample_synth.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()