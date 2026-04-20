import os
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algs.aipw import estimate_aipw_df
from algs.ipw import estimate_ipw_df
from algs.outcome_regression import estimate_outcome_regression_df
from algs.tmle import estimate_tmle_df

COVARIATES = ["W1", "W2", "W3", "W4", "W5", "W6"]
OUTCOME_COL = "Y"
TREATMENT_COL = "A"
SEEDS = [1, 2, 3, 4, 5]
SUBSAMPLE_N = 1000
TRUTH_PATH = os.path.join(PROJECT_ROOT, "truth.json")

DATASETS = {
    "llm_syn_clean": os.path.join(PROJECT_ROOT, "llm_data", "syn_clean.csv"),
    "llm_syn_hybrid": os.path.join(PROJECT_ROOT, "llm_data", "syn_hybrid.csv"),
    "gan_syn_clean": os.path.join(PROJECT_ROOT, "gan_data", "syn_clean.csv"),
    "gan_syn_hybrid": os.path.join(PROJECT_ROOT, "gan_data", "syn_hybrid.csv"),
}

def load_truth():
    with open(TRUTH_PATH, "r") as f:
        truth = json.load(f)
    return float(truth["ate_true"]), truth

def evaluate_dataset(path, estimator_fn, estimator_name, ate_true):
    df = pd.read_csv(path)
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
    mean_est = float(np.mean(estimates))
    bias = mean_est - ate_true
    sq_errors = (estimates - ate_true) ** 2

    return {
        "file": path,
        "estimator": estimator_name,
        "n_full": int(n),
        "subsample_n": int(subsample_n),
        "n_reps": len(SEEDS),
        "seeds": SEEDS,
        "ate_true": float(ate_true),
        "estimates": estimates.tolist(),
        "mean": mean_est,
        "bias": float(bias),
        "abs_bias": float(abs(bias)),
        "std": float(np.std(estimates, ddof=1)),
        "var": float(np.var(estimates, ddof=1)),
        "mse": float(np.mean(sq_errors)),
        "rmse": float(np.sqrt(np.mean(sq_errors))),
        "min": float(np.min(estimates)),
        "max": float(np.max(estimates)),
    }

def main():
    ate_true, truth = load_truth()

    results = {
        "truth": truth,
        "subsample_n": SUBSAMPLE_N,
        "seeds": SEEDS,
        "datasets": {},
    }

    estimators = {
        "aipw": estimate_aipw_df,
        "ipw": estimate_ipw_df,
        "outcome_regression": estimate_outcome_regression_df,
        "tmle": estimate_tmle_df,
    }

    for dataset_name, path in DATASETS.items():
        results["datasets"][dataset_name] = {}
        for est_name, est_fn in estimators.items():
            print(f"Running {est_name} on {dataset_name}")
            results["datasets"][dataset_name][est_name] = evaluate_dataset(
                path=path,
                estimator_fn=est_fn,
                estimator_name=est_name,
                ate_true=ate_true,
            )

    results_dir = os.path.join(PROJECT_ROOT, "privacy", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "privacy_estimators.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()