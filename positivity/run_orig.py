#!/usr/bin/env python3
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

SAMPLE_SIZES = [100, 200, 500]
SEEDS = [1, 2, 3, 4, 5]

DATA_BASE_DIR = os.path.join(PROJECT_ROOT, "positivity", "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "positivity", "results")
TRUTH_PATH = os.path.join(PROJECT_ROOT, "truth.json")


def load_truth():
    with open(TRUTH_PATH, "r") as f:
        truth = json.load(f)
    return float(truth["ate_true"]), truth


def propensity_truncation_level(n):
    if n <= 1:
        return 0.25
    delta_n = 5.0 / (np.sqrt(n) * np.log(n))
    delta_n = max(1e-6, float(delta_n))
    return min(delta_n, 0.25)


def overlap_summary(df):
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(random_state=42)
    rf.fit(df[COVARIATES], df[TREATMENT_COL])

    delta_n = propensity_truncation_level(len(df))
    ps_hat_raw = rf.predict_proba(df[COVARIATES])[:, 1]
    ps_hat = np.clip(ps_hat_raw, delta_n, 1 - delta_n)

    return {
        "n": int(len(df)),
        "treated": int((df[TREATMENT_COL] == 1).sum()),
        "control": int((df[TREATMENT_COL] == 0).sum()),
        "g_trunc_level": float(delta_n),
        "min_ps_hat_raw": float(ps_hat_raw.min()),
        "max_ps_hat_raw": float(ps_hat_raw.max()),
        "min_ps_hat_trunc": float(ps_hat.min()),
        "max_ps_hat_trunc": float(ps_hat.max()),
        "count_ps_hat_raw_lt_0.001": int((ps_hat_raw < 0.001).sum()),
        "count_ps_hat_raw_gt_0.999": int((ps_hat_raw > 0.999).sum()),
        "count_ps_hat_raw_lt_0.01": int((ps_hat_raw < 0.01).sum()),
        "count_ps_hat_raw_gt_0.99": int((ps_hat_raw > 0.99).sum()),
    }


def evaluate_dataset(path, estimator_fn, estimator_name, ate_true):
    df = pd.read_csv(path)
    delta_n = propensity_truncation_level(len(df))

    kwargs = {
        "covariates": COVARIATES,
        "outcome_col": OUTCOME_COL,
        "treatment_col": TREATMENT_COL,
        "random_state": 42,
    }

    if estimator_name in {"aipw", "ipw", "tmle"}:
        kwargs["clip_min"] = delta_n

    est = estimator_fn(df.copy(), **kwargs)
    bias = float(est - ate_true)

    return {
        "file": path,
        "estimator": estimator_name,
        "estimate": float(est),
        "ate_true": float(ate_true),
        "bias": bias,
        "abs_bias": float(abs(bias)),
        "sq_error": float(bias ** 2),
        "g_trunc_level": float(delta_n),
        "n": int(len(df)),
    }


def summarize_estimator_results(estimator_records, ate_true):
    estimates = np.array([r["estimate"] for r in estimator_records], dtype=float)
    errors = estimates - float(ate_true)
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2

    n = len(estimates)

    mean_est = float(np.mean(estimates))
    std_est = float(np.std(estimates, ddof=1)) if n > 1 else 0.0
    se_est = float(std_est / np.sqrt(n)) if n > 1 else 0.0

    bias = float(mean_est - ate_true)
    bias_std = float(np.std(errors, ddof=1)) if n > 1 else 0.0
    bias_se = float(bias_std / np.sqrt(n)) if n > 1 else 0.0

    mae = float(np.mean(abs_errors))
    mae_std = float(np.std(abs_errors, ddof=1)) if n > 1 else 0.0
    mae_se = float(mae_std / np.sqrt(n)) if n > 1 else 0.0

    mse = float(np.mean(sq_errors))
    mse_std = float(np.std(sq_errors, ddof=1)) if n > 1 else 0.0
    mse_se = float(mse_std / np.sqrt(n)) if n > 1 else 0.0

    return {
        "n_reps": int(n),
        "ate_true": float(ate_true),
        "estimates": estimates.tolist(),
        "errors": errors.tolist(),
        "abs_errors": abs_errors.tolist(),
        "sq_errors": sq_errors.tolist(),

        "mean": mean_est,
        "std": std_est,
        "se": se_est,
        "ci95_low": float(mean_est - 1.96 * se_est),
        "ci95_high": float(mean_est + 1.96 * se_est),

        "bias": bias,
        "bias_std": bias_std,
        "bias_se": bias_se,
        "abs_bias": float(abs(bias)),

        "mae": mae,
        "mae_std": mae_std,
        "mae_se": mae_se,

        "mse": mse,
        "mse_std": mse_std,
        "mse_se": mse_se,
        "mse_ci95_low": float(max(0.0, mse - 1.96 * mse_se)),
        "mse_ci95_high": float(mse + 1.96 * mse_se),
        "rmse": float(np.sqrt(mse)),
    }


def make_datasets_for_n(n):
    data_dir = os.path.join(DATA_BASE_DIR, f"n{n}")
    return {
        f"data_{seed}": os.path.join(data_dir, f"data_{seed}.csv")
        for seed in SEEDS
    }


def main():
    ate_true, truth = load_truth()

    estimators = {
        "aipw": estimate_aipw_df,
        "ipw": estimate_ipw_df,
        "outcome_regression": estimate_outcome_regression_df,
        "tmle": estimate_tmle_df,
    }

    results = {
        "truth": truth,
        "sample_sizes": SAMPLE_SIZES,
        "seeds": SEEDS,
        "g_bounds_rule": "delta_n = 5 / (sqrt(n) * log(n)); g in [delta_n, 1-delta_n]",
        "sample_size_results": {},
    }

    for n in SAMPLE_SIZES:
        n_key = f"n{n}"
        datasets = make_datasets_for_n(n)

        results["sample_size_results"][n_key] = {
            "datasets": {},
            "summary_by_estimator": {},
        }

        estimator_records = {name: [] for name in estimators}

        for dataset_name, path in datasets.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing dataset: {path}")

            df = pd.read_csv(path)

            results["sample_size_results"][n_key]["datasets"][dataset_name] = {
                "file": path,
                "overlap": overlap_summary(df),
                "estimators": {},
            }

            for est_name, est_fn in estimators.items():
                print(f"Running original | n={n} | {est_name} on {dataset_name}")

                record = evaluate_dataset(
                    path=path,
                    estimator_fn=est_fn,
                    estimator_name=est_name,
                    ate_true=ate_true,
                )

                results["sample_size_results"][n_key]["datasets"][dataset_name]["estimators"][est_name] = record
                estimator_records[est_name].append(record)

        for est_name, records in estimator_records.items():
            results["sample_size_results"][n_key]["summary_by_estimator"][est_name] = (
                summarize_estimator_results(records, ate_true)
            )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "orig_estimators_by_n.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()