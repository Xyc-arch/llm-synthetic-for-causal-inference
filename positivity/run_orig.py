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

DATA_DIR = os.path.join(PROJECT_ROOT, "positivity", "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "positivity", "results")
TRUTH_PATH = os.path.join(PROJECT_ROOT, "truth.json")

DATASETS = {
    f"data_{i}": os.path.join(DATA_DIR, f"data_{i}.csv")
    for i in [1, 2, 3, 4, 5]
}


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
        "min_ps_hat": float(ps_hat.min()),
        "max_ps_hat": float(ps_hat.max()),
        "count_ps_hat_lt_0.001": int((ps_hat < 0.001).sum()),
        "count_ps_hat_gt_0.999": int((ps_hat > 0.999).sum()),
        "count_ps_hat_lt_0.01": int((ps_hat < 0.01).sum()),
        "count_ps_hat_gt_0.99": int((ps_hat > 0.99).sum()),
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

    # Only these estimators accept and use clip_min
    if estimator_name in {"aipw", "ipw", "tmle"}:
        kwargs["clip_min"] = delta_n

    est = estimator_fn(df, **kwargs)
    bias = float(est - ate_true)

    return {
        "file": path,
        "estimator": estimator_name,
        "estimate": float(est),
        "ate_true": float(ate_true),
        "bias": bias,
        "abs_bias": float(abs(bias)),
        "g_trunc_level": float(delta_n),
    }


def main():
    ate_true, truth = load_truth()

    results = {
        "truth": truth,
        "g_bounds_rule": "delta_n = 5 / (sqrt(n) * log(n)); g in [delta_n, 1-delta_n]",
        "datasets": {},
    }

    estimators = {
        "aipw": estimate_aipw_df,
        "ipw": estimate_ipw_df,
        "outcome_regression": estimate_outcome_regression_df,
        "tmle": estimate_tmle_df,
    }

    for dataset_name, path in DATASETS.items():
        df = pd.read_csv(path)

        results["datasets"][dataset_name] = {
            "overlap": overlap_summary(df),
            "estimators": {},
        }

        for est_name, est_fn in estimators.items():
            print(f"Running {est_name} on {dataset_name}")
            results["datasets"][dataset_name]["estimators"][est_name] = evaluate_dataset(
                path=path,
                estimator_fn=est_fn,
                estimator_name=est_name,
                ate_true=ate_true,
            )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "orig_estimators.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()