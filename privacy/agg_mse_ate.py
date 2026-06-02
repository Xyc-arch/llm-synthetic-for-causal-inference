import os
import json
from collections import OrderedDict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(PROJECT_ROOT, "privacy", "results", "privacy_estimators.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "privacy", "results", "privacy_estimators_compact.json")


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def summarize_from_estimates(estimates, ate_true):
    estimates = np.array(estimates, dtype=float)
    errors = estimates - ate_true
    sq_errors = errors ** 2

    n = len(estimates)

    mean = float(np.mean(estimates))
    std = float(np.std(estimates, ddof=1))
    se = float(std / np.sqrt(n))

    bias = float(mean - ate_true)
    abs_bias = float(abs(bias))

    mse = float(np.mean(sq_errors))
    mse_std = float(np.std(sq_errors, ddof=1))
    mse_se = float(mse_std / np.sqrt(n))
    rmse = float(np.sqrt(mse))

    return OrderedDict([
        ("mean", round(mean, 6)),
        ("std", round(std, 6)),
        ("se", round(se, 6)),
        ("ci95_low", round(mean - 1.96 * se, 6)),
        ("ci95_high", round(mean + 1.96 * se, 6)),

        ("bias", round(bias, 6)),
        ("abs_bias", round(abs_bias, 6)),

        ("mse", round(mse, 6)),
        ("mse_std", round(mse_std, 6)),
        ("mse_se", round(mse_se, 6)),
        ("mse_ci95_low", round(mse - 1.96 * mse_se, 6)),
        ("mse_ci95_high", round(mse + 1.96 * mse_se, 6)),

        ("rmse", round(rmse, 6)),
    ])


def build_compact(results):
    compact = OrderedDict()
    ate_true = float(results["truth"]["ate_true"])

    compact["ate_true"] = ate_true
    compact["subsample_n"] = results["subsample_n"]
    compact["seeds"] = results["seeds"]
    compact["datasets"] = OrderedDict()

    for dataset_name, dataset_results in results["datasets"].items():
        compact["datasets"][dataset_name] = OrderedDict()
        for est_name, metrics in dataset_results.items():
            compact["datasets"][dataset_name][est_name] = summarize_from_estimates(
                estimates=metrics["estimates"],
                ate_true=ate_true,
            )

    return compact


def main():
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("INPUT_PATH   =", INPUT_PATH)
    print("OUTPUT_PATH  =", OUTPUT_PATH)

    results = load_results(INPUT_PATH)
    compact = build_compact(results)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(compact, f, indent=2)

    print(f"Saved compact aggregate JSON to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()