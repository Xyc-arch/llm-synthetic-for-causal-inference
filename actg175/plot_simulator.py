#!/usr/bin/env python3
import os
import json
import matplotlib.pyplot as plt

PROJECT_ROOT = "/home/ubuntu/syn_causal"
ACTG_DIR = os.path.join(PROJECT_ROOT, "actg175")

INPUT_JSON = os.path.join(ACTG_DIR, "results", "simulation_engine_synth_only.json")
PLOT_DIR = os.path.join(ACTG_DIR, "plot")

METRICS = ["bias", "var", "mse", "rmse"]
METRIC_LABELS = {
    "bias": "Bias",
    "var": "Variance",
    "mse": "MSE",
    "rmse": "RMSE",
}

ESTIMATOR_ORDER = [
    "ipw",
    "tmle_continuous",
    "aipw_continuous",
    "outcome_regression_continuous",
]

ESTIMATOR_LABELS = {
    "ipw": "IPW",
    "tmle_continuous": "TMLE",
    "aipw_continuous": "AIPW",
    "outcome_regression_continuous": "OR",
}


def load_results(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing input JSON: {path}")
    with open(path, "r") as f:
        return json.load(f)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    results = load_results(INPUT_JSON)

    synthetic_engine = results["synthetic_engine"]

    for source_name, source_obj in synthetic_engine.items():
        by_sample_size = source_obj["by_sample_size"]

        sample_sizes = sorted(int(k) for k in by_sample_size.keys())
        first_n = str(sample_sizes[0])
        estimators_found = list(by_sample_size[first_n].keys())

        estimators = [e for e in ESTIMATOR_ORDER if e in estimators_found]

        for metric in METRICS:
            plt.figure(figsize=(8, 5))

            for est_name in estimators:
                y_vals = []
                for n in sample_sizes:
                    metric_val = by_sample_size[str(n)][est_name]["against_synthetic_reference_truth"][metric]
                    y_vals.append(metric_val)

                plt.plot(
                    sample_sizes,
                    y_vals,
                    marker="o",
                    label=ESTIMATOR_LABELS.get(est_name, est_name),
                )

            plt.xlabel("Sample size")
            plt.ylabel(METRIC_LABELS[metric])
            plt.title(f"{source_name.upper()}: {METRIC_LABELS[metric]} vs sample size")
            plt.xticks(sample_sizes)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            out_path = os.path.join(PLOT_DIR, f"{metric}_{source_name}.png")
            plt.savefig(out_path, dpi=200)
            plt.close()

            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()