#!/usr/bin/env python3
import os
import json

PROJECT_ROOT = "/home/ubuntu/syn_causal"
ACTG_DIR = os.path.join(PROJECT_ROOT, "actg175")

SIM_JSON = os.path.join(ACTG_DIR, "results", "simulation_engine_synth_only.json")
FULL_JSON = os.path.join(ACTG_DIR, "results", "full_sample_synth.json")
OUT_JSON = os.path.join(ACTG_DIR, "results", "aggregate_simulator_llm.json")

SOURCE_NAME = "llm"

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


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with open(path, "r") as f:
        return json.load(f)


def main():
    sim = load_json(SIM_JSON)
    full = load_json(FULL_JSON)

    if SOURCE_NAME not in sim["synthetic_engine"]:
        raise ValueError(f"Missing {SOURCE_NAME} in {SIM_JSON}")

    if SOURCE_NAME not in full:
        raise ValueError(f"Missing {SOURCE_NAME} in {FULL_JSON}")

    source_obj = sim["synthetic_engine"][SOURCE_NAME]
    full_obj = full[SOURCE_NAME]

    full_tmle = float(full_obj["tmle_reference"])
    full_estimates = {
        est: float(val)
        for est, val in full_obj["estimates"].items()
    }

    output = {
        "source": SOURCE_NAME,
        "n_full_pool": int(full_obj["n_full_pool"]),
        "full_synth_tmle": full_tmle,
        "full_synth_estimates": {},
        "by_sample_size": {},
    }

    for est_name in ESTIMATOR_ORDER:
        if est_name not in full_estimates:
            continue

        full_est = full_estimates[est_name]
        output["full_synth_estimates"][est_name] = {
            "label": ESTIMATOR_LABELS.get(est_name, est_name),
            "estimate": full_est,
            "diff_from_full_tmle": full_est - full_tmle,
        }

    for n_str in sorted(source_obj["by_sample_size"].keys(), key=lambda x: int(x)):
        output["by_sample_size"][n_str] = {}

        for est_name in ESTIMATOR_ORDER:
            if est_name not in source_obj["by_sample_size"][n_str]:
                continue
            if est_name not in full_estimates:
                continue

            summary = source_obj["by_sample_size"][n_str][est_name][
                "against_synthetic_reference_truth"
            ]

            finite_mean = float(summary["mean_estimate"])
            full_est = full_estimates[est_name]

            output["by_sample_size"][n_str][est_name] = {
                "label": ESTIMATOR_LABELS.get(est_name, est_name),
                "bias_vs_full_tmle": float(summary["bias"]),
                "mse_vs_full_tmle": float(summary["mse"]),
                "diff_mean_from_own_full": finite_mean - full_est,
            }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Saved LLM-only aggregate JSON to {OUT_JSON}")


if __name__ == "__main__":
    main()