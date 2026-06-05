#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algs.aipw import estimate_aipw_df
from algs.ipw import estimate_ipw_df
from algs.outcome_regression import estimate_outcome_regression_df
from algs.tmle import estimate_tmle_df


OUTCOME_COL = "Y"
TREATMENT_COL = "A"
SEEDS = [1, 2, 3, 4, 5]
SUBSAMPLE_N = 1000
EXPERIMENT_NAME = "logistic_outcome_misspecification"


def get_w_cols(columns):
    return sorted(
        [c for c in columns if c.startswith("W") and c[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )


def load_truth(setting_dir: Path):
    truth_path = setting_dir / "truth.json"
    if not truth_path.exists():
        raise FileNotFoundError(f"Missing truth.json: {truth_path}")

    with open(truth_path, "r") as f:
        truth = json.load(f)

    return float(truth["ate_true"]), truth


def candidate_datasets(setting_dir: Path):
    return {
        "llm_syn_clean": setting_dir / "llm_data" / "syn_clean.csv",
        "llm_syn_hybrid": setting_dir / "llm_data" / "syn_hybrid.csv",
        "gan_syn_clean": setting_dir / "gan_data" / "syn_clean.csv",
        "gan_syn_hybrid": setting_dir / "gan_data" / "syn_hybrid.csv",
    }


def summarize_estimates(estimates, ate_true):
    estimates = np.asarray(estimates, dtype=float)
    errors = estimates - ate_true
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2

    n = len(estimates)

    mean_est = float(np.mean(estimates))
    std_est = float(np.std(estimates, ddof=1)) if n > 1 else 0.0
    se_est = float(std_est / np.sqrt(n)) if n > 1 else 0.0

    bias = float(mean_est - ate_true)

    mse = float(np.mean(sq_errors))
    mse_std = float(np.std(sq_errors, ddof=1)) if n > 1 else 0.0
    mse_se = float(mse_std / np.sqrt(n)) if n > 1 else 0.0

    mae = float(np.mean(abs_errors))
    mae_std = float(np.std(abs_errors, ddof=1)) if n > 1 else 0.0
    mae_se = float(mae_std / np.sqrt(n)) if n > 1 else 0.0

    return {
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
        "min": float(np.min(estimates)),
        "max": float(np.max(estimates)),
    }


def get_estimator_config(estimator_name):
    """
    Logistic-outcome misspecification comparison.

    Main RF comparison:
      g = random forest
      Q = random forest

    This script:
      g = random forest for all estimators that use a propensity model
      Q = logistic regression for estimators that use an outcome model

    Because Y is binary, all estimators treat Y as binary here.

    IPW has no Q model, so it is included as a weighting-only benchmark.
    The complex settings are misspecified because the true outcome model is
    nonlinear while Q is restricted to logistic regression.
    """
    if estimator_name == "ipw":
        return {
            "outcome_type": "binary",
            "g_learner": "rf",
            "q_learner": None,
        }

    if estimator_name == "aipw":
        return {
            "outcome_type": "binary",
            "g_learner": "rf",
            "q_learner": "logistic",
        }

    if estimator_name == "outcome_regression":
        return {
            "outcome_type": "binary",
            "g_learner": None,
            "q_learner": "logistic",
        }

    if estimator_name == "tmle":
        return {
            "outcome_type": "binary",
            "g_learner": "rf",
            "q_learner": "logistic",
        }

    raise ValueError(f"Unknown estimator_name: {estimator_name}")


def evaluate_dataset(path, estimator_fn, estimator_name, covariates, ate_true):
    df = pd.read_csv(path)

    needed = covariates + [TREATMENT_COL, OUTCOME_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df = df[needed].copy()

    for c in covariates:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[TREATMENT_COL] = (
        pd.to_numeric(df[TREATMENT_COL], errors="coerce")
        .round()
        .astype("Int64")
    )

    df[OUTCOME_COL] = (
        pd.to_numeric(df[OUTCOME_COL], errors="coerce")
        .round()
        .astype("Int64")
    )

    before = len(df)
    df = df.dropna().copy()
    after = len(df)

    for c in [TREATMENT_COL, OUTCOME_COL]:
        df[c] = df[c].astype(int)

    n = len(df)
    subsample_n = min(SUBSAMPLE_N, n)

    if n == 0:
        raise ValueError(f"No valid rows after cleaning for {path}")

    config = get_estimator_config(estimator_name)

    estimates = []

    for seed in SEEDS:
        sub = df.sample(n=subsample_n, random_state=seed).copy()

        est_kwargs = {
            "covariates": covariates,
            "outcome_col": OUTCOME_COL,
            "treatment_col": TREATMENT_COL,
            "random_state": 42,
            "outcome_type": config["outcome_type"],
        }

        if config["g_learner"] is not None:
            est_kwargs["g_learner"] = config["g_learner"]

        if config["q_learner"] is not None:
            est_kwargs["q_learner"] = config["q_learner"]

        est = estimator_fn(sub, **est_kwargs)
        estimates.append(float(est))

    summary = summarize_estimates(estimates, ate_true)

    summary.update(
        {
            "file": str(path),
            "estimator": estimator_name,
            "n_full": int(before),
            "n_valid": int(after),
            "subsample_n": int(subsample_n),
            "n_reps": len(SEEDS),
            "seeds": SEEDS,
            "ate_true": float(ate_true),
            "covariates": covariates,
            "outcome_type": config["outcome_type"],
            "q_learner": config["q_learner"],
            "g_learner": config["g_learner"],
            "experiment": EXPERIMENT_NAME,
        }
    )

    return summary


def evaluate_setting(setting_dir: Path):
    ate_true, truth = load_truth(setting_dir)

    seed_path = setting_dir / "data_seed.csv"
    if not seed_path.exists():
        raise FileNotFoundError(f"Missing data_seed.csv: {seed_path}")

    seed_df = pd.read_csv(seed_path, nrows=5)
    covariates = get_w_cols(seed_df.columns)

    if not covariates:
        raise ValueError(f"No W covariates found in {seed_path}")

    estimators = {
        "aipw": estimate_aipw_df,
        "ipw": estimate_ipw_df,
        "outcome_regression": estimate_outcome_regression_df,
        "tmle": estimate_tmle_df,
    }

    results = {
        "setting": setting_dir.name,
        "setting_dir": str(setting_dir),
        "truth": truth,
        "ate_true": float(ate_true),
        "covariates": covariates,
        "d": len(covariates),
        "subsample_n": SUBSAMPLE_N,
        "seeds": SEEDS,
        "experiment": EXPERIMENT_NAME,
        "estimator_configs": {
            est_name: get_estimator_config(est_name)
            for est_name in estimators.keys()
        },
        "datasets": {},
    }

    for dataset_name, path in candidate_datasets(setting_dir).items():
        if not path.exists():
            print(f"Skipping missing dataset: {path}")
            continue

        results["datasets"][dataset_name] = {}

        for est_name, est_fn in estimators.items():
            config = get_estimator_config(est_name)

            print(
                f"Running setting={setting_dir.name} "
                f"dataset={dataset_name} "
                f"estimator={est_name} "
                f"outcome_type={config['outcome_type']} "
                f"g_learner={config['g_learner']} "
                f"q_learner={config['q_learner']}"
            )

            try:
                results["datasets"][dataset_name][est_name] = evaluate_dataset(
                    path=path,
                    estimator_fn=est_fn,
                    estimator_name=est_name,
                    covariates=covariates,
                    ate_true=ate_true,
                )
            except Exception as e:
                print(
                    f"ERROR setting={setting_dir.name} "
                    f"dataset={dataset_name} "
                    f"estimator={est_name}: {e}"
                )

                results["datasets"][dataset_name][est_name] = {
                    "error": str(e),
                    "file": str(path),
                    "estimator": est_name,
                    "outcome_type": config["outcome_type"],
                    "q_learner": config["q_learner"],
                    "g_learner": config["g_learner"],
                    "experiment": EXPERIMENT_NAME,
                }

    return results


def compact_results(all_results):
    compact = {
        "subsample_n": SUBSAMPLE_N,
        "seeds": SEEDS,
        "experiment": EXPERIMENT_NAME,
        "settings": {},
    }

    for setting_name, setting_results in all_results["settings"].items():
        truth = setting_results.get("truth", {})

        compact["settings"][setting_name] = {
            "ate_true": setting_results.get("ate_true"),
            "d": setting_results.get("d"),
            "n_seed": truth.get("n_seed"),
            "n_test": truth.get("n_test"),
            "overlap": truth.get("overlap"),
            "outcome_mode": truth.get("outcome_mode"),
            "experiment": EXPERIMENT_NAME,
            "estimator_configs": setting_results.get("estimator_configs", {}),
            "datasets": {},
        }

        for dataset_name, dataset_results in setting_results.get("datasets", {}).items():
            compact["settings"][setting_name]["datasets"][dataset_name] = {}

            for est_name, metrics in dataset_results.items():
                config = get_estimator_config(est_name)

                if "error" in metrics:
                    compact["settings"][setting_name]["datasets"][dataset_name][est_name] = {
                        "error": metrics["error"],
                        "outcome_type": metrics.get("outcome_type", config["outcome_type"]),
                        "q_learner": metrics.get("q_learner", config["q_learner"]),
                        "g_learner": metrics.get("g_learner", config["g_learner"]),
                        "experiment": EXPERIMENT_NAME,
                    }
                    continue

                compact["settings"][setting_name]["datasets"][dataset_name][est_name] = {
                    "mean": round(metrics["mean"], 6),
                    "se": round(metrics["se"], 6),
                    "bias": round(metrics["bias"], 6),
                    "abs_bias": round(metrics["abs_bias"], 6),
                    "mse": round(metrics["mse"], 6),
                    "mse_se": round(metrics["mse_se"], 6),
                    "rmse": round(metrics["rmse"], 6),
                    "n_valid": metrics["n_valid"],
                    "subsample_n": metrics["subsample_n"],
                    "outcome_type": metrics.get("outcome_type", config["outcome_type"]),
                    "q_learner": metrics.get("q_learner", config["q_learner"]),
                    "g_learner": metrics.get("g_learner", config["g_learner"]),
                    "experiment": EXPERIMENT_NAME,
                }

    return compact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="simulator_vary_data")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="vary_results_logistic",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        "subsample_n": SUBSAMPLE_N,
        "seeds": SEEDS,
        "experiment": EXPERIMENT_NAME,
        "settings": {},
    }

    if args.data_dir is not None:
        setting_dirs = [Path(args.data_dir)]
    else:
        root = base_dir / args.root
        setting_dirs = [
            p for p in sorted(root.iterdir())
            if p.is_dir()
            and (p / "data_seed.csv").exists()
            and (p / "truth.json").exists()
        ]

    for setting_dir in setting_dirs:
        print("=" * 100)
        print(f"Evaluating setting: {setting_dir}")
        print("=" * 100)

        setting_results = evaluate_setting(setting_dir)
        all_results["settings"][setting_dir.name] = setting_results

        per_setting_out = output_dir / f"{setting_dir.name}_estimators_logistic.json"
        with open(per_setting_out, "w") as f:
            json.dump(setting_results, f, indent=2)

        print(f"Saved per-setting results to {per_setting_out}")

    full_out = output_dir / "vary_estimators_logistic.json"
    with open(full_out, "w") as f:
        json.dump(all_results, f, indent=2)

    compact = compact_results(all_results)
    compact_out = output_dir / "vary_estimators_logistic_compact.json"
    with open(compact_out, "w") as f:
        json.dump(compact, f, indent=2)

    print(f"Saved full results to {full_out}")
    print(f"Saved compact results to {compact_out}")


if __name__ == "__main__":
    main()