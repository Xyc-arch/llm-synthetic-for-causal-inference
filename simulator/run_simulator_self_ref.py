import os
import sys
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = "/home/ubuntu/syn_causal"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algs.aipw import estimate_aipw_df
from algs.ipw import estimate_ipw_df
from algs.outcome_regression import estimate_outcome_regression_df
from algs.tmle import estimate_tmle_df

# =========================
# Paths
# =========================

REAL_DIR = os.path.join(PROJECT_ROOT, "simulator", "real_data_n1000_rct")
REAL_TRUTH_PATH = os.path.join(REAL_DIR, "truth.json")

SYN_SOURCES = {
    "llm": os.path.join(PROJECT_ROOT, "llm_data", "syn_hybrid.csv"),
    "gan": os.path.join(PROJECT_ROOT, "gan_data", "syn_hybrid.csv"),
}

OUT_DIR = os.path.join(PROJECT_ROOT, "simulator", "results")
OUT_JSON = os.path.join(OUT_DIR, "simulation_engine_global_hybrid.json")
OUT_CSV = os.path.join(OUT_DIR, "simulation_engine_global_hybrid_summary.csv")

# =========================
# Experiment setup
# =========================

COVARIATES = ["W1", "W2", "W3", "W4", "W5", "W6"]
OUTCOME_COL = "Y"
TREATMENT_COL = "A"

N_REAL_DATASETS = 20
N_SYN_REPS = 20
REAL_SAMPLE_SIZE = 1000

REAL_DATASETS = {
    f"data_{i}": os.path.join(REAL_DIR, f"data_{i}.csv")
    for i in range(1, N_REAL_DATASETS + 1)
}

ESTIMATORS = {
    "ipw": estimate_ipw_df,
    "tmle": estimate_tmle_df,
    "aipw": estimate_aipw_df,
    "outcome_regression": estimate_outcome_regression_df,
}


# =========================
# Utilities
# =========================

def load_real_truth():
    with open(REAL_TRUTH_PATH, "r") as f:
        truth = json.load(f)
    return float(truth["ate_true"]), truth


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
        "n": int(len(arr)),
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

    real_ate_true, real_truth_meta = load_real_truth()

    results = {
        "config": {
            "project_root": PROJECT_ROOT,
            "real_dir": REAL_DIR,
            "real_truth_path": REAL_TRUTH_PATH,
            "synthetic_sources": SYN_SOURCES,
            "out_dir": OUT_DIR,
            "n_real_datasets": N_REAL_DATASETS,
            "n_syn_reps_per_source": N_SYN_REPS,
            "real_sample_size": REAL_SAMPLE_SIZE,
            "estimators": list(ESTIMATORS.keys()),
            "covariates": COVARIATES,
            "outcome_col": OUTCOME_COL,
            "treatment_col": TREATMENT_COL,
        },
        "real_truth": real_truth_meta,
        "real_group_against_real_truth": {},
        "synthetic_engine": {},
        "comparison_to_real_group": {},
    }

    summary_rows = []

    # ---------------------------------
    # 1) Real datasets against real truth
    # ---------------------------------
    real_collect = {e: [] for e in ESTIMATORS}
    real_dataset_results = {}

    for dataset_name, real_path in REAL_DATASETS.items():
        print(f"Evaluating real dataset {dataset_name}")
        real_df = pd.read_csv(real_path)

        real_dataset_results[dataset_name] = {}
        for est_name in ESTIMATORS:
            est = evaluate_estimator(real_df, est_name)
            real_dataset_results[dataset_name][est_name] = {
                "estimate": est,
                "truth_real": real_ate_true,
                "error_vs_real_truth": float(est - real_ate_true),
                "abs_error_vs_real_truth": float(abs(est - real_ate_true)),
            }
            real_collect[est_name].append(est)

    results["real_datasets"] = real_dataset_results

    for est_name in ESTIMATORS:
        results["real_group_against_real_truth"][est_name] = summarize_against_truth(
            estimates=real_collect[est_name],
            truth_value=real_ate_true,
        )

    # ---------------------------------
    # 2) Synthetic engine by source
    # ---------------------------------
    for source_name, source_path in SYN_SOURCES.items():
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Missing synthetic hybrid file: {source_path}")

        print(f"Evaluating synthetic engine for source: {source_name}")
        syn_pool = pd.read_csv(source_path)

        if len(syn_pool) < REAL_SAMPLE_SIZE:
            raise ValueError(
                f"Synthetic pool {source_name} has size {len(syn_pool)} < {REAL_SAMPLE_SIZE}."
            )

        results["synthetic_engine"][source_name] = {
            "source_file": source_path,
            "synthetic_reference_truths": {},
            "replicate_summaries": {},
        }

        # Estimator-specific synthetic reference truth on full synthetic pool
        syn_ref_truths = {}
        for est_name in ESTIMATORS:
            syn_ref_truths[est_name] = evaluate_estimator(syn_pool, est_name)

        results["synthetic_engine"][source_name]["synthetic_reference_truths"] = {
            est_name: {"estimate_on_full_hybrid_pool": float(val)}
            for est_name, val in syn_ref_truths.items()
        }

        # Repeated subsamples of size 1000 from the hybrid pool
        syn_rep_collect = {e: [] for e in ESTIMATORS}

        for rep in range(1, N_SYN_REPS + 1):
            rep_df = sample_without_replacement(
                syn_pool,
                n=REAL_SAMPLE_SIZE,
                seed=100000 + 1000 * (1 if source_name == "llm" else 2) + rep,
            )

            for est_name in ESTIMATORS:
                rep_est = evaluate_estimator(rep_df, est_name)
                syn_rep_collect[est_name].append(rep_est)

        # Summaries against synthetic truth and against real truth
        for est_name in ESTIMATORS:
            summary_internal = summarize_against_truth(
                estimates=syn_rep_collect[est_name],
                truth_value=syn_ref_truths[est_name],
            )
            summary_vs_real = summarize_against_truth(
                estimates=syn_rep_collect[est_name],
                truth_value=real_ate_true,
            )

            real_group_summary = results["real_group_against_real_truth"][est_name]

            comparison = {
                "real_group_rmse": float(real_group_summary["rmse"]),
                "real_group_mse": float(real_group_summary["mse"]),
                "real_group_bias": float(real_group_summary["bias"]),
                "real_group_var": float(real_group_summary["var"]),
                "synthetic_internal_rmse": float(summary_internal["rmse"]),
                "synthetic_internal_mse": float(summary_internal["mse"]),
                "synthetic_internal_bias": float(summary_internal["bias"]),
                "synthetic_internal_var": float(summary_internal["var"]),
                "synthetic_vs_real_rmse": float(summary_vs_real["rmse"]),
                "synthetic_vs_real_mse": float(summary_vs_real["mse"]),
                "synthetic_vs_real_bias": float(summary_vs_real["bias"]),
                "synthetic_vs_real_var": float(summary_vs_real["var"]),
                "abs_gap_rmse_real_vs_syn_internal": float(abs(real_group_summary["rmse"] - summary_internal["rmse"])),
                "abs_gap_rmse_real_vs_syn_real": float(abs(real_group_summary["rmse"] - summary_vs_real["rmse"])),
                "abs_gap_mse_real_vs_syn_internal": float(abs(real_group_summary["mse"] - summary_internal["mse"])),
                "abs_gap_mse_real_vs_syn_real": float(abs(real_group_summary["mse"] - summary_vs_real["mse"])),
                "abs_gap_bias_real_vs_syn_internal": float(abs(real_group_summary["bias"] - summary_internal["bias"])),
                "abs_gap_bias_real_vs_syn_real": float(abs(real_group_summary["bias"] - summary_vs_real["bias"])),
                "abs_gap_var_real_vs_syn_internal": float(abs(real_group_summary["var"] - summary_internal["var"])),
                "abs_gap_var_real_vs_syn_real": float(abs(real_group_summary["var"] - summary_vs_real["var"])),
            }

            results["synthetic_engine"][source_name]["replicate_summaries"][est_name] = {
                "against_synthetic_reference_truth": summary_internal,
                "against_real_truth": summary_vs_real,
                "comparison_to_real_group": comparison,
            }

            summary_rows.append(
                {
                    "source": source_name,
                    "estimator": est_name,
                    "synthetic_reference_truth": float(syn_ref_truths[est_name]),
                    "real_truth": float(real_ate_true),

                    "real_group_bias": float(real_group_summary["bias"]),
                    "real_group_var": float(real_group_summary["var"]),
                    "real_group_mse": float(real_group_summary["mse"]),
                    "real_group_rmse": float(real_group_summary["rmse"]),

                    "synthetic_internal_bias": float(summary_internal["bias"]),
                    "synthetic_internal_var": float(summary_internal["var"]),
                    "synthetic_internal_mse": float(summary_internal["mse"]),
                    "synthetic_internal_rmse": float(summary_internal["rmse"]),

                    "synthetic_vs_real_bias": float(summary_vs_real["bias"]),
                    "synthetic_vs_real_var": float(summary_vs_real["var"]),
                    "synthetic_vs_real_mse": float(summary_vs_real["mse"]),
                    "synthetic_vs_real_rmse": float(summary_vs_real["rmse"]),

                    "abs_gap_rmse_real_vs_syn_internal": float(abs(real_group_summary["rmse"] - summary_internal["rmse"])),
                    "abs_gap_rmse_real_vs_syn_real": float(abs(real_group_summary["rmse"] - summary_vs_real["rmse"])),
                    "abs_gap_mse_real_vs_syn_internal": float(abs(real_group_summary["mse"] - summary_internal["mse"])),
                    "abs_gap_mse_real_vs_syn_real": float(abs(real_group_summary["mse"] - summary_vs_real["mse"])),
                    "abs_gap_bias_real_vs_syn_internal": float(abs(real_group_summary["bias"] - summary_internal["bias"])),
                    "abs_gap_bias_real_vs_syn_real": float(abs(real_group_summary["bias"] - summary_vs_real["bias"])),
                    "abs_gap_var_real_vs_syn_internal": float(abs(real_group_summary["var"] - summary_internal["var"])),
                    "abs_gap_var_real_vs_syn_real": float(abs(real_group_summary["var"] - summary_vs_real["var"])),
                }
            )

    # ---------------------------------
    # 3) Winners by metric
    # ---------------------------------
    comparison_df = pd.DataFrame(summary_rows)

    metric_winners = {}
    for metric in [
        "abs_gap_rmse_real_vs_syn_internal",
        "abs_gap_rmse_real_vs_syn_real",
        "abs_gap_mse_real_vs_syn_internal",
        "abs_gap_mse_real_vs_syn_real",
        "abs_gap_bias_real_vs_syn_internal",
        "abs_gap_bias_real_vs_syn_real",
        "abs_gap_var_real_vs_syn_internal",
        "abs_gap_var_real_vs_syn_real",
    ]:
        winners = []
        for est_name in comparison_df["estimator"].unique():
            sub = comparison_df[comparison_df["estimator"] == est_name].copy()
            best_idx = sub[metric].idxmin()
            winners.append(
                {
                    "estimator": est_name,
                    "best_source": str(comparison_df.loc[best_idx, "source"]),
                    "best_value": float(comparison_df.loc[best_idx, metric]),
                }
            )
        metric_winners[metric] = winners

    results["comparison_to_real_group"]["metric_winners"] = metric_winners

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=4)

    comparison_df.to_csv(OUT_CSV, index=False)

    print(f"Saved detailed JSON to {OUT_JSON}")
    print(f"Saved summary CSV to {OUT_CSV}")


if __name__ == "__main__":
    main()