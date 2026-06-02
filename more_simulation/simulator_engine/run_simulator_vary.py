#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
MORE_SIM_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = MORE_SIM_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algs.aipw import estimate_aipw_df
from algs.ipw import estimate_ipw_df
from algs.outcome_regression import estimate_outcome_regression_df
from algs.tmle import estimate_tmle_df


DEFAULT_SYN_ROOT = MORE_SIM_DIR / "simulator_vary_data"
DEFAULT_REAL_ROOT = SCRIPT_DIR / "real_data_vary"
DEFAULT_OUT_DIR = SCRIPT_DIR / "results"

REAL_SAMPLE_SIZE = 200
N_REAL_DATASETS = 20
N_SYN_REPS = 20

OUTCOME_COL = "Y"
TREATMENT_COL = "A"

ESTIMATORS = {
    "ipw": estimate_ipw_df,
    "tmle": estimate_tmle_df,
    "aipw": estimate_aipw_df,
    "outcome_regression": estimate_outcome_regression_df,
}

SYNTHETIC_REFERENCE_ESTIMATOR = "tmle"


def get_w_cols(df):
    return sorted(
        [c for c in df.columns if c.startswith("W") and c[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )


def load_truth(real_setting_dir: Path):
    truth_path = real_setting_dir / "truth.json"
    if not truth_path.exists():
        raise FileNotFoundError(f"Missing truth.json: {truth_path}")

    with open(truth_path, "r") as f:
        truth = json.load(f)

    return float(truth["ate_true"]), truth


def evaluate_estimator(df, estimator_name, covariates):
    fn = ESTIMATORS[estimator_name]

    kwargs = {
        "covariates": covariates,
        "outcome_col": OUTCOME_COL,
        "treatment_col": TREATMENT_COL,
        "random_state": 42,
    }

    try:
        return float(fn(df.copy(), **kwargs))
    except TypeError:
        kwargs["data"] = df.copy()
        return float(fn(**kwargs))


def sample_without_replacement(df, n, seed):
    if n > len(df):
        raise ValueError(f"Requested sample size {n} exceeds pool size {len(df)}.")

    rng = np.random.default_rng(seed)
    idx = rng.choice(df.index.to_numpy(), size=n, replace=False)
    return df.loc[idx].reset_index(drop=True)


def summarize_against_truth(estimates, truth_value):
    arr = np.asarray(estimates, dtype=float)
    errors = arr - float(truth_value)
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2

    n = len(arr)

    mean_estimate = float(np.mean(arr))
    std_estimate = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se_estimate = float(std_estimate / np.sqrt(n)) if n > 1 else 0.0

    bias = float(mean_estimate - truth_value)
    bias_std = float(np.std(errors, ddof=1)) if n > 1 else 0.0
    bias_se = float(bias_std / np.sqrt(n)) if n > 1 else 0.0

    mae = float(np.mean(abs_errors))
    mae_std = float(np.std(abs_errors, ddof=1)) if n > 1 else 0.0
    mae_se = float(mae_std / np.sqrt(n)) if n > 1 else 0.0

    mse = float(np.mean(sq_errors))
    mse_std = float(np.std(sq_errors, ddof=1)) if n > 1 else 0.0
    mse_se = float(mse_std / np.sqrt(n)) if n > 1 else 0.0

    return {
        "n": int(n),
        "estimates": arr.tolist(),

        "mean_estimate": mean_estimate,
        "std_estimate": std_estimate,
        "se_estimate": se_estimate,

        "bias": bias,
        "bias_std": bias_std,
        "bias_se": bias_se,
        "abs_bias": float(abs(bias)),

        "mae": mae,
        "mae_std": mae_std,
        "mae_se": mae_se,

        "var": float(np.var(arr, ddof=1)) if n > 1 else 0.0,

        "mse": mse,
        "mse_std": mse_std,
        "mse_se": mse_se,
        "rmse": float(np.sqrt(mse)),

        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "truth": float(truth_value),
    }


def add_summary_fields(prefix, summary):
    return {
        f"{prefix}_mean_estimate": float(summary["mean_estimate"]),
        f"{prefix}_std_estimate": float(summary["std_estimate"]),
        f"{prefix}_se_estimate": float(summary["se_estimate"]),

        f"{prefix}_bias": float(summary["bias"]),
        f"{prefix}_bias_std": float(summary["bias_std"]),
        f"{prefix}_bias_se": float(summary["bias_se"]),
        f"{prefix}_abs_bias": float(summary["abs_bias"]),

        f"{prefix}_mae": float(summary["mae"]),
        f"{prefix}_mae_std": float(summary["mae_std"]),
        f"{prefix}_mae_se": float(summary["mae_se"]),

        f"{prefix}_var": float(summary["var"]),

        f"{prefix}_mse": float(summary["mse"]),
        f"{prefix}_mse_std": float(summary["mse_std"]),
        f"{prefix}_mse_se": float(summary["mse_se"]),
        f"{prefix}_rmse": float(summary["rmse"]),

        f"{prefix}_min": float(summary["min"]),
        f"{prefix}_max": float(summary["max"]),
        f"{prefix}_truth": float(summary["truth"]),
        f"{prefix}_n": int(summary["n"]),
    }


def run_setting(setting_name, syn_setting_dir: Path, real_setting_dir: Path, out_dir: Path, reference_mode: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    real_ate_true, real_truth_meta = load_truth(real_setting_dir)

    real_datasets = {
        f"data_{i}": real_setting_dir / f"data_{i}.csv"
        for i in range(1, N_REAL_DATASETS + 1)
    }

    sources = {
        "llm": syn_setting_dir / "llm_data" / "syn_hybrid.csv",
        "gan": syn_setting_dir / "gan_data" / "syn_hybrid.csv",
    }

    first_real = pd.read_csv(next(iter(real_datasets.values())))
    covariates = get_w_cols(first_real)

    if not covariates:
        raise ValueError(f"No W covariates found in {real_setting_dir}")

    results = {
        "config": {
            "setting": setting_name,
            "synthetic_setting_dir": str(syn_setting_dir),
            "real_setting_dir": str(real_setting_dir),
            "synthetic_sources": {k: str(v) for k, v in sources.items()},
            "n_real_datasets": N_REAL_DATASETS,
            "n_syn_reps_per_source": N_SYN_REPS,
            "real_sample_size": REAL_SAMPLE_SIZE,
            "estimators": list(ESTIMATORS.keys()),
            "covariates": covariates,
            "outcome_col": OUTCOME_COL,
            "treatment_col": TREATMENT_COL,
            "reference_mode": reference_mode,
            "synthetic_reference_estimator": SYNTHETIC_REFERENCE_ESTIMATOR
            if reference_mode == "self_ref"
            else None,
        },
        "real_truth": real_truth_meta,
        "real_group_against_real_truth": {},
        "real_datasets": {},
        "synthetic_engine": {},
        "comparison_to_real_group": {},
    }

    summary_rows = []
    real_collect = {e: [] for e in ESTIMATORS}

    for dataset_name, real_path in real_datasets.items():
        if not real_path.exists():
            raise FileNotFoundError(f"Missing real dataset: {real_path}")

        print(f"[{setting_name}] Evaluating real dataset {dataset_name}")
        real_df = pd.read_csv(real_path)

        results["real_datasets"][dataset_name] = {}

        for est_name in ESTIMATORS:
            est = evaluate_estimator(real_df, est_name, covariates)
            results["real_datasets"][dataset_name][est_name] = {
                "estimate": est,
                "truth_real": real_ate_true,
                "error_vs_real_truth": float(est - real_ate_true),
                "abs_error_vs_real_truth": float(abs(est - real_ate_true)),
                "sq_error_vs_real_truth": float((est - real_ate_true) ** 2),
            }
            real_collect[est_name].append(est)

    for est_name in ESTIMATORS:
        results["real_group_against_real_truth"][est_name] = summarize_against_truth(
            estimates=real_collect[est_name],
            truth_value=real_ate_true,
        )

    for source_name, source_path in sources.items():
        if not source_path.exists():
            print(f"[{setting_name}] Skipping missing synthetic source: {source_path}")
            continue

        print(f"[{setting_name}] Evaluating synthetic source: {source_name}")
        syn_pool = pd.read_csv(source_path)

        if len(syn_pool) < REAL_SAMPLE_SIZE:
            raise ValueError(
                f"Synthetic pool {source_name} has size {len(syn_pool)} < {REAL_SAMPLE_SIZE}."
            )

        results["synthetic_engine"][source_name] = {
            "source_file": str(source_path),
            "synthetic_reference_truth": {},
            "replicate_summaries": {},
        }

        if reference_mode == "self_ref":
            common_syn_truth = evaluate_estimator(
                syn_pool,
                SYNTHETIC_REFERENCE_ESTIMATOR,
                covariates,
            )
            syn_ref_truths = {est_name: common_syn_truth for est_name in ESTIMATORS}
            results["synthetic_engine"][source_name]["synthetic_reference_truth"] = {
                "reference_mode": "self_ref",
                "estimator_used": SYNTHETIC_REFERENCE_ESTIMATOR,
                "estimate_on_full_hybrid_pool": float(common_syn_truth),
            }
        elif reference_mode == "estimator_specific":
            syn_ref_truths = {
                est_name: evaluate_estimator(syn_pool, est_name, covariates)
                for est_name in ESTIMATORS
            }
            results["synthetic_engine"][source_name]["synthetic_reference_truth"] = {
                "reference_mode": "estimator_specific",
                "estimator_specific_truths": {
                    est_name: float(val)
                    for est_name, val in syn_ref_truths.items()
                },
            }
        else:
            raise ValueError(f"Unknown reference_mode: {reference_mode}")

        syn_rep_collect = {e: [] for e in ESTIMATORS}

        for rep in range(1, N_SYN_REPS + 1):
            rep_df = sample_without_replacement(
                syn_pool,
                n=REAL_SAMPLE_SIZE,
                seed=100000 + 1000 * (1 if source_name == "llm" else 2) + rep,
            )

            for est_name in ESTIMATORS:
                rep_est = evaluate_estimator(rep_df, est_name, covariates)
                syn_rep_collect[est_name].append(rep_est)

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
                "real_group_mse_se": float(real_group_summary["mse_se"]),
                "real_group_bias": float(real_group_summary["bias"]),
                "real_group_bias_se": float(real_group_summary["bias_se"]),
                "real_group_var": float(real_group_summary["var"]),

                "synthetic_internal_rmse": float(summary_internal["rmse"]),
                "synthetic_internal_mse": float(summary_internal["mse"]),
                "synthetic_internal_mse_se": float(summary_internal["mse_se"]),
                "synthetic_internal_bias": float(summary_internal["bias"]),
                "synthetic_internal_bias_se": float(summary_internal["bias_se"]),
                "synthetic_internal_var": float(summary_internal["var"]),

                "synthetic_vs_real_rmse": float(summary_vs_real["rmse"]),
                "synthetic_vs_real_mse": float(summary_vs_real["mse"]),
                "synthetic_vs_real_mse_se": float(summary_vs_real["mse_se"]),
                "synthetic_vs_real_bias": float(summary_vs_real["bias"]),
                "synthetic_vs_real_bias_se": float(summary_vs_real["bias_se"]),
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

            row = {
                "setting": setting_name,
                "source": source_name,
                "estimator": est_name,
                "reference_mode": reference_mode,
                "synthetic_reference_truth": float(syn_ref_truths[est_name]),
                "real_truth": float(real_ate_true),
            }
            row.update(add_summary_fields("real_group", real_group_summary))
            row.update(add_summary_fields("synthetic_internal", summary_internal))
            row.update(add_summary_fields("synthetic_vs_real", summary_vs_real))
            row.update(comparison)
            summary_rows.append(row)

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

    suffix = "self_ref" if reference_mode == "self_ref" else "estimator_specific"
    out_json = out_dir / f"{setting_name}_simulation_engine_{suffix}.json"
    out_csv = out_dir / f"{setting_name}_simulation_engine_{suffix}_summary.csv"

    with open(out_json, "w") as f:
        json.dump(results, f, indent=4)

    comparison_df.to_csv(out_csv, index=False)

    print(f"[{setting_name}] Saved detailed JSON to {out_json}")
    print(f"[{setting_name}] Saved summary CSV to {out_csv}")

    return comparison_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--syn-root", type=str, default=str(DEFAULT_SYN_ROOT))
    parser.add_argument("--real-root", type=str, default=str(DEFAULT_REAL_ROOT))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--reference-mode",
        type=str,
        choices=["estimator_specific", "self_ref"],
        default="self_ref",
    )
    parser.add_argument("--setting", type=str, default=None)
    args = parser.parse_args()

    syn_root = Path(args.syn_root)
    real_root = Path(args.real_root)
    out_dir = Path(args.out_dir)

    if not syn_root.exists():
        raise FileNotFoundError(f"Missing synthetic root: {syn_root}")
    if not real_root.exists():
        raise FileNotFoundError(f"Missing real root: {real_root}")

    if args.setting is not None:
        setting_names = [args.setting]
    else:
        setting_names = sorted([
            p.name for p in real_root.iterdir()
            if p.is_dir() and (p / "truth.json").exists()
        ])

    all_summaries = []

    for setting_name in setting_names:
        syn_setting_dir = syn_root / setting_name
        real_setting_dir = real_root / setting_name

        if not syn_setting_dir.exists():
            print(f"Skipping {setting_name}; missing synthetic setting dir: {syn_setting_dir}")
            continue
        if not real_setting_dir.exists():
            print(f"Skipping {setting_name}; missing real setting dir: {real_setting_dir}")
            continue

        summary_df = run_setting(
            setting_name=setting_name,
            syn_setting_dir=syn_setting_dir,
            real_setting_dir=real_setting_dir,
            out_dir=out_dir,
            reference_mode=args.reference_mode,
        )
        all_summaries.append(summary_df)

    if all_summaries:
        suffix = "self_ref" if args.reference_mode == "self_ref" else "estimator_specific"
        all_df = pd.concat(all_summaries, ignore_index=True)
        all_csv = out_dir / f"all_settings_simulation_engine_{suffix}_summary.csv"
        all_json = out_dir / f"all_settings_simulation_engine_{suffix}_summary.json"

        all_df.to_csv(all_csv, index=False)
        all_df.to_json(all_json, orient="records", indent=2)

        print(f"Saved all-setting summary CSV to {all_csv}")
        print(f"Saved all-setting summary JSON to {all_json}")


if __name__ == "__main__":
    main()