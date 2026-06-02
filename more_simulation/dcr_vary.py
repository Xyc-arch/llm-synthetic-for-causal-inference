#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


A_COL = "A"
Y_COL = "Y"
SEEDS = [1, 2, 3, 4, 5]

SYN_FILES = {
    "llm_syn_clean": "llm_data/syn_clean.csv",
    "llm_syn_hybrid": "llm_data/syn_hybrid.csv",
    "gan_syn_clean": "gan_data/syn_clean.csv",
    "gan_syn_hybrid": "gan_data/syn_hybrid.csv",
}

PLOT_LABELS = {
    "llm_syn_clean": "LLM",
    "llm_syn_hybrid": "LLM Hybrid",
    "gan_syn_clean": "GAN",
    "gan_syn_hybrid": "GAN Hybrid",
}


def get_w_cols(columns):
    return sorted(
        [c for c in columns if c.startswith("W") and c[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )


def validate_columns(df: pd.DataFrame, needed_cols, name: str):
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")


def min_distances(query_array, reference_array):
    out = []

    for row in query_array:
        dists = np.sqrt(((reference_array - row) ** 2).sum(axis=1))
        out.append(float(dists.min()))

    return out


def load_and_standardize(seed_path: Path, syn_path: Path, test_path: Path, distance_cols):
    seed = pd.read_csv(seed_path)
    syn = pd.read_csv(syn_path)
    test = pd.read_csv(test_path)

    validate_columns(seed, distance_cols, "seed data")
    validate_columns(syn, distance_cols, "synthetic data")
    validate_columns(test, distance_cols, "test data")

    seed = seed[distance_cols].copy()
    syn = syn[distance_cols].copy()
    test = test[distance_cols].copy()

    for c in distance_cols:
        seed[c] = pd.to_numeric(seed[c], errors="coerce")
        syn[c] = pd.to_numeric(syn[c], errors="coerce")
        test[c] = pd.to_numeric(test[c], errors="coerce")

    seed = seed.dropna().copy()
    syn = syn.dropna().copy()
    test = test.dropna().copy()

    if len(seed) == 0:
        raise ValueError(f"No valid seed rows after cleaning: {seed_path}")
    if len(syn) == 0:
        raise ValueError(f"No valid synthetic rows after cleaning: {syn_path}")
    if len(test) == 0:
        raise ValueError(f"No valid test rows after cleaning: {test_path}")

    means = seed[distance_cols].mean()
    stds = seed[distance_cols].std().replace(0, 1.0)

    seed_std = (seed[distance_cols] - means) / stds
    syn_std = (syn[distance_cols] - means) / stds
    test_std = (test[distance_cols] - means) / stds

    return seed_std, syn_std, test_std


def compute_dcr(seed_path: Path, syn_path: Path, test_path: Path, distance_cols, sample_seed=None):
    """
    Compute Distance to Closest Record for synthetic rows and test rows
    relative to the seed dataset.

    For d20, this now uses W1-W20, A, Y.
    For d6, this uses W1-W6, A, Y.
    """
    seed_std, syn_std, test_std = load_and_standardize(
        seed_path=seed_path,
        syn_path=syn_path,
        test_path=test_path,
        distance_cols=distance_cols,
    )

    n_test = test_std.shape[0]

    if syn_std.shape[0] > n_test:
        syn_std = syn_std.sample(n=n_test, random_state=sample_seed)

    seed_array = seed_std.to_numpy(dtype=float)
    syn_array = syn_std.to_numpy(dtype=float)
    test_array = test_std.to_numpy(dtype=float)

    syn_dcr = min_distances(syn_array, seed_array)
    test_dcr = min_distances(test_array, seed_array)

    return syn_dcr, test_dcr


def compute_dcr_repeated(seed_path: Path, syn_path: Path, test_path: Path, distance_cols):
    all_dcr_values = []
    mean_by_seed = []
    first_values = None

    for seed in SEEDS:
        syn_dcr, _ = compute_dcr(
            seed_path=seed_path,
            syn_path=syn_path,
            test_path=test_path,
            distance_cols=distance_cols,
            sample_seed=seed,
        )

        syn_dcr = list(map(float, syn_dcr))

        if first_values is None:
            first_values = syn_dcr

        all_dcr_values.extend(syn_dcr)
        mean_by_seed.append(float(np.mean(syn_dcr)))

    return all_dcr_values, mean_by_seed, first_values


def summarize(values):
    arr = np.asarray(values, dtype=float)

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(len(arr)),
    }


def summarize_repeated_means(mean_by_seed):
    arr = np.asarray(mean_by_seed, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    se = float(std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0

    return {
        "mean": mean,
        "std": std,
        "se": se,
        "ci95_low": float(mean - 1.96 * se),
        "ci95_high": float(mean + 1.96 * se),
        "n_reps": int(len(arr)),
        "seeds": SEEDS,
        "values": arr.tolist(),
    }


def round_nested(obj, digits=6):
    if isinstance(obj, dict):
        return {k: round_nested(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_nested(v, digits) for v in obj]
    if isinstance(obj, float):
        return round(obj, digits)
    return obj


def evaluate_setting(setting_dir: Path, output_dir: Path, make_plot=True):
    seed_file = setting_dir / "data_seed.csv"
    test_file = setting_dir / "data_test.csv"

    if not seed_file.exists():
        raise FileNotFoundError(f"Missing seed file: {seed_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Missing test file: {test_file}")

    seed_head = pd.read_csv(seed_file, nrows=5)
    w_cols = get_w_cols(seed_head.columns)

    if not w_cols:
        raise ValueError(f"No W columns found in {seed_file}")

    distance_cols = w_cols + [A_COL, Y_COL]

    print(f"Setting       : {setting_dir.name}")
    print(f"Distance cols : {distance_cols}")
    print(f"Seeds         : {SEEDS}")

    results = {}
    aggregate = {}

    plot_data = []
    plot_labels = []

    existing_syn = [
        setting_dir / rel_path
        for rel_path in SYN_FILES.values()
        if (setting_dir / rel_path).exists()
    ]

    if not existing_syn:
        raise FileNotFoundError(f"No synthetic files found in {setting_dir}")

    # Test DCR is deterministic with respect to seed data.
    _, test_dcr = compute_dcr(
        seed_path=seed_file,
        syn_path=existing_syn[0],
        test_path=test_file,
        distance_cols=distance_cols,
        sample_seed=SEEDS[0],
    )

    test_summary = summarize(test_dcr)

    results["data_test"] = {
        "file": str(test_file),
        "distance_cols": distance_cols,
        "summary": test_summary,
        "dcr_values": test_dcr,
    }

    aggregate["data_test"] = {
        "file": str(test_file),
        "distance_cols": distance_cols,
        "summary": round_nested(test_summary),
    }

    plot_data.append(test_dcr)
    plot_labels.append("Data Test")

    print(
        "Data Test: Mean DCR = {:.4f}, Std DCR = {:.4f}, Median DCR = {:.4f}".format(
            test_summary["mean"],
            test_summary["std"],
            test_summary["median"],
        )
    )

    plot_order = [
        "llm_syn_hybrid",
        "llm_syn_clean",
        "gan_syn_hybrid",
        "gan_syn_clean",
    ]

    for name in plot_order:
        syn_path = setting_dir / SYN_FILES[name]

        if not syn_path.exists():
            print(f"Skipping missing synthetic file for {name}: {syn_path}")
            continue

        all_dcr_values, mean_by_seed, first_values = compute_dcr_repeated(
            seed_path=seed_file,
            syn_path=syn_path,
            test_path=test_file,
            distance_cols=distance_cols,
        )

        record_summary = summarize(all_dcr_values)
        repeated_summary = summarize_repeated_means(mean_by_seed)

        results[name] = {
            "file": str(syn_path),
            "distance_cols": distance_cols,
            "seeds": SEEDS,
            "summary": record_summary,
            "repeated_mean_summary": repeated_summary,
            "dcr_values_first_seed": first_values,
            "dcr_mean_by_seed": mean_by_seed,
        }

        aggregate[name] = {
            "file": str(syn_path),
            "distance_cols": distance_cols,
            "summary": round_nested(record_summary),
            "repeated_mean_summary": round_nested(repeated_summary),
        }

        plot_data.append(first_values)
        plot_labels.append(PLOT_LABELS[name])

        print(
            f"{name}: Mean DCR = {repeated_summary['mean']:.4f} ± {repeated_summary['se']:.4f}, "
            f"Median DCR = {record_summary['median']:.4f}"
        )

    setting_out_dir = output_dir / setting_dir.name
    setting_out_dir.mkdir(parents=True, exist_ok=True)

    output_json = setting_out_dir / "dcr.json"
    output_agg_json = setting_out_dir / "dcr_aggregate.json"

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    with open(output_agg_json, "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"DCR results saved to: {output_json}")
    print(f"Aggregate DCR results saved to: {output_agg_json}")

    if make_plot:
        output_plot = setting_out_dir / "dcr_boxplot.png"

        plt.figure(figsize=(10, 6))
        plt.boxplot(plot_data, labels=plot_labels)
        plt.ylabel("Distance to Closest Record")
        plt.title(f"DCR Boxplot: {setting_dir.name}")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(output_plot, dpi=200)
        plt.close()

        print(f"Boxplot saved to: {output_plot}")

    return {
        "setting": setting_dir.name,
        "setting_dir": str(setting_dir),
        "distance_cols": distance_cols,
        "results": aggregate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="simulator_vary_data")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="vary_results_dcr")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_dir is not None:
        setting_dirs = [Path(args.data_dir)]
    else:
        root = script_dir / args.root
        if not root.exists():
            raise FileNotFoundError(f"Missing root folder: {root}")

        setting_dirs = [
            p for p in sorted(root.iterdir())
            if p.is_dir() and (p / "data_seed.csv").exists() and (p / "data_test.csv").exists()
        ]

    all_results = {
        "seeds": SEEDS,
        "settings": {},
    }

    for setting_dir in setting_dirs:
        print("=" * 100)
        print(f"Running DCR for setting: {setting_dir}")
        print("=" * 100)

        setting_results = evaluate_setting(
            setting_dir=setting_dir,
            output_dir=output_dir,
            make_plot=not args.no_plot,
        )
        all_results["settings"][setting_dir.name] = setting_results

    full_out = output_dir / "dcr_all_settings.json"
    with open(full_out, "w") as f:
        json.dump(all_results, f, indent=2)

    compact = {
        "seeds": SEEDS,
        "settings": {},
    }

    for setting_name, setting_results in all_results["settings"].items():
        compact["settings"][setting_name] = {
            "distance_cols": setting_results["distance_cols"],
            "datasets": {},
        }

        for ds, metrics in setting_results["results"].items():
            compact["settings"][setting_name]["datasets"][ds] = {
                "mean": metrics["repeated_mean_summary"]["mean"]
                if "repeated_mean_summary" in metrics
                else metrics["summary"]["mean"],
                "se": metrics["repeated_mean_summary"]["se"]
                if "repeated_mean_summary" in metrics
                else 0.0,
                "median": metrics["summary"]["median"],
                "n": metrics["summary"]["n"],
            }

    compact_out = output_dir / "dcr_aggregate.json"
    with open(compact_out, "w") as f:
        json.dump(round_nested(compact), f, indent=2)

    print(f"Saved all-setting DCR results to {full_out}")
    print(f"Saved compact DCR aggregate to {compact_out}")


if __name__ == "__main__":
    main()