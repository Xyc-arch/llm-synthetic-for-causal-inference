#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DISTANCE_COLS = ["W1", "W2", "W3", "W4", "W5", "W6", "A", "Y"]
SEEDS = [1, 2, 3, 4, 5]

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_JSON = RESULTS_DIR / "dcr.json"
OUTPUT_AGG_JSON = RESULTS_DIR / "dcr_aggregate.json"
OUTPUT_PLOT = RESULTS_DIR / "dcr_boxplot.png"


def detect_data_dir() -> Path:
    """
    Use current working directory if it contains the core files.
    Otherwise fall back to the directory containing this script.
    """
    cwd = Path.cwd()

    if (cwd / "data_seed.csv").exists() and (cwd / "data_test.csv").exists():
        return cwd

    return SCRIPT_DIR


DATA_DIR = detect_data_dir()

SEED_FILE = DATA_DIR / "data_seed.csv"
TEST_FILE = DATA_DIR / "data_test.csv"

SYN_FILES = {
    "llm_syn_clean": DATA_DIR / "llm_data" / "syn_clean.csv",
    "llm_syn_hybrid": DATA_DIR / "llm_data" / "syn_hybrid.csv",
    "gan_syn_clean": DATA_DIR / "gan_data" / "syn_clean.csv",
    "gan_syn_hybrid": DATA_DIR / "gan_data" / "syn_hybrid.csv",
}

PLOT_LABELS = {
    "llm_syn_clean": "LLM",
    "llm_syn_hybrid": "LLM Hybrid",
    "gan_syn_clean": "GAN",
    "gan_syn_hybrid": "GAN Hybrid",
}


def validate_columns(df: pd.DataFrame, name: str):
    missing = [c for c in DISTANCE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")


def min_distances(query_array, reference_array):
    out = []
    for row in query_array:
        dists = np.sqrt(((reference_array - row) ** 2).sum(axis=1))
        out.append(float(dists.min()))
    return out


def load_and_standardize(seed_path: Path, syn_path: Path, test_path: Path):
    seed = pd.read_csv(seed_path)
    syn = pd.read_csv(syn_path)
    test = pd.read_csv(test_path)

    validate_columns(seed, "seed data")
    validate_columns(syn, "synthetic data")
    validate_columns(test, "test data")

    means = seed[DISTANCE_COLS].mean()
    stds = seed[DISTANCE_COLS].std().replace(0, 1.0)

    seed_std = seed.copy()
    syn_std = syn.copy()
    test_std = test.copy()

    seed_std[DISTANCE_COLS] = (seed_std[DISTANCE_COLS] - means) / stds
    syn_std[DISTANCE_COLS] = (syn_std[DISTANCE_COLS] - means) / stds
    test_std[DISTANCE_COLS] = (test_std[DISTANCE_COLS] - means) / stds

    return seed_std, syn_std, test_std


def compute_dcr(seed_path: Path, syn_path: Path, test_path: Path, sample_seed=None):
    """
    Compute Distance to Closest Record (DCR) for a synthetic dataset relative
    to the seed dataset, using W1-W6, A, Y.

    Standardization uses seed stats. Synthetic rows are subsampled to match
    test set size. If sample_seed is provided, it controls this subsampling.
    """
    seed_std, syn_std, test_std = load_and_standardize(seed_path, syn_path, test_path)

    n_test = test_std.shape[0]
    if syn_std.shape[0] > n_test:
        syn_std = syn_std.sample(n=n_test, random_state=sample_seed)

    seed_array = seed_std[DISTANCE_COLS].to_numpy(dtype=float)
    syn_array = syn_std[DISTANCE_COLS].to_numpy(dtype=float)
    test_array = test_std[DISTANCE_COLS].to_numpy(dtype=float)

    syn_dcr = min_distances(syn_array, seed_array)
    test_dcr = min_distances(test_array, seed_array)

    return syn_dcr, test_dcr


def compute_dcr_repeated(seed_path: Path, syn_path: Path, test_path: Path):
    """
    Recompute synthetic DCR over multiple subsampling seeds.
    Returns:
      - all_dcr_values: flattened DCR values across repeated subsamples
      - mean_by_seed: mean DCR for each subsampling seed
      - first_values: DCR values from the first seed, useful for boxplot display
    """
    all_dcr_values = []
    mean_by_seed = []
    first_values = None

    for seed in SEEDS:
        syn_dcr, _ = compute_dcr(
            seed_path=seed_path,
            syn_path=syn_path,
            test_path=test_path,
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


def main():
    print(f"Using DATA_DIR    = {DATA_DIR}")
    print(f"Saving results to = {RESULTS_DIR}")
    print(f"Using columns     = {DISTANCE_COLS}")
    print(f"Using seeds       = {SEEDS}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not SEED_FILE.exists():
        raise FileNotFoundError(f"Missing seed file: {SEED_FILE}")
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Missing test file: {TEST_FILE}")

    results = {}
    aggregate = {}

    plot_data = []
    plot_labels = []

    first_syn = next(iter(SYN_FILES.values()))
    if not first_syn.exists():
        raise FileNotFoundError(f"Missing synthetic file: {first_syn}")

    # Test DCR is deterministic with respect to the seed data.
    _, test_dcr = compute_dcr(SEED_FILE, first_syn, TEST_FILE, sample_seed=SEEDS[0])
    test_summary = summarize(test_dcr)

    results["data_test"] = {
        "file": str(TEST_FILE),
        "distance_cols": DISTANCE_COLS,
        "summary": test_summary,
        "dcr_values": test_dcr,
    }
    aggregate["data_test"] = {
        "file": str(TEST_FILE),
        "distance_cols": DISTANCE_COLS,
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
        syn_path = SYN_FILES[name]
        if not syn_path.exists():
            raise FileNotFoundError(f"Missing synthetic file for {name}: {syn_path}")

        all_dcr_values, mean_by_seed, first_values = compute_dcr_repeated(
            SEED_FILE,
            syn_path,
            TEST_FILE,
        )

        record_summary = summarize(all_dcr_values)
        repeated_summary = summarize_repeated_means(mean_by_seed)

        results[name] = {
            "file": str(syn_path),
            "distance_cols": DISTANCE_COLS,
            "seeds": SEEDS,
            "summary": record_summary,
            "repeated_mean_summary": repeated_summary,
            "dcr_values_first_seed": first_values,
            "dcr_mean_by_seed": mean_by_seed,
        }

        aggregate[name] = {
            "file": str(syn_path),
            "distance_cols": DISTANCE_COLS,
            "summary": round_nested(record_summary),
            "repeated_mean_summary": round_nested(repeated_summary),
        }

        plot_data.append(first_values)
        plot_labels.append(PLOT_LABELS[name])

        print(
            f"{name}: Mean DCR = {repeated_summary['mean']:.4f} ± {repeated_summary['se']:.4f}, "
            f"Median DCR = {record_summary['median']:.4f}"
        )

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=4)

    print(f"DCR results saved to: {OUTPUT_JSON}")

    with open(OUTPUT_AGG_JSON, "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"Aggregate DCR results saved to: {OUTPUT_AGG_JSON}")

    plt.figure(figsize=(10, 6))
    plt.boxplot(plot_data, labels=plot_labels)
    plt.ylabel("Distance to Closest Record")
    plt.title("DCR Boxplot")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=200)
    plt.close()

    print(f"Boxplot saved to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()