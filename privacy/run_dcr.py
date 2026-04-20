#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DISTANCE_COLS = ["W1", "W2", "W3", "W4", "W5", "W6", "A", "Y"]

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


def compute_dcr(seed_path: Path, syn_path: Path, test_path: Path):
    """
    Compute Distance to Closest Record (DCR) for a synthetic dataset relative
    to the seed dataset, using the full record:
        W1-W6, A, Y

    Standardization uses seed stats. Synthetic rows are subsampled to match
    test set size.
    """
    seed = pd.read_csv(seed_path)
    syn = pd.read_csv(syn_path)
    test = pd.read_csv(test_path)

    missing_seed = [c for c in DISTANCE_COLS if c not in seed.columns]
    missing_syn = [c for c in DISTANCE_COLS if c not in syn.columns]
    missing_test = [c for c in DISTANCE_COLS if c not in test.columns]

    if missing_seed:
        raise ValueError(f"Missing columns in seed data: {missing_seed}")
    if missing_syn:
        raise ValueError(f"Missing columns in synthetic data: {missing_syn}")
    if missing_test:
        raise ValueError(f"Missing columns in test data: {missing_test}")

    means = seed[DISTANCE_COLS].mean()
    stds = seed[DISTANCE_COLS].std().replace(0, 1.0)

    seed_std = seed.copy()
    syn_std = syn.copy()
    test_std = test.copy()

    seed_std[DISTANCE_COLS] = (seed_std[DISTANCE_COLS] - means) / stds
    syn_std[DISTANCE_COLS] = (syn_std[DISTANCE_COLS] - means) / stds
    test_std[DISTANCE_COLS] = (test_std[DISTANCE_COLS] - means) / stds

    n_test = test_std.shape[0]
    if syn_std.shape[0] > n_test:
        syn_std = syn_std.sample(n=n_test, random_state=42)

    seed_array = seed_std[DISTANCE_COLS].to_numpy(dtype=float)
    syn_array = syn_std[DISTANCE_COLS].to_numpy(dtype=float)
    test_array = test_std[DISTANCE_COLS].to_numpy(dtype=float)

    def min_distances(query_array, reference_array):
        out = []
        for row in query_array:
            dists = np.sqrt(((reference_array - row) ** 2).sum(axis=1))
            out.append(float(dists.min()))
        return out

    syn_dcr = min_distances(syn_array, seed_array)
    test_dcr = min_distances(test_array, seed_array)

    return syn_dcr, test_dcr


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(len(arr)),
    }


def round_summary(summary, digits=6):
    return {
        "mean": round(summary["mean"], digits),
        "std": round(summary["std"], digits),
        "median": round(summary["median"], digits),
        "min": round(summary["min"], digits),
        "max": round(summary["max"], digits),
        "n": summary["n"],
    }


def main():
    print(f"Using DATA_DIR    = {DATA_DIR}")
    print(f"Saving results to = {RESULTS_DIR}")
    print(f"Using columns     = {DISTANCE_COLS}")

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

    _, test_dcr = compute_dcr(SEED_FILE, first_syn, TEST_FILE)
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
        "summary": round_summary(test_summary),
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

        syn_dcr, _ = compute_dcr(SEED_FILE, syn_path, TEST_FILE)
        syn_summary = summarize(syn_dcr)

        results[name] = {
            "file": str(syn_path),
            "distance_cols": DISTANCE_COLS,
            "summary": syn_summary,
            "dcr_values": syn_dcr,
        }
        aggregate[name] = {
            "file": str(syn_path),
            "distance_cols": DISTANCE_COLS,
            "summary": round_summary(syn_summary),
        }

        plot_data.append(syn_dcr)
        plot_labels.append(PLOT_LABELS[name])

        print(
            f"{name}: Mean DCR = {syn_summary['mean']:.4f}, "
            f"Std DCR = {syn_summary['std']:.4f}, "
            f"Median DCR = {syn_summary['median']:.4f}"
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