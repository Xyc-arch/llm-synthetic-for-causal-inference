#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ACTG_DIR = os.path.dirname(os.path.abspath(__file__))

REAL_FILE = os.path.join(ACTG_DIR, "data", "actg175_clean.csv")

SYN_FILES = {
    "llm_syn_clean": os.path.join(ACTG_DIR, "llm_data", "syn_clean.csv"),
    "llm_syn_hybrid": os.path.join(ACTG_DIR, "llm_data", "syn_hybrid.csv"),
    "ctgan_syn_clean": os.path.join(ACTG_DIR, "ctgan_data", "syn_clean.csv"),
    "ctgan_syn_hybrid": os.path.join(ACTG_DIR, "ctgan_data", "syn_hybrid.csv"),
}

W_COLS = [
    "age", "wtkg", "hemo", "homo", "drugs", "karnof",
    "oprior", "z30", "zprior", "preanti", "race", "gender",
    "str2", "strat", "symptom", "cd40", "cd80"
]

CONT_VARS = ["age", "wtkg", "karnof", "preanti", "cd40", "cd80"]
CAT_VARS = [c for c in W_COLS if c not in CONT_VARS]


def load_real():
    if not os.path.exists(REAL_FILE):
        raise FileNotFoundError(f"Missing real ACTG cleaned file: {REAL_FILE}")

    df = pd.read_csv(REAL_FILE).copy()
    missing = [c for c in W_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Real ACTG file missing W columns: {missing}")

    df = df[W_COLS].dropna().copy()

    for c in CONT_VARS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    for c in CAT_VARS:
        df[c] = pd.to_numeric(df[c], errors="coerce").round().astype(int)

    df = df.dropna().reset_index(drop=True)
    return df


def sanitize_synthetic_w(df, ref_df):
    missing = [c for c in W_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Synthetic dataset missing W columns: {missing}")

    df = df[W_COLS].copy()

    for c in CONT_VARS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    for c in CAT_VARS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    for c in CAT_VARS:
        df[c] = df[c].round()
        observed = sorted(ref_df[c].dropna().unique().tolist())
        df[c] = df[c].apply(
            lambda x: np.nan if pd.isna(x) else min(observed, key=lambda z: abs(z - x))
        )

    for c in CONT_VARS:
        lo = float(ref_df[c].min())
        hi = float(ref_df[c].max())
        df[c] = df[c].clip(lo, hi)

    df = df.dropna().copy()

    for c in CAT_VARS:
        df[c] = df[c].astype(int)

    return df.reset_index(drop=True)


def compute_dcr(ref_df, syn_df):
    ref = ref_df.copy()
    syn = syn_df.copy()

    means = ref[CONT_VARS].mean()
    stds = ref[CONT_VARS].std().replace(0, 1.0)

    ref[CONT_VARS] = (ref[CONT_VARS] - means) / stds
    syn[CONT_VARS] = (syn[CONT_VARS] - means) / stds

    ref_array = ref[W_COLS].to_numpy(dtype=float)
    syn_array = syn[W_COLS].to_numpy(dtype=float)

    dcr_values = []
    for row in syn_array:
        dists = np.sqrt(((ref_array - row) ** 2).sum(axis=1))
        dcr_values.append(float(dists.min()))

    return dcr_values


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


def main():
    ref_df = load_real()

    full_results = {}
    summary_results = {}
    plot_data = []
    plot_labels = []

    for name, path in SYN_FILES.items():
        if not os.path.exists(path):
            print(f"Skipping missing dataset: {name} -> {path}")
            continue

        syn_df = pd.read_csv(path)
        syn_df = sanitize_synthetic_w(syn_df, ref_df)

        dcr_vals = compute_dcr(ref_df, syn_df)
        stats = summarize(dcr_vals)

        full_results[name] = {
            "file": path,
            "summary": stats,
            "dcr_values": dcr_vals,
        }

        summary_results[name] = {
            "file": path,
            **stats,
        }

        plot_data.append(dcr_vals)
        plot_labels.append(name)

        print(
            f"{name}: "
            f"Mean DCR = {stats['mean']:.6f}, "
            f"Std DCR = {stats['std']:.6f}, "
            f"Median DCR = {stats['median']:.6f}, "
            f"n = {stats['n']}"
        )

    results_dir = os.path.join(ACTG_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    json_out = os.path.join(results_dir, "dcr.json")
    with open(json_out, "w") as f:
        json.dump(full_results, f, indent=4)
    print(f"Saved full DCR results to {json_out}")

    summary_json_out = os.path.join(results_dir, "dcr_summary.json")
    with open(summary_json_out, "w") as f:
        json.dump(summary_results, f, indent=4)
    print(f"Saved aggregated DCR summary to {summary_json_out}")

    if plot_data:
        plt.figure(figsize=(10, 6))
        plt.boxplot(plot_data, labels=plot_labels)
        plt.ylabel("Distance to Closest Real Record")
        plt.title("DCR for Synthetic ACTG175 Datasets")
        plt.xticks(rotation=20)
        plt.tight_layout()

        plot_out = os.path.join(results_dir, "dcr_boxplot.png")
        plt.savefig(plot_out, dpi=200)
        plt.close()

        print(f"Saved boxplot to {plot_out}")


if __name__ == "__main__":
    main()