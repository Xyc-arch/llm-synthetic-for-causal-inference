#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


Y_COL = "Y"
A_COL = "A"

SEEDS = [1, 2, 3, 4, 5]
SUBSAMPLE_N = 1000

DATASET_FILES = {
    "data_seed": "data_seed.csv",
    "gan_syn_hybrid": "gan_data/syn_hybrid.csv",
    "gan_syn_clean": "gan_data/syn_clean.csv",
    "llm_syn_hybrid": "llm_data/syn_hybrid.csv",
    "llm_syn_clean": "llm_data/syn_clean.csv",
}


def get_w_cols(columns):
    return sorted(
        [c for c in columns if c.startswith("W") and c[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )


def summarize(values):
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    se = float(std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0

    return {
        "values": arr.tolist(),
        "mean": mean,
        "std": std,
        "se": se,
        "ci95_low": float(mean - 1.96 * se),
        "ci95_high": float(mean + 1.96 * se),
        "n_reps": int(len(arr)),
        "seeds": SEEDS,
        "auc": mean,
    }


def evaluate_model(training_file: Path, test_file: Path, w_cols):
    train_full = pd.read_csv(training_file)
    test = pd.read_csv(test_file)

    x_cols = w_cols + [A_COL]
    needed = x_cols + [Y_COL]

    missing_train = [c for c in needed if c not in train_full.columns]
    missing_test = [c for c in needed if c not in test.columns]

    if missing_train:
        raise ValueError(f"Missing columns in training file {training_file}: {missing_train}")
    if missing_test:
        raise ValueError(f"Missing columns in test file {test_file}: {missing_test}")

    train_full = train_full[needed].copy()
    test = test[needed].copy()

    for c in x_cols:
        train_full[c] = pd.to_numeric(train_full[c], errors="coerce")
        test[c] = pd.to_numeric(test[c], errors="coerce")

    train_full[Y_COL] = pd.to_numeric(train_full[Y_COL], errors="coerce").round().astype("Int64")
    test[Y_COL] = pd.to_numeric(test[Y_COL], errors="coerce").round().astype("Int64")

    train_full = train_full.dropna().copy()
    test = test.dropna().copy()

    if len(train_full) == 0:
        raise ValueError(f"No valid training rows after cleaning: {training_file}")
    if len(test) == 0:
        raise ValueError(f"No valid test rows after cleaning: {test_file}")

    X_test = test[x_cols]
    y_test = test[Y_COL].astype(int)

    subsample_n = min(SUBSAMPLE_N, len(train_full))
    aucs = []

    for seed in SEEDS:
        train = train_full.sample(n=subsample_n, random_state=seed).copy()

        X_train = train[x_cols]
        y_train = train[Y_COL].astype(int)

        rf = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        y_prob = rf.predict_proba(X_test)[:, 1]
        aucs.append(float(roc_auc_score(y_test, y_prob)))

    summary = summarize(aucs)
    summary.update(
        {
            "train_file": str(training_file),
            "test_file": str(test_file),
            "subsample_n": int(subsample_n),
            "n_train_full": int(len(train_full)),
            "n_test": int(len(test)),
            "w_cols": w_cols,
            "x_cols": x_cols,
        }
    )

    return summary


def evaluate_setting(setting_dir: Path):
    test_file = setting_dir / "data_test.csv"
    seed_file = setting_dir / "data_seed.csv"

    if not test_file.exists():
        raise FileNotFoundError(f"Missing test file: {test_file}")
    if not seed_file.exists():
        raise FileNotFoundError(f"Missing seed file: {seed_file}")

    seed_head = pd.read_csv(seed_file, nrows=5)
    w_cols = get_w_cols(seed_head.columns)

    if not w_cols:
        raise ValueError(f"No W columns found in {seed_file}")

    print(f"Setting      : {setting_dir.name}")
    print(f"Using W cols : {w_cols}")
    print(f"Subsample_n  : {SUBSAMPLE_N}")
    print(f"Seeds        : {SEEDS}")

    results = {
        "setting": setting_dir.name,
        "setting_dir": str(setting_dir),
        "w_cols": w_cols,
        "x_cols": w_cols + [A_COL],
        "subsample_n": SUBSAMPLE_N,
        "seeds": SEEDS,
        "datasets": {},
    }

    for name, rel_path in DATASET_FILES.items():
        training_file = setting_dir / rel_path

        if not training_file.exists():
            print(f"Skipping missing training file for {name}: {training_file}")
            continue

        summary = evaluate_model(training_file, test_file, w_cols)
        results["datasets"][name] = summary

        print(
            f"AUC on test set trained on {name}: "
            f"{summary['mean']:.6f} ± {summary['se']:.6f}"
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="simulator_vary_data")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="vary_results")
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
        "subsample_n": SUBSAMPLE_N,
        "seeds": SEEDS,
        "settings": {},
    }

    for setting_dir in setting_dirs:
        print("=" * 100)
        print(f"Running TSTR for setting: {setting_dir}")
        print("=" * 100)

        setting_results = evaluate_setting(setting_dir)
        all_results["settings"][setting_dir.name] = setting_results

        per_setting_out = output_dir / f"{setting_dir.name}_tstr.json"
        with open(per_setting_out, "w") as f:
            json.dump(setting_results, f, indent=2)

        print(f"Saved per-setting TSTR to {per_setting_out}")

    full_out = output_dir / "tstr.json"
    with open(full_out, "w") as f:
        json.dump(all_results, f, indent=2)

    compact = {
        "subsample_n": SUBSAMPLE_N,
        "seeds": SEEDS,
        "settings": {},
    }

    for setting_name, setting_results in all_results["settings"].items():
        compact["settings"][setting_name] = {}

        for ds, metrics in setting_results["datasets"].items():
            compact["settings"][setting_name][ds] = {
                "mean": round(metrics["mean"], 6),
                "se": round(metrics["se"], 6),
                "ci95_low": round(metrics["ci95_low"], 6),
                "ci95_high": round(metrics["ci95_high"], 6),
                "subsample_n": metrics["subsample_n"],
                "n_train_full": metrics["n_train_full"],
                "n_test": metrics["n_test"],
            }

    compact_out = output_dir / "tstr_compact.json"
    with open(compact_out, "w") as f:
        json.dump(compact, f, indent=2)

    print(f"Saved full TSTR results to {full_out}")
    print(f"Saved compact TSTR results to {compact_out}")


if __name__ == "__main__":
    main()