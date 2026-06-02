#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

W_COLS = ["W1", "W2", "W3", "W4", "W5", "W6"]
X_COLS = W_COLS + ["A"]
Y_COL = "Y"

SEEDS = [1, 2, 3, 4, 5]
SUBSAMPLE_N = 1000

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "tstr.json"


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
TEST_FILE = DATA_DIR / "data_test.csv"

TRAINING_FILES = {
    "data_seed": DATA_DIR / "data_seed.csv",
    "gan_syn_hybrid": DATA_DIR / "gan_data" / "syn_hybrid.csv",
    "gan_syn_clean": DATA_DIR / "gan_data" / "syn_clean.csv",
    "llm_syn_hybrid": DATA_DIR / "llm_data" / "syn_hybrid.csv",
    "llm_syn_clean": DATA_DIR / "llm_data" / "syn_clean.csv",
}


def evaluate_model(training_file: Path, test_file: Path) -> dict:
    train_full = pd.read_csv(training_file)
    test = pd.read_csv(test_file)

    missing_train = [c for c in X_COLS + [Y_COL] if c not in train_full.columns]
    missing_test = [c for c in X_COLS + [Y_COL] if c not in test.columns]

    if missing_train:
        raise ValueError(f"Missing columns in training file {training_file}: {missing_train}")
    if missing_test:
        raise ValueError(f"Missing columns in test file {test_file}: {missing_test}")

    X_test = test[X_COLS]
    y_test = test[Y_COL]

    subsample_n = min(SUBSAMPLE_N, len(train_full))
    aucs = []

    for seed in SEEDS:
        train = train_full.sample(n=subsample_n, random_state=seed).copy()

        X_train = train[X_COLS]
        y_train = train[Y_COL]

        rf = RandomForestClassifier(random_state=seed)
        rf.fit(X_train, y_train)

        y_prob = rf.predict_proba(X_test)[:, 1]
        aucs.append(float(roc_auc_score(y_test, y_prob)))

    aucs = np.array(aucs, dtype=float)

    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs, ddof=1))
    se_auc = float(std_auc / np.sqrt(len(aucs)))

    return {
        "aucs": aucs.tolist(),
        "mean": mean_auc,
        "std": std_auc,
        "se": se_auc,
        "ci95_low": float(mean_auc - 1.96 * se_auc),
        "ci95_high": float(mean_auc + 1.96 * se_auc),
        "n_reps": len(SEEDS),
        "seeds": SEEDS,
        "subsample_n": int(subsample_n),
        # Keep this for backward compatibility with plotting code that expects "auc".
        "auc": mean_auc,
    }


def main():
    print(f"Using DATA_DIR    = {DATA_DIR}")
    print(f"Saving results to = {RESULTS_DIR}")
    print(f"Subsample_n       = {SUBSAMPLE_N}")
    print(f"Seeds             = {SEEDS}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Missing test file: {TEST_FILE}")

    auc_results = {}

    for name, file_path in TRAINING_FILES.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing training file for {name}: {file_path}")

        summary = evaluate_model(file_path, TEST_FILE)

        auc_results[name] = {
            "train_file": str(file_path),
            "test_file": str(TEST_FILE),
            **summary,
        }

        print(
            f"AUC on test set (trained on {name}): "
            f"{summary['mean']:.6f} ± {summary['se']:.6f}"
        )

    with open(OUTPUT_FILE, "w") as f:
        json.dump(auc_results, f, indent=4)

    print(f"AUC results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()