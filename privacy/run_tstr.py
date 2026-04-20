#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

W_COLS = ["W1", "W2", "W3", "W4", "W5", "W6"]
X_COLS = W_COLS + ["A"]
Y_COL = "Y"

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


def evaluate_model(training_file: Path, test_file: Path) -> float:
    train = pd.read_csv(training_file)
    test = pd.read_csv(test_file)

    X_train = train[X_COLS]
    y_train = train[Y_COL]

    X_test = test[X_COLS]
    y_test = test[Y_COL]

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    y_prob = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    return float(auc)


def main():
    print(f"Using DATA_DIR   = {DATA_DIR}")
    print(f"Saving results to = {RESULTS_DIR}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Missing test file: {TEST_FILE}")

    auc_results = {}

    for name, file_path in TRAINING_FILES.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing training file for {name}: {file_path}")

        auc = evaluate_model(file_path, TEST_FILE)
        auc_results[name] = {
            "train_file": str(file_path),
            "test_file": str(TEST_FILE),
            "auc": auc,
        }
        print(f"AUC on test set (trained on {name}): {auc:.6f}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(auc_results, f, indent=4)

    print(f"AUC results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()