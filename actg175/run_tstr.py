#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ACTG_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_TEST_REAL = os.path.join(ACTG_DIR, "data", "actg175_clean.csv")

DATASETS = {
    "actg_original": TRAIN_TEST_REAL,
    "llm_syn_clean": os.path.join(ACTG_DIR, "llm_data", "syn_clean.csv"),
    "llm_syn_hybrid": os.path.join(ACTG_DIR, "llm_data", "syn_hybrid.csv"),
    "ctgan_syn_clean": os.path.join(ACTG_DIR, "ctgan_data", "syn_clean.csv"),
    "ctgan_syn_hybrid": os.path.join(ACTG_DIR, "ctgan_data", "syn_hybrid.csv"),
}

W_VARS = [
    "age", "wtkg", "hemo", "homo", "drugs", "karnof",
    "oprior", "z30", "zprior", "preanti", "race", "gender",
    "str2", "strat", "symptom", "cd40", "cd80"
]
TREATMENT_COL = "A"
OUTCOME_COL = "cd420"

# For TSTR on ACTG, make outcome binary for AUC:
# predict whether cd420 is above the real-data median.
CONT_VARS = ["age", "wtkg", "karnof", "preanti", "cd40", "cd80"]
CAT_VARS = [c for c in W_VARS if c not in CONT_VARS]


def load_real():
    if not os.path.exists(TRAIN_TEST_REAL):
        raise FileNotFoundError(f"Missing real ACTG cleaned file: {TRAIN_TEST_REAL}")
    df = pd.read_csv(TRAIN_TEST_REAL).copy()
    needed = W_VARS + [TREATMENT_COL, OUTCOME_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Real ACTG file missing required columns: {missing}")
    df = df[needed].dropna().copy()
    for c in CAT_VARS + [TREATMENT_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce").round().astype(int)
    for c in CONT_VARS + [OUTCOME_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df = df.dropna().reset_index(drop=True)
    return df


def sanitize_synthetic(df, ref_df):
    needed = W_VARS + [TREATMENT_COL, OUTCOME_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Synthetic dataset missing required columns: {missing}")

    df = df[needed].copy()

    for c in CONT_VARS + [OUTCOME_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    for c in CAT_VARS + [TREATMENT_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    # Snap categoricals to observed support
    for c in CAT_VARS + [TREATMENT_COL]:
        df[c] = df[c].round()
        observed = sorted(ref_df[c].dropna().unique().tolist())
        df[c] = df[c].apply(
            lambda x: np.nan if pd.isna(x) else min(observed, key=lambda z: abs(z - x))
        )

    # Clip continuous covariates
    for c in CONT_VARS:
        lo = float(ref_df[c].min())
        hi = float(ref_df[c].max())
        df[c] = df[c].clip(lo, hi)

    # Clip outcome to observed range
    y_lo = float(ref_df[OUTCOME_COL].min())
    y_hi = float(ref_df[OUTCOME_COL].max())
    df[OUTCOME_COL] = df[OUTCOME_COL].clip(y_lo, y_hi)

    df = df.dropna().copy()

    for c in CAT_VARS + [TREATMENT_COL]:
        df[c] = df[c].astype(int)

    return df.reset_index(drop=True)


def make_binary_outcome(train_df, test_df):
    """
    Convert continuous cd420 to a binary classification target using the
    real-data median threshold, so TSTR can be reported as AUC.
    """
    threshold = float(np.median(test_df[OUTCOME_COL]))
    y_train = (train_df[OUTCOME_COL] > threshold).astype(int).to_numpy()
    y_test = (test_df[OUTCOME_COL] > threshold).astype(int).to_numpy()
    return y_train, y_test, threshold


def build_classifier():
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_VARS + [TREATMENT_COL]),
            ("num", "passthrough", CONT_VARS),
        ]
    )
    clf = Pipeline(
        steps=[
            ("preprocess", pre),
            ("rf", RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                min_samples_leaf=5,
                n_jobs=-1
            )),
        ]
    )
    return clf


def auc_score_manual(y_true, y_prob):
    """
    Lightweight AUC without importing sklearn.metrics, to keep style simple.
    """
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_prob))


def evaluate_tstr(train_df, test_df):
    y_train, y_test, threshold = make_binary_outcome(train_df, test_df)

    X_train = train_df[W_VARS + [TREATMENT_COL]].copy()
    X_test = test_df[W_VARS + [TREATMENT_COL]].copy()

    model = build_classifier()
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = auc_score_manual(y_test, y_prob)

    return {
        "auc": auc,
        "threshold_cd420": threshold,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "positive_rate_train": float(np.mean(y_train)),
        "positive_rate_test": float(np.mean(y_test)),
    }


def main():
    real_df = load_real()

    results = {}

    for dataset_name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"Skipping missing dataset: {dataset_name} -> {path}")
            continue

        train_df = pd.read_csv(path)

        if dataset_name != "actg_original":
            train_df = sanitize_synthetic(train_df, real_df)
        else:
            train_df = real_df.copy()

        res = evaluate_tstr(train_df, real_df)
        results[dataset_name] = {
            "train_file": path,
            "test_file": TRAIN_TEST_REAL,
            **res,
        }

        print(
            f"{dataset_name}: "
            f"AUC={res['auc']:.6f}, "
            f"n_train={res['n_train']}, "
            f"n_test={res['n_test']}"
        )

    results_dir = os.path.join(ACTG_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "tstr.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved TSTR results to {out_path}")


if __name__ == "__main__":
    main()