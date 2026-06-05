#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error


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
    "str2", "strat", "symptom", "cd40", "cd80",
]

TREATMENT_COL = "A"
OUTCOME_COL = "cd420"

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

    df = df[needed].copy()

    for c in CAT_VARS + [TREATMENT_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")

    for c in CONT_VARS + [OUTCOME_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    df = df.dropna().copy()

    for c in CAT_VARS + [TREATMENT_COL]:
        df[c] = df[c].astype(int)

    return df.reset_index(drop=True)


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

    # Snap categorical variables to observed real-data support.
    for c in CAT_VARS + [TREATMENT_COL]:
        df[c] = df[c].round()
        observed = sorted(ref_df[c].dropna().unique().tolist())
        df[c] = df[c].apply(
            lambda x: np.nan if pd.isna(x) else min(observed, key=lambda z: abs(z - x))
        )

    # Clip continuous covariates to observed real-data ranges.
    for c in CONT_VARS:
        lo = float(ref_df[c].min())
        hi = float(ref_df[c].max())
        df[c] = df[c].clip(lo, hi)

    # Clip continuous outcome to observed real-data range.
    y_lo = float(ref_df[OUTCOME_COL].min())
    y_hi = float(ref_df[OUTCOME_COL].max())
    df[OUTCOME_COL] = df[OUTCOME_COL].clip(y_lo, y_hi)

    df = df.dropna().copy()

    for c in CAT_VARS + [TREATMENT_COL]:
        df[c] = df[c].astype(int)

    return df.reset_index(drop=True)


def build_regressor():
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_VARS + [TREATMENT_COL]),
            ("num", "passthrough", CONT_VARS),
        ]
    )

    reg = Pipeline(
        steps=[
            ("preprocess", pre),
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=500,
                    random_state=42,
                    min_samples_leaf=5,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return reg


def evaluate_tstr_mse(train_df, test_df):
    X_train = train_df[W_VARS + [TREATMENT_COL]].copy()
    y_train = train_df[OUTCOME_COL].astype(float).to_numpy()

    X_test = test_df[W_VARS + [TREATMENT_COL]].copy()
    y_test = test_df[OUTCOME_COL].astype(float).to_numpy()

    model = build_regressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_test - y_pred)))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "y_train_mean": float(np.mean(y_train)),
        "y_test_mean": float(np.mean(y_test)),
        "y_pred_mean": float(np.mean(y_pred)),
        "y_train_std": float(np.std(y_train, ddof=1)),
        "y_test_std": float(np.std(y_test, ddof=1)),
        "y_pred_std": float(np.std(y_pred, ddof=1)),
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

        res = evaluate_tstr_mse(train_df, real_df)

        results[dataset_name] = {
            "train_file": path,
            "test_file": TRAIN_TEST_REAL,
            "target": OUTCOME_COL,
            "metric": "continuous_tstr_mse_rmse",
            **res,
        }

        print(
            f"{dataset_name}: "
            f"MSE={res['mse']:.6f}, "
            f"RMSE={res['rmse']:.6f}, "
            f"MAE={res['mae']:.6f}, "
            f"n_train={res['n_train']}, "
            f"n_test={res['n_test']}"
        )

    results_dir = os.path.join(ACTG_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    out_path = os.path.join(results_dir, "tstr_mse.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved TSTR MSE results to {out_path}")


if __name__ == "__main__":
    main()