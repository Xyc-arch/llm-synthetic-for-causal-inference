#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Folder containing this script, e.g. /home/ubuntu/syn_causal/privacy
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Input ACTG175 file
ACTG175_PATH = PROJECT_DIR / "actg175" / "actg175.csv"

# Output folder under this script folder: /home/ubuntu/syn_causal/privacy/data
DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / "actg175_clean.csv"

# Baseline covariates
W_VARS = [
    "age", "wtkg", "hemo", "homo", "drugs", "karnof",
    "oprior", "z30", "zprior", "preanti", "race", "gender",
    "str2", "strat", "symptom", "cd40", "cd80"
]

# Continuous covariates to normalize
CONT_VARS = ["age", "wtkg", "karnof", "preanti", "cd40", "cd80"]

# Outcome
Y_VAR = "cd420"


def main():
    if not ACTG175_PATH.exists():
        raise FileNotFoundError(f"ACTG175 file not found: {ACTG175_PATH}")

    df = pd.read_csv(ACTG175_PATH)

    print(f"Loaded ACTG175 from: {ACTG175_PATH}")
    print(f"Original shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Keep only arms 1 and 2
    if "arms" not in df.columns:
        raise ValueError("Column 'arms' not found in ACTG175 data.")

    data_sub = df[df["arms"].isin([1, 2])].copy()

    # Recode treatment:
    # A = 0 if arms == 1
    # A = 1 if arms == 2
    data_sub["A"] = (data_sub["arms"] == 2).astype(int)

    needed_cols = W_VARS + ["A", Y_VAR]
    missing_cols = [c for c in needed_cols if c not in data_sub.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Remove rows with missing data in W or outcome
    data_clean = data_sub[needed_cols].dropna().copy()

    print(f"After restricting to arms 1 and 2: {data_sub.shape}")
    print(f"After selecting columns and dropping missing rows: {data_clean.shape}")

    # Normalize continuous covariates
    scaler = StandardScaler()
    data_clean.loc[:, CONT_VARS] = scaler.fit_transform(data_clean[CONT_VARS])

    # Save cleaned dataset
    data_clean.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved cleaned ACTG175 data to: {OUTPUT_FILE}")

    print("\nTreatment counts:")
    print(data_clean["A"].value_counts(dropna=False).sort_index())

    print("\nOutcome summary:")
    print(data_clean[Y_VAR].describe())

    print("\nPreview:")
    print(data_clean.head().to_string(index=False))


if __name__ == "__main__":
    main()