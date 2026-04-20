#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# This script lives in /home/ubuntu/syn_causal/actg175/
ACTG_DIR = Path(__file__).resolve().parent

RAW_FILE = ACTG_DIR / "actg175.csv"

LLM_DIR = ACTG_DIR / "llm_data"
CTGAN_DIR = ACTG_DIR / "ctgan_data"

LLM_INPUT = LLM_DIR / "syn_clean.csv"
CTGAN_INPUT = CTGAN_DIR / "syn_clean.csv"

LLM_OUTPUT = LLM_DIR / "syn_filter.csv"
CTGAN_OUTPUT = CTGAN_DIR / "syn_filter.csv"

W_VARS = [
    "age", "wtkg", "hemo", "homo", "drugs", "karnof",
    "oprior", "z30", "zprior", "preanti", "race", "gender",
    "str2", "strat", "symptom", "cd40", "cd80"
]
CONT_VARS = ["age", "wtkg", "karnof", "preanti", "cd40", "cd80"]
CAT_VARS = [
    "hemo", "homo", "drugs", "oprior", "z30", "zprior",
    "race", "gender", "str2", "strat", "symptom"
]


def load_seed_w() -> pd.DataFrame:
    """
    Load ACTG175, restrict to arms 1 and 2, build A like your R code,
    keep W columns, drop missing, and standardize continuous variables.
    """
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Missing ACTG file: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)

    if "arms" not in df.columns:
        raise ValueError("Column 'arms' not found in ACTG175 data.")

    # Same restriction as your R code
    df = df[df["arms"].isin([1, 2])].copy()

    needed = W_VARS + ["cd420"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw ACTG file: {missing}")

    df = df[needed].dropna().copy()

    # Make continuous columns float before scaling
    for c in CONT_VARS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    scaler = StandardScaler()
    df[CONT_VARS] = scaler.fit_transform(df[CONT_VARS])

    # Cast categorical columns to int
    for c in CAT_VARS:
        df[c] = pd.to_numeric(df[c], errors="coerce").round().astype(int)

    return df[W_VARS].copy()


def build_validity_reference(seed_w: pd.DataFrame):
    """
    Build observed support from real ACTG seed W.
    """
    cat_support = {
        c: sorted(seed_w[c].dropna().unique().tolist())
        for c in CAT_VARS
    }

    cont_bounds = {
        c: (
            float(seed_w[c].min()),
            float(seed_w[c].max())
        )
        for c in CONT_VARS
    }

    return cat_support, cont_bounds


def filter_synthetic_w(syn_df: pd.DataFrame, cat_support, cont_bounds):
    """
    Keep only rows with valid W:
      - all required columns present
      - finite numeric values
      - categorical values in observed support
      - continuous values within observed seed range
      - no missing values
    """
    missing = [c for c in W_VARS if c not in syn_df.columns]
    if missing:
        raise ValueError(f"Synthetic file missing required W columns: {missing}")

    df = syn_df[W_VARS].copy()

    # Coerce continuous columns
    for c in CONT_VARS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Coerce categorical columns to integers
    for c in CAT_VARS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].round()

    # Start validity mask
    valid = pd.Series(True, index=df.index)

    # No missing values
    valid &= ~df.isna().any(axis=1)

    # Continuous bounds check
    for c in CONT_VARS:
        lo, hi = cont_bounds[c]
        valid &= df[c].between(lo, hi, inclusive="both")

    # Categorical support check
    for c in CAT_VARS:
        valid &= df[c].isin(cat_support[c])

    filtered = df.loc[valid].copy()

    # Final categorical cast
    for c in CAT_VARS:
        filtered[c] = filtered[c].astype(int)

    return filtered, valid


def process_one(name: str, input_file: Path, output_file: Path, cat_support, cont_bounds):
    if not input_file.exists():
        raise FileNotFoundError(f"Missing synthetic input for {name}: {input_file}")

    syn_df = pd.read_csv(input_file)
    n_before = len(syn_df)

    filtered_df, valid_mask = filter_synthetic_w(syn_df, cat_support, cont_bounds)
    n_after = len(filtered_df)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_file, index=False)

    print(f"[{name}] input:  {input_file}")
    print(f"[{name}] output: {output_file}")
    print(f"[{name}] kept {n_after} / {n_before} rows ({100.0 * n_after / max(n_before, 1):.2f}%)")
    print(filtered_df.head().to_string(index=False))
    print()


def main():
    seed_w = load_seed_w()
    cat_support, cont_bounds = build_validity_reference(seed_w)

    print(f"Loaded ACTG seed support from: {RAW_FILE}")
    print(f"Seed W shape: {seed_w.shape}")
    print()

    process_one("llm", LLM_INPUT, LLM_OUTPUT, cat_support, cont_bounds)
    process_one("ctgan", CTGAN_INPUT, CTGAN_OUTPUT, cat_support, cont_bounds)


if __name__ == "__main__":
    main()