#!/usr/bin/env python3
import os
from pathlib import Path

import pandas as pd
from ctgan import CTGAN

# Folder containing this script, e.g. /home/ubuntu/syn_causal/privacy
SCRIPT_DIR = Path(__file__).resolve().parent

# Input cleaned ACTG175 data
input_path = SCRIPT_DIR / "data" / "actg175_clean.csv"

# Output directory
out_dir = SCRIPT_DIR / "ctgan_data"
os.makedirs(out_dir, exist_ok=True)

# Output file
output_path = out_dir / "syn_clean.csv"

# Expected columns
cols = [
    "age", "wtkg", "hemo", "homo", "drugs", "karnof",
    "oprior", "z30", "zprior", "preanti", "race", "gender",
    "str2", "strat", "symptom", "cd40", "cd80", "A", "cd420"
]

# Discrete / categorical columns
discrete_columns = [
    "hemo", "homo", "drugs", "oprior", "z30", "zprior",
    "race", "gender", "str2", "strat", "symptom", "A"
]


def main():
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    # Load cleaned data
    data = pd.read_csv(input_path)

    # Keep expected columns only
    data = data[cols].copy()

    # Make sure discrete columns are integers
    for c in discrete_columns:
        data[c] = data[c].round().astype(int)

    print(f"Loaded data from: {input_path}")
    print(f"Training shape: {data.shape}")

    # Train CTGAN
    ctgan = CTGAN(
        epochs=50,
        verbose=True,
    )

    ctgan.fit(data, discrete_columns=discrete_columns)

    # Sample synthetic data
    sample_size = 50000
    synthetic_data = ctgan.sample(sample_size)

    # Post-process discrete columns
    for c in discrete_columns:
        synthetic_data[c] = synthetic_data[c].round().astype(int)

    # Save
    synthetic_data.to_csv(output_path, index=False)

    print(f"Finished. Saved to {output_path}")
    print(synthetic_data.head())


if __name__ == "__main__":
    main()