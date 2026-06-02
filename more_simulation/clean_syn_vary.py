#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

LARGE_THRESHOLD = 1e6


def get_w_cols(columns):
    return sorted(
        [c for c in columns if c.startswith("W") and c[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )


def infer_expected_cols(data_dir: Path):
    seed_path = data_dir / "data_seed.csv"
    if not seed_path.exists():
        raise FileNotFoundError(f"Missing seed file: {seed_path}")

    seed = pd.read_csv(seed_path, nrows=5)
    w_cols = get_w_cols(seed.columns)
    return w_cols + ["A", "Y"]


def clean_one(label, path, all_cols, bin_cols):
    print("\n" + "=" * 100)
    print(f"{label.upper()} | reading {path}")

    if not path.exists():
        print("File not found.")
        return

    df = pd.read_csv(path)
    print(f"Original shape: {df.shape}")

    missing_cols = [c for c in all_cols if c not in df.columns]
    if missing_cols:
        print("Missing columns:", missing_cols)
        return

    num = df[all_cols].copy()

    for c in all_cols:
        num[c] = pd.to_numeric(num[c], errors="coerce")

    bad_mask = pd.Series(False, index=df.index)

    for c in all_cols:
        bad_mask |= num[c].isna()
        bad_mask |= np.isinf(num[c])
        bad_mask |= ((num[c].abs() > LARGE_THRESHOLD) & (~np.isinf(num[c])))

    for c in bin_cols:
        if c in num.columns:
            bad_mask |= ~num[c].round().isin([0, 1])

    bad_rows = df.loc[bad_mask].copy()

    print(f"Bad row count: {len(bad_rows)}")

    if len(bad_rows) > 0 and len(bad_rows) <= 20:
        print("\nBad rows:")
        print(bad_rows.to_string())
    elif len(bad_rows) > 20:
        print("Too many bad rows to print fully. Showing first 20:")
        print(bad_rows.head(20).to_string())

    clean = num.loc[~bad_mask, all_cols].copy()

    for c in bin_cols:
        if c in clean.columns:
            clean[c] = clean[c].round().astype(int)

    out_path = path.parent / "syn_clean.csv"
    clean.to_csv(out_path, index=False)

    print(f"Clean shape: {clean.shape}")
    print(f"Saved cleaned file to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    all_cols = infer_expected_cols(data_dir)
    bin_cols = [c for c in ["W1", "W2", "W3", "A", "Y"] if c in all_cols]

    files = {
        "llm": data_dir / "llm_data" / "syn_full.csv",
        "gan": data_dir / "gan_data" / "syn_full.csv",
    }

    print("Data dir:", data_dir)
    print("All cols:", all_cols)
    print("Binary cols:", bin_cols)

    for label, path in files.items():
        clean_one(label, path, all_cols, bin_cols)


if __name__ == "__main__":
    main()