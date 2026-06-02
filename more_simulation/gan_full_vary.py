#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import pandas as pd
from ctgan import CTGAN


def get_w_cols(columns):
    return sorted(
        [c for c in columns if c.startswith("W") and c[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--sample-size", type=int, default=50000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    input_csv = data_dir / "data_seed.csv"
    output_dir = data_dir / "gan_data"
    output_csv = output_dir / "syn_full.csv"

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input file: {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_csv)

    w_cols = get_w_cols(data.columns)
    cols = w_cols + ["A", "Y"]

    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in {input_csv}: {missing}")

    data = data[cols].copy()

    # In this DGP, W1-W3 are binary. Extra W columns are continuous.
    discrete_columns = [c for c in ["W1", "W2", "W3", "A", "Y"] if c in data.columns]

    for c in discrete_columns:
        data[c] = pd.to_numeric(data[c], errors="coerce").round().astype(int)

    for c in cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    if data.isna().any().any():
        bad_cols = data.columns[data.isna().any()].tolist()
        raise ValueError(f"NaNs after numeric conversion in columns: {bad_cols}")

    print("=" * 80)
    print("CTGAN training")
    print("Data dir:", data_dir)
    print("Training rows:", len(data))
    print("Columns:", cols)
    print("Discrete columns:", discrete_columns)
    print("=" * 80)

    ctgan = CTGAN(
        epochs=args.epochs,
        verbose=True,
    )

    ctgan.fit(data, discrete_columns=discrete_columns)

    synthetic_data = ctgan.sample(args.sample_size)

    # Keep expected columns only and enforce order.
    synthetic_data = synthetic_data[cols].copy()

    for c in cols:
        synthetic_data[c] = pd.to_numeric(synthetic_data[c], errors="coerce")

    for c in discrete_columns:
        synthetic_data[c] = synthetic_data[c].round().astype("Int64")

    synthetic_data.to_csv(output_csv, index=False)

    print(f"Finished CTGAN. Saved to {output_csv}")
    print(synthetic_data.head())


if __name__ == "__main__":
    main()