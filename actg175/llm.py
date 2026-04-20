#!/usr/bin/env python3
import os
from pathlib import Path

import pandas as pd
from be_great import GReaT


def main():
    # Folder containing this script, e.g. /home/ubuntu/syn_causal/privacy
    script_dir = Path(__file__).resolve().parent

    # Paths
    input_csv = script_dir / "data" / "actg175_clean.csv"
    output_dir = script_dir / "llm_data"
    output_csv = output_dir / "syn_clean.csv"
    model_dir = output_dir / "great_checkpoint"

    os.makedirs(output_dir, exist_ok=True)

    # Load training data
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input file: {input_csv}")

    data = pd.read_csv(input_csv)

    # Keep only expected columns
    cols = [
        "age", "wtkg", "hemo", "homo", "drugs", "karnof",
        "oprior", "z30", "zprior", "preanti", "race", "gender",
        "str2", "strat", "symptom", "cd40", "cd80", "A", "cd420"
    ]
    data = data[cols].copy()

    # Binary / discrete-like columns
    discrete_cols = [
        "hemo", "homo", "drugs", "oprior", "z30", "zprior",
        "race", "gender", "str2", "strat", "symptom", "A"
    ]

    # Cast discrete columns explicitly
    for c in discrete_cols:
        data[c] = data[c].astype(int)

    print("Training rows:", len(data))
    print("Columns:", list(data.columns))

    # GReaT model
    # Use "distilgpt2" if GPU memory is tight
    model = GReaT(
        llm="gpt2",
        batch_size=32,
        epochs=50,
        fp16=True,
        dataloader_num_workers=4,
    )

    # Train
    model.fit(data)

    # Save checkpoint
    model.save(str(model_dir))
    print(f"Saved model to: {model_dir}")

    # Sample synthetic data
    synthetic_data = model.sample(
        n_samples=50000,
        random_feature_order=True,
        temperature=0.7,
        max_length=1024,
    )

    # Keep only expected columns if extras appear
    synthetic_data = synthetic_data[cols].copy()

    # Post-process discrete columns
    for c in discrete_cols:
        synthetic_data[c] = synthetic_data[c].round().astype(int)

    # Save
    synthetic_data.to_csv(output_csv, index=False)
    print(f"Saved synthetic data to: {output_csv}")
    print(synthetic_data.head())


if __name__ == "__main__":
    main()