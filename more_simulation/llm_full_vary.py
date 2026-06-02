#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import torch
from be_great import GReaT


def get_w_cols(columns):
    return sorted(
        [c for c in columns if c.startswith("W") and c[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )


def move_great_to_device(model):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Using device for sampling: {device}")

    if hasattr(model, "model"):
        model.model.to(device)
        model.model.eval()

        try:
            first_param_device = next(model.model.parameters()).device
            print(f"GReaT underlying model device: {first_param_device}")
        except StopIteration:
            print("Warning: underlying model has no parameters.")
    else:
        print("Warning: GReaT object has no `.model` attribute; could not move model manually.")

    return model, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--llm", type=str, default="gpt2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)

    # Original-style sampling defaults, but smaller sample-size default for stress tests.
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--sample-batch-size", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=1024)

    # Safety controls
    parser.add_argument("--max-d", type=int, default=20)
    parser.add_argument("--skip-if-output-exists", action="store_true")
    parser.add_argument("--use-existing-checkpoint", action="store_true")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    input_csv = data_dir / "data_seed.csv"
    output_dir = data_dir / "llm_data"
    output_csv = output_dir / "syn_full.csv"
    model_dir = output_dir / "great_checkpoint"

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input file: {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_if_output_exists and output_csv.exists() and output_csv.stat().st_size > 0:
        print(f"Output already exists, skipping: {output_csv}")
        return

    data = pd.read_csv(input_csv)

    w_cols = get_w_cols(data.columns)
    d = len(w_cols)

    if d > args.max_d:
        print(f"Skipping {data_dir}: d={d} exceeds max_d={args.max_d}")
        return

    cols = w_cols + ["A", "Y"]

    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in {input_csv}: {missing}")

    data = data[cols].copy()

    binary_cols = [c for c in ["W1", "W2", "W3", "A", "Y"] if c in data.columns]

    for c in cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    for c in binary_cols:
        data[c] = data[c].round().astype(int)

    if data.isna().any().any():
        bad_cols = data.columns[data.isna().any()].tolist()
        raise ValueError(f"NaNs after numeric conversion in columns: {bad_cols}")

    print("=" * 80)
    print("GReaT")
    print("Data dir:", data_dir)
    print("Training rows:", len(data))
    print("d:", d)
    print("Columns:", cols)
    print("Binary columns:", binary_cols)
    print("Output:", output_csv)
    print("=" * 80)

    if args.use_existing_checkpoint and model_dir.exists():
        print(f"Loading existing checkpoint from: {model_dir}")
        model = GReaT.load_from_dir(str(model_dir))
    else:
        print("Training new GReaT model...")
        model = GReaT(
            llm=args.llm,
            batch_size=args.batch_size,
            epochs=args.epochs,
            fp16=True,
            dataloader_num_workers=4,
        )

        model.fit(data)

        model.save(str(model_dir))
        print(f"Saved model to: {model_dir}")

    model, device = move_great_to_device(model)

    print(f"Sampling {args.sample_size} rows...")
    print(f"Sampling batch size k={args.sample_batch_size}")
    print(f"Sampling max_length={args.max_length}")
    print(f"Sampling temperature={args.temperature}")

    synthetic_data = model.sample(
        n_samples=args.sample_size,
        k=args.sample_batch_size,
        random_feature_order=True,
        temperature=args.temperature,
        max_length=args.max_length,
        device=device,
    )

    synthetic_data = synthetic_data[cols].copy()

    for c in cols:
        synthetic_data[c] = pd.to_numeric(synthetic_data[c], errors="coerce")

    for c in binary_cols:
        synthetic_data[c] = synthetic_data[c].round().astype("Int64")

    synthetic_data.to_csv(output_csv, index=False)

    print(f"Saved LLM synthetic data to: {output_csv}")
    print(synthetic_data.head())


if __name__ == "__main__":
    main()