#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
MORE_SIM_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = MORE_SIM_DIR.parent

if str(MORE_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(MORE_SIM_DIR))

from data_vary import generate_dataset


DEFAULT_SYN_ROOT = MORE_SIM_DIR / "simulator_vary_data"
DEFAULT_OUT_ROOT = SCRIPT_DIR / "real_data_vary"

N_REAL_DATASETS = 20
REAL_SAMPLE_SIZE = 200
N_TRUTH = 100000
SEEDS = list(range(1, N_REAL_DATASETS + 1))


def load_setting_truth(setting_dir: Path):
    truth_path = setting_dir / "truth.json"
    if not truth_path.exists():
        raise FileNotFoundError(f"Missing truth.json: {truth_path}")

    with open(truth_path, "r") as f:
        return json.load(f)


def save_without_probs(df, path: Path):
    drop_cols = [c for c in ["pA", "pY"] if c in df.columns]
    df.drop(columns=drop_cols).to_csv(path, index=False)


def summarize_assignment(df):
    out = {
        "n": int(len(df)),
        "treated": int((df["A"] == 1).sum()),
        "control": int((df["A"] == 0).sum()),
        "treat_rate": float(df["A"].mean()),
        "outcome_1": int((df["Y"] == 1).sum()),
        "outcome_0": int((df["Y"] == 0).sum()),
        "outcome_rate": float(df["Y"].mean()),
    }

    if "pA" in df.columns:
        out.update(
            {
                "min_pA": float(df["pA"].min()),
                "max_pA": float(df["pA"].max()),
                "pA_q05": float(np.quantile(df["pA"], 0.05)),
                "pA_median": float(np.median(df["pA"])),
                "pA_q95": float(np.quantile(df["pA"], 0.95)),
                "count_pA_lt_0.001": int((df["pA"] < 0.001).sum()),
                "count_pA_gt_0.999": int((df["pA"] > 0.999).sum()),
                "count_pA_lt_0.01": int((df["pA"] < 0.01).sum()),
                "count_pA_gt_0.99": int((df["pA"] > 0.99).sum()),
            }
        )

    return out


def generate_real_for_setting(setting_dir: Path, out_dir: Path):
    setting_truth = load_setting_truth(setting_dir)

    d = int(setting_truth["d"])
    overlap = setting_truth["overlap"]
    outcome_mode = setting_truth["outcome_mode"]

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"Generating real benchmark data for setting: {setting_dir.name}")
    print(f"Output: {out_dir}")
    print(f"d={d}, overlap={overlap}, outcome_mode={outcome_mode}")
    print("=" * 100)

    truth_df, ate_true, y1_truth, y0_truth, truth_diag = generate_dataset(
        n=N_TRUTH,
        seed=42,
        d=d,
        rct=True,
        truth=True,
        overlap=overlap,
        outcome_mode=outcome_mode,
        verbose=False,
    )

    save_without_probs(truth_df, out_dir / "data_truth.csv")

    truth = {
        "source_setting": setting_dir.name,
        "source_setting_dir": str(setting_dir),
        "seed": 42,
        "n_truth": int(N_TRUTH),
        "n_real_datasets": int(N_REAL_DATASETS),
        "real_sample_size": int(REAL_SAMPLE_SIZE),
        "rct": True,
        "d": int(d),
        "overlap": overlap,
        "outcome_mode": outcome_mode,
        "ate_true": float(ate_true),
        "y1_truth": float(y1_truth),
        "y0_truth": float(y0_truth),
        "truth_diagnostics": truth_diag,
    }

    with open(out_dir / "truth.json", "w") as f:
        json.dump(truth, f, indent=4)

    manifest = {
        "truth": truth,
        "datasets": {},
    }

    for i, seed in enumerate(SEEDS, start=1):
        df = generate_dataset(
            n=REAL_SAMPLE_SIZE,
            seed=seed,
            d=d,
            rct=True,
            truth=False,
            overlap=overlap,
            outcome_mode=outcome_mode,
            verbose=False,
        )

        out_path = out_dir / f"data_{i}.csv"
        save_without_probs(df, out_path)

        manifest["datasets"][f"data_{i}"] = {
            "seed": int(seed),
            "file": str(out_path),
            **summarize_assignment(df),
        }

        print(
            f"saved data_{i}.csv | seed={seed} | n={len(df)} | "
            f"A=1:{int((df['A'] == 1).sum())} | Y=1:{int((df['Y'] == 1).sum())}"
        )

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"Saved truth to: {out_dir / 'truth.json'}")
    print(f"Saved manifest to: {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(DEFAULT_SYN_ROOT))
    parser.add_argument("--out-root", type=str, default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--setting", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root)

    if not root.exists():
        raise FileNotFoundError(f"Missing root: {root}")

    if args.setting is not None:
        setting_dirs = [root / args.setting]
    else:
        setting_dirs = [
            p for p in sorted(root.iterdir())
            if p.is_dir() and (p / "truth.json").exists()
        ]

    if not setting_dirs:
        raise ValueError(f"No setting folders found under {root}")

    for setting_dir in setting_dirs:
        if not setting_dir.exists():
            raise FileNotFoundError(f"Missing setting dir: {setting_dir}")

        generate_real_for_setting(
            setting_dir=setting_dir,
            out_dir=out_root / setting_dir.name,
        )


if __name__ == "__main__":
    main()