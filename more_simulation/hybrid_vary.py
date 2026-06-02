#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def get_w_cols(columns):
    return sorted(
        [c for c in columns if c.startswith("W") and c[1:].isdigit()],
        key=lambda x: int(x[1:]),
    )


def validate_columns(df, needed, name):
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")


def sample_bernoulli(probs, rng):
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return rng.binomial(1, probs)


def hybrid_one(
    syn_clean_path,
    data_seed_path,
    out_path,
    treatment_mode="estimated",
    random_state=42,
):
    rng = np.random.default_rng(random_state)

    seed = pd.read_csv(data_seed_path)
    syn_raw = pd.read_csv(syn_clean_path)

    w_cols = get_w_cols(seed.columns)

    if not w_cols:
        raise ValueError(f"No W covariates found in {data_seed_path}")

    validate_columns(seed, w_cols + ["A", "Y"], "seed data")
    validate_columns(syn_raw, w_cols, "synthetic data")

    seed = seed[w_cols + ["A", "Y"]].copy()
    syn = syn_raw[w_cols].copy()

    for c in w_cols:
        seed[c] = pd.to_numeric(seed[c], errors="coerce")
        syn[c] = pd.to_numeric(syn[c], errors="coerce")

    seed["A"] = pd.to_numeric(seed["A"], errors="coerce").round().astype(int)
    seed["Y"] = pd.to_numeric(seed["Y"], errors="coerce").round().astype(int)

    if seed.isna().any().any():
        bad_cols = seed.columns[seed.isna().any()].tolist()
        raise ValueError(f"NaNs in seed data after numeric conversion: {bad_cols}")

    if syn.isna().any().any():
        bad_cols = syn.columns[syn.isna().any()].tolist()
        raise ValueError(f"NaNs in synthetic data after numeric conversion: {bad_cols}")

    # Treatment assignment
    if treatment_mode == "estimated":
        ps_model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )
        ps_model.fit(seed[w_cols], seed["A"])

        ps = ps_model.predict_proba(syn[w_cols])[:, 1]
        ps = np.clip(ps, 1e-6, 1.0 - 1e-6)

        syn["ps"] = ps
        syn["A"] = sample_bernoulli(ps, rng)

    elif treatment_mode == "randomized":
        syn["ps"] = 0.5
        syn["A"] = rng.binomial(1, 0.5, size=len(syn))

    else:
        raise ValueError(
            f"Unknown treatment_mode={treatment_mode}. "
            "Use 'estimated' or 'randomized'."
        )

    # Outcome model
    outcome_model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    outcome_model.fit(seed[w_cols + ["A"]], seed["Y"])

    y_prob = outcome_model.predict_proba(syn[w_cols + ["A"]])[:, 1]
    y_prob = np.clip(y_prob, 1e-6, 1.0 - 1e-6)

    syn["y_prob"] = y_prob
    syn["Y"] = sample_bernoulli(y_prob, rng)

    syn_hybrid = syn[w_cols + ["A", "Y"]].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    syn_hybrid.to_csv(out_path, index=False)

    diagnostics = {
        "n": int(len(syn_hybrid)),
        "d": int(len(w_cols)),
        "treatment_mode": treatment_mode,
        "mean_A": float(syn_hybrid["A"].mean()),
        "mean_Y": float(syn_hybrid["Y"].mean()),
        "ps_min": float(np.min(syn["ps"])),
        "ps_median": float(np.median(syn["ps"])),
        "ps_max": float(np.max(syn["ps"])),
        "y_prob_min": float(np.min(y_prob)),
        "y_prob_median": float(np.median(y_prob)),
        "y_prob_max": float(np.max(y_prob)),
        "w_cols": w_cols,
        "input": str(syn_clean_path),
        "output": str(out_path),
    }

    diag_path = out_path.parent / "hybrid_diagnostics.json"
    pd.Series(diagnostics).to_json(diag_path, indent=2)

    print(f"Saved hybrid data: {out_path}")
    print(f"Saved diagnostics: {diag_path}")
    print(diagnostics)

    return syn_hybrid, diagnostics


def run_setting(data_dir, treatment_mode="estimated", random_state=42):
    data_dir = Path(data_dir)
    data_seed_path = data_dir / "data_seed.csv"

    if not data_seed_path.exists():
        print(f"Skipping {data_dir}: missing data_seed.csv")
        return

    for gen in ["llm", "gan"]:
        syn_clean_path = data_dir / f"{gen}_data" / "syn_clean.csv"
        out_path = data_dir / f"{gen}_data" / "syn_hybrid.csv"

        if not syn_clean_path.exists():
            print(f"Skipping {gen} in {data_dir}: missing {syn_clean_path}")
            continue

        print("=" * 80)
        print(f"Hybrid generation | setting={data_dir.name} | generator={gen}")
        print("=" * 80)

        hybrid_one(
            syn_clean_path=syn_clean_path,
            data_seed_path=data_seed_path,
            out_path=out_path,
            treatment_mode=treatment_mode,
            random_state=random_state,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="simulator_vary_data",
        help="Root folder containing setting folders.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Run one setting folder only.",
    )
    parser.add_argument(
        "--treatment-mode",
        type=str,
        default="estimated",
        choices=["estimated", "randomized"],
        help="estimated uses RF propensity; randomized sets A~Bernoulli(0.5).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-if-output-exists", action="store_true")
    args = parser.parse_args()

    if args.data_dir is not None:
        run_setting(
            data_dir=args.data_dir,
            treatment_mode=args.treatment_mode,
            random_state=args.random_state,
        )
        return

    root = Path(args.root)

    if not root.exists():
        raise FileNotFoundError(f"Missing root folder: {root}")

    for data_dir in sorted(root.iterdir()):
        if not data_dir.is_dir():
            continue

        if not (data_dir / "data_seed.csv").exists():
            continue

        if args.skip_if_output_exists:
            llm_done = (data_dir / "llm_data" / "syn_hybrid.csv").exists()
            gan_done = (data_dir / "gan_data" / "syn_hybrid.csv").exists()
            if llm_done and gan_done:
                print(f"Skipping {data_dir}: both hybrid outputs already exist.")
                continue

        run_setting(
            data_dir=data_dir,
            treatment_mode=args.treatment_mode,
            random_state=args.random_state,
        )


if __name__ == "__main__":
    main()