import os
import json
import numpy as np
import pandas as pd
from scipy.special import expit

OUT_DIR = "/home/ubuntu/syn_causal/simulator/real_data_n1000_rct"
TRUTH_PATH = os.path.join(OUT_DIR, "truth.json")

N_DATASETS = 20
N_PER_DATASET = 1000
N_TRUTH = 100000
SEEDS = list(range(1, N_DATASETS + 1))


def generate_dataset(n, seed=42, rct=True, truth=False):
    rng = np.random.default_rng(seed)

    W1 = rng.binomial(1, 0.5, n)
    W2 = rng.binomial(1, 0.5, n)

    pW3 = 0.3 + 0.35 * ((W1 + W2) / 2.0)
    W3 = rng.binomial(1, pW3, n)

    W4 = rng.normal(0.0, 1.0, n)
    W5 = rng.normal(0.0, 1.0, n)
    W6 = 0.5 * W4 + 0.5 * W5 + rng.normal(0.0, 1.0, n)

    # Observational propensity, saved for reference even in RCT mode
    logits_a = -30 + 16 * W1 - 24 * W2 + 12 * W3 + 6 * W4 - 10 * W5 + 16 * W6
    pA_obs = expit(logits_a)

    if rct:
        A = rng.binomial(1, 0.5, n)
        pA = np.full(n, 0.5, dtype=float)
    else:
        A = rng.binomial(1, pA_obs, n)
        pA = pA_obs

    tau = (
        2.0
        + 0.5 * np.sin(W1)
        + 0.3 * np.log(np.abs(W2) + 1.0)
        - 0.2 * (W3 ** 2)
        + 0.1 * np.exp(W4)
        - 0.3 * np.tanh(W5)
        + 0.2 * np.cos(W6)
    )

    logits_y = (
        -0.5
        + tau * A
        + 0.5 * W1
        + 1.0 * W2
        - 1.0 * W3
        + 0.2 * W4
        - 0.3 * W5
        + 0.1 * W6
    )
    pY = expit(logits_y)
    Y = rng.binomial(1, pY, n)

    logits_y1 = (
        -0.5
        + tau
        + 0.5 * W1
        + 1.0 * W2
        - 1.0 * W3
        + 0.2 * W4
        - 0.3 * W5
        + 0.1 * W6
    )
    logits_y0 = (
        -0.5
        + 0.5 * W1
        + 1.0 * W2
        - 1.0 * W3
        + 0.2 * W4
        - 0.3 * W5
        + 0.1 * W6
    )

    pY1 = expit(logits_y1)
    pY0 = expit(logits_y0)
    ate = float(np.mean(pY1 - pY0))

    df = pd.DataFrame(
        {
            "W1": W1,
            "W2": W2,
            "W3": W3,
            "W4": W4,
            "W5": W5,
            "W6": W6,
            "A": A.astype(int),
            "Y": Y.astype(int),
            "pA": pA,
            "pA_obs": pA_obs,
            "pY": pY,
        }
    )

    if truth:
        return df, ate, float(np.mean(pY1)), float(np.mean(pY0))

    return df


def summarize_assignment(df):
    return {
        "n": int(len(df)),
        "treated": int((df["A"] == 1).sum()),
        "control": int((df["A"] == 0).sum()),
        "treat_rate": float(df["A"].mean()),
        "min_pA": float(df["pA"].min()),
        "max_pA": float(df["pA"].max()),
        "min_pA_obs": float(df["pA_obs"].min()),
        "max_pA_obs": float(df["pA_obs"].max()),
        "count_pA_obs_lt_0.001": int((df["pA_obs"] < 0.001).sum()),
        "count_pA_obs_gt_0.999": int((df["pA_obs"] > 0.999).sum()),
        "count_pA_obs_lt_0.01": int((df["pA_obs"] < 0.01).sum()),
        "count_pA_obs_gt_0.99": int((df["pA_obs"] > 0.99).sum()),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    truth_df, ate_true, y1_true, y0_true = generate_dataset(
        N_TRUTH,
        seed=42,
        rct=True,
        truth=True,
    )
    truth_df.drop(columns=["pA", "pA_obs", "pY"]).to_csv(
        os.path.join(OUT_DIR, "data_truth.csv"),
        index=False,
    )

    truth = {
        "seed": 42,
        "n": N_TRUTH,
        "rct": True,
        "ate_true": ate_true,
        "y1_truth": y1_true,
        "y0_truth": y0_true,
        "n_datasets": N_DATASETS,
        "n_per_dataset": N_PER_DATASET,
    }
    with open(TRUTH_PATH, "w") as f:
        json.dump(truth, f, indent=4)

    manifest = {
        "truth": truth,
        "datasets": {},
    }

    for i, seed in enumerate(SEEDS, start=1):
        df = generate_dataset(
            N_PER_DATASET,
            seed=seed,
            rct=True,
            truth=False,
        )

        out_path = os.path.join(OUT_DIR, f"data_{i}.csv")
        df.drop(columns=["pA", "pA_obs", "pY"]).to_csv(out_path, index=False)

        summary = summarize_assignment(df)
        manifest["datasets"][f"data_{i}"] = {
            "seed": int(seed),
            "file": out_path,
            **summary,
        }

        print(
            f"saved data_{i}.csv | seed={seed} | n={len(df)} | "
            f"A=1:{int((df['A'] == 1).sum())} | Y=1:{int((df['Y'] == 1).sum())}"
        )

    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"saved truth to {TRUTH_PATH}")
    print(f"saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()