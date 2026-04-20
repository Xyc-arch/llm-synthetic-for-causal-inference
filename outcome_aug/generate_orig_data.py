import os
import json
import numpy as np
import pandas as pd
from scipy.special import expit

OUT_DIR = "/home/ubuntu/syn_causal/outcome_aug/data"
TRUTH_PATH = os.path.join(OUT_DIR, "truth.json")
SEEDS = [1, 2, 3, 4, 5]
N_PER_DATASET = 200
N_TRUTH = 100000


def generate_dataset(n, seed=42, rct=False, truth=False):
    rng = np.random.default_rng(seed)

    W1 = rng.binomial(1, 0.5, n)
    W2 = rng.binomial(1, 0.5, n)

    pW3 = 0.3 + 0.35 * ((W1 + W2) / 2)
    W3 = rng.binomial(1, pW3, n)

    W4 = rng.normal(0, 1, n)
    W5 = rng.normal(0, 1, n)
    W6 = 0.5 * W4 + 0.5 * W5 + rng.normal(0, 1, n)

    # Observational treatment mechanism.
    # Intercept softened from -30 to -3.0 to avoid near-deterministic treatment.
    logits = -3.0 + 1.6 * W1 - 2.4 * W2 + 1.2 * W3 + 0.6 * W4 - 1.0 * W5 + 1.6 * W6
    pA = expit(logits)

    if rct:
        A = rng.binomial(1, 0.5, n)
    else:
        A = rng.binomial(1, pA, n)

    tau = (
        2.0
        + 0.5 * np.sin(W1)
        + 0.3 * np.log(np.abs(W2) + 1)
        - 0.2 * (W3 ** 2)
        + 0.1 * np.exp(W4)
        - 0.3 * np.tanh(W5)
        + 0.2 * np.cos(W6)
    )

    outcome_logits = (
        -0.5
        + tau * A
        + 0.5 * W1
        + 1.0 * W2
        - 1.0 * W3
        + 0.2 * W4
        - 0.3 * W5
        + 0.1 * W6
    )
    pY = expit(outcome_logits)
    Y = rng.binomial(1, pY, n)

    # Potential outcome means for ground-truth ATE
    outcome_logits_treated = (
        -0.5
        + tau
        + 0.5 * W1
        + 1.0 * W2
        - 1.0 * W3
        + 0.2 * W4
        - 0.3 * W5
        + 0.1 * W6
    )
    outcome_logits_control = (
        -0.5
        + 0.5 * W1
        + 1.0 * W2
        - 1.0 * W3
        + 0.2 * W4
        - 0.3 * W5
        + 0.1 * W6
    )

    pY1 = expit(outcome_logits_treated)
    pY0 = expit(outcome_logits_control)
    ate = float(np.mean(pY1 - pY0))

    data = pd.DataFrame(
        {
            "W1": W1,
            "W2": W2,
            "W3": W3,
            "W4": W4,
            "W5": W5,
            "W6": W6,
            "A": A,
            "Y": Y,
            "pA": pA,
            "pY": pY,
        }
    )

    if truth:
        return data, ate, float(np.mean(pY1)), float(np.mean(pY0))
    return data


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Generate truth under observational treatment assignment
    data_truth, ate_true, y1_truth, y0_truth = generate_dataset(
        N_TRUTH, seed=42, rct=False, truth=True
    )
    data_truth.drop(columns=["pA", "pY"]).to_csv(
        os.path.join(OUT_DIR, "data_truth.csv"), index=False
    )

    truth = {
        "seed": 42,
        "n": N_TRUTH,
        "rct": False,
        "ate_true": ate_true,
        "y1_truth": y1_truth,
        "y0_truth": y0_truth,
    }
    with open(TRUTH_PATH, "w") as f:
        json.dump(truth, f, indent=4)

    for i, seed in enumerate(SEEDS, start=1):
        df = generate_dataset(N_PER_DATASET, seed=seed, rct=False, truth=False)
        df.drop(columns=["pA", "pY"]).to_csv(
            os.path.join(OUT_DIR, f"data_{i}.csv"), index=False
        )

        print(
            f"saved data_{i}.csv | seed={seed} | n={len(df)} | "
            f"A=1:{int((df['A'] == 1).sum())} | Y=1:{int((df['Y'] == 1).sum())} | "
            f"mean_pA:{df['pA'].mean():.3f} | min_pA:{df['pA'].min():.3f} | max_pA:{df['pA'].max():.3f}"
        )

    print(f"saved truth to {TRUTH_PATH}")


if __name__ == "__main__":
    main()