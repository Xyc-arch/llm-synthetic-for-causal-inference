#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


def generate_covariates(n, d=6, seed=42):
    rng = np.random.default_rng(seed)

    if d < 6:
        raise ValueError("d must be at least 6.")

    # ------------------------------------------------------------------
    # Low-dimensional d6-style covariates.
    # Keep this simple/original so d6 and d20 are intentionally different.
    # ------------------------------------------------------------------
    if d < 15:
        W = {}

        W["W1"] = rng.binomial(1, 0.5, n)
        W["W2"] = rng.binomial(1, 0.5, n)

        pW3 = 0.3 + 0.35 * ((W["W1"] + W["W2"]) / 2)
        W["W3"] = rng.binomial(1, pW3, n)

        W["W4"] = rng.normal(0, 1, n)
        W["W5"] = rng.normal(0, 1, n)
        W["W6"] = 0.5 * W["W4"] + 0.5 * W["W5"] + rng.normal(0, 1, n)

        # Mild extra covariates if d is between 7 and 14.
        for j in range(7, d + 1):
            prev = W[f"W{j - 1}"]
            W[f"W{j}"] = 0.3 * prev + rng.normal(0, 1, n)

        return pd.DataFrame(W)

    # ------------------------------------------------------------------
    # High-dimensional d20-style covariates.
    # This creates strong, nonlinear, multi-way dependence among W1-W20.
    # ------------------------------------------------------------------

    # Shared latent factors create global dependence across many covariates.
    Z1 = rng.normal(0, 1, n)
    Z2 = rng.normal(0, 1, n)
    Z3 = rng.normal(0, 1, n)
    Z4 = rng.normal(0, 1, n)

    # Correlated noise block for continuous covariates.
    # AR(1)-like covariance creates additional dependence beyond the latent factors.
    rho = 0.65
    idx = np.arange(d)
    cov = rho ** np.abs(np.subtract.outer(idx, idx))
    E = rng.multivariate_normal(mean=np.zeros(d), cov=cov, size=n)

    W = {}

    # Binary covariates with nonlinear latent dependence.
    p1 = expit(0.4 * Z1 - 0.3 * Z2 + 0.2 * Z1 * Z2)
    W["W1"] = rng.binomial(1, p1, n)

    p2 = expit(-0.2 + 0.5 * Z2 + 0.4 * Z3 - 0.3 * Z1)
    W["W2"] = rng.binomial(1, p2, n)

    p3 = expit(
        -0.4
        + 0.8 * W["W1"]
        - 0.6 * W["W2"]
        + 0.5 * Z1
        + 0.4 * np.sin(Z3)
    )
    W["W3"] = rng.binomial(1, p3, n)

    # Continuous covariates with shared latent factors and binary parents.
    W["W4"] = (
        0.8 * Z1
        + 0.4 * Z2
        + 0.6 * W["W1"]
        - 0.4 * W["W2"]
        + E[:, 3]
    )

    W["W5"] = (
        -0.5 * Z1
        + 0.9 * Z3
        + 0.5 * W["W2"]
        + 0.3 * W["W3"]
        + E[:, 4]
    )

    W["W6"] = (
        0.5 * W["W4"]
        + 0.4 * W["W5"]
        + 0.6 * Z2
        - 0.3 * Z4
        + E[:, 5]
    )

    # W7-W15: strongly dependent nonlinear covariates.
    W["W7"] = (
        0.6 * W["W4"]
        - 0.4 * W["W5"]
        + 0.5 * np.sin(W["W6"])
        + 0.4 * Z1 * Z3
        + E[:, 6]
    )

    W["W8"] = (
        0.5 * W["W6"]
        + 0.5 * W["W7"]
        + 0.4 * np.tanh(W["W4"])
        - 0.3 * W["W1"]
        + E[:, 7]
    )

    W["W9"] = (
        -0.4 * W["W4"]
        + 0.6 * W["W8"]
        + 0.5 * np.sin(W["W5"] * W["W7"])
        + 0.3 * Z2
        + E[:, 8]
    )

    W["W10"] = (
        0.5 * W["W7"]
        - 0.5 * W["W9"]
        + 0.4 * W["W3"]
        + 0.4 * np.cos(W["W6"])
        + E[:, 9]
    )

    W["W11"] = (
        0.4 * W["W8"]
        + 0.4 * W["W10"]
        + 0.5 * np.tanh(W["W4"] * W["W5"])
        - 0.3 * Z3
        + E[:, 10]
    )

    W["W12"] = (
        0.5 * W["W9"]
        - 0.4 * W["W11"]
        + 0.5 * np.sin(W["W7"])
        + 0.4 * W["W1"] * W["W2"]
        + E[:, 11]
    )

    W["W13"] = (
        0.5 * W["W10"]
        + 0.4 * W["W12"]
        - 0.4 * np.abs(W["W5"])
        + 0.3 * Z4
        + E[:, 12]
    )

    W["W14"] = (
        0.4 * W["W11"]
        - 0.5 * W["W13"]
        + 0.4 * np.sin(W["W8"] * W["W9"])
        + 0.3 * W["W3"]
        + E[:, 13]
    )

    W["W15"] = (
        0.5 * W["W12"]
        + 0.5 * W["W14"]
        + 0.4 * np.tanh(W["W10"])
        - 0.3 * Z1
        + E[:, 14]
    )

    # W16-W20: nuisance covariates, but still strongly dependent.
    # They do not directly enter the outcome model, but they are correlated
    # with outcome-relevant covariates W1-W15.
    W["W16"] = (
        0.6 * W["W13"]
        - 0.4 * W["W15"]
        + 0.4 * np.sin(W["W7"])
        + E[:, 15]
    )

    W["W17"] = (
        0.5 * W["W14"]
        + 0.4 * W["W16"]
        + 0.3 * W["W2"]
        - 0.3 * Z2
        + E[:, 16]
    )

    W["W18"] = (
        0.5 * W["W15"]
        - 0.4 * W["W17"]
        + 0.4 * np.cos(W["W11"])
        + E[:, 17]
    )

    W["W19"] = (
        0.4 * W["W16"]
        + 0.4 * W["W18"]
        + 0.3 * np.tanh(W["W12"] * W["W13"])
        + E[:, 18]
    )

    W["W20"] = (
        0.5 * W["W17"]
        - 0.4 * W["W19"]
        + 0.4 * np.sin(W["W15"])
        + 0.3 * Z4
        + E[:, 19]
    )

    # If d > 20, generate additional dependent nuisance covariates.
    # You said d50 is not needed now, but this keeps the function safe.
    for j in range(21, d + 1):
        W[f"W{j}"] = (
            0.35 * W[f"W{j - 1}"]
            + 0.25 * W[f"W{j - 3}"]
            - 0.20 * np.sin(W[f"W{j - 5}"])
            + rng.normal(0, 1, n)
        )

    return pd.DataFrame(W)


def treatment_logits(W, overlap="poor"):
    W1 = W["W1"]
    W2 = W["W2"]
    W3 = W["W3"]
    W4 = W["W4"]
    W5 = W["W5"]
    W6 = W["W6"]

    if overlap == "moderate":
        logits = (
            -0.4
            + 0.7 * W1
            - 0.7 * W2
            + 0.5 * W3
            + 0.4 * W4
            - 0.4 * W5
            + 0.5 * W6
        )

    elif overlap == "poor":
        logits = (
            -1.0
            + 1.4 * W1
            - 1.4 * W2
            + 1.0 * W3
            + 0.9 * W4
            - 0.9 * W5
            + 1.2 * W6
        )

    elif overlap == "extreme":
        logits = (
            -8.0
            + 4.0 * W1
            - 6.0 * W2
            + 3.0 * W3
            + 2.0 * W4
            - 3.0 * W5
            + 4.0 * W6
        )

    else:
        raise ValueError(f"Unknown overlap setting: {overlap}")

    # Weak dependence on extra dimensions for treatment assignment.
    # This keeps the main positivity/overlap structure comparable across d6 and d20.
    extra_cols = [c for c in W.columns if int(c[1:]) > 6]
    if extra_cols:
        extra = W[extra_cols].to_numpy()
        coefs = np.array([0.15 * ((-1) ** j) for j in range(extra.shape[1])])
        logits = logits + extra @ coefs

    return logits


def baseline_outcome_logits(W):
    d = W.shape[1]

    W1 = W["W1"]
    W2 = W["W2"]
    W3 = W["W3"]
    W4 = W["W4"]
    W5 = W["W5"]
    W6 = W["W6"]

    if d >= 15:
        # High-dimensional baseline outcome model for d20 settings.
        # This is intentionally quite different from the d6 outcome model.
        # The baseline outcome depends meaningfully on W1-W15.
        W7 = W["W7"]
        W8 = W["W8"]
        W9 = W["W9"]
        W10 = W["W10"]
        W11 = W["W11"]
        W12 = W["W12"]
        W13 = W["W13"]
        W14 = W["W14"]
        W15 = W["W15"]

        eta = (
            -0.8
            + 0.35 * W1
            - 0.45 * W2
            + 0.50 * W3
            + 0.40 * np.sin(W4)
            - 0.35 * np.tanh(W5)
            + 0.25 * W6
            + 0.45 * np.sin(W7)
            - 0.40 * np.cos(W8)
            + 0.35 * W9
            - 0.35 * W10
            + 0.30 * np.clip(W11, -3, 3) ** 2
            - 0.30 * np.abs(W12)
            + 0.35 * W13 * W14
            - 0.25 * W15
            + 0.30 * W4 * W9
            - 0.25 * W7 * W10
            + 0.20 * W11 * W15
        )

    else:
        # Original low-dimensional d6-style baseline outcome model.
        eta = (
            -0.5
            + 0.5 * W1
            + 1.0 * W2
            - 1.0 * W3
            + 0.2 * W4
            - 0.3 * W5
            + 0.1 * W6
        )

        # Weak nuisance dependence if d is between 7 and 14.
        extra_cols = [c for c in W.columns if int(c[1:]) > 6]
        if extra_cols:
            extra = W[extra_cols].to_numpy()
            coefs = np.array([0.05 * ((-1) ** j) for j in range(extra.shape[1])])
            eta = eta + extra @ coefs

    return eta


def treatment_effect(W, outcome_mode="complex"):
    d = W.shape[1]

    W1 = W["W1"]
    W2 = W["W2"]
    W3 = W["W3"]
    W4 = W["W4"]
    W5 = W["W5"]
    W6 = W["W6"]

    if outcome_mode == "simple":
        # Simple means constant treatment effect on the logit scale.
        # For d20, the baseline outcome can still be high-dimensional,
        # but treatment-effect heterogeneity is intentionally removed.
        tau = np.full(len(W), 2.0)

    elif outcome_mode == "complex":
        if d >= 15:
            # High-dimensional heterogeneous treatment effect for d20.
            # Uses W1-W15 meaningfully and differs strongly from the d6 tau model.
            W7 = W["W7"]
            W8 = W["W8"]
            W9 = W["W9"]
            W10 = W["W10"]
            W11 = W["W11"]
            W12 = W["W12"]
            W13 = W["W13"]
            W14 = W["W14"]
            W15 = W["W15"]

            tau = (
                1.8
                + 0.35 * W1
                - 0.25 * W2
                + 0.30 * W3
                + 0.30 * np.sin(W4)
                - 0.25 * np.tanh(W5)
                + 0.20 * np.cos(W6)
                + 0.35 * np.tanh(W7)
                - 0.30 * np.sin(W8)
                + 0.25 * W9
                - 0.25 * W10
                + 0.20 * np.cos(W11)
                + 0.20 * W12 * W13
                - 0.20 * np.abs(W14)
                + 0.25 * W15
                + 0.20 * W7 * W10
                - 0.15 * W11 * W15
            )

        else:
            # Original low-dimensional heterogeneous treatment effect.
            tau = (
                2.0
                + 0.5 * np.sin(W1)
                + 0.3 * np.log(np.abs(W2) + 1)
                - 0.2 * (W3 ** 2)
                + 0.1 * np.exp(np.clip(W4, -3, 3))
                - 0.3 * np.tanh(W5)
                + 0.2 * np.cos(W6)
            )

            # Weak nuisance dependence if d is between 7 and 14.
            extra_cols = [c for c in W.columns if int(c[1:]) > 6]
            if extra_cols:
                extra = W[extra_cols].to_numpy()
                tau = tau + 0.05 * np.tanh(extra[:, : min(5, extra.shape[1])]).sum(axis=1)

    else:
        raise ValueError(f"Unknown outcome_mode: {outcome_mode}")

    return tau


def generate_dataset(
    n,
    seed=42,
    d=6,
    rct=False,
    truth=False,
    overlap="poor",
    outcome_mode="complex",
    verbose=True,
):
    rng = np.random.default_rng(seed)

    W = generate_covariates(n=n, d=d, seed=seed)

    logits_A = treatment_logits(W, overlap=overlap)
    pA = expit(logits_A)

    if rct:
        A = rng.binomial(1, 0.5, n)
    else:
        A = rng.binomial(1, pA, n)

    tau = treatment_effect(W, outcome_mode=outcome_mode)
    eta0 = baseline_outcome_logits(W)

    pY0 = expit(eta0)
    pY1 = expit(eta0 + tau)

    pY = expit(eta0 + tau * A)
    Y = rng.binomial(1, pY, n)

    ate = float(np.mean(pY1 - pY0))
    y1_truth = float(np.mean(pY1))
    y0_truth = float(np.mean(pY0))

    data = W.copy()
    data["A"] = A
    data["Y"] = Y
    data["pA"] = pA
    data["pY"] = pY

    diagnostics = {
        "n": int(n),
        "d": int(d),
        "rct": bool(rct),
        "overlap": overlap,
        "outcome_mode": outcome_mode,
        "count_A1": int((A == 1).sum()),
        "count_A0": int((A == 0).sum()),
        "count_Y1": int((Y == 1).sum()),
        "count_Y0": int((Y == 0).sum()),
        "pA_min": float(np.min(pA)),
        "pA_q05": float(np.quantile(pA, 0.05)),
        "pA_median": float(np.median(pA)),
        "pA_q95": float(np.quantile(pA, 0.95)),
        "pA_max": float(np.max(pA)),
        "ate": ate,
        "y1_truth": y1_truth,
        "y0_truth": y0_truth,
    }

    if verbose:
        print(json.dumps(diagnostics, indent=2))

    if truth:
        return data, ate, y1_truth, y0_truth, diagnostics

    return data


def save_without_probs(df, path):
    drop_cols = [c for c in ["pA", "pY"] if c in df.columns]
    df.drop(drop_cols, axis=1).to_csv(path, index=False)


def save_setting(
    out_dir,
    d,
    n_seed,
    n_test,
    n_obs,
    n_truth,
    overlap,
    outcome_mode,
    seed_data_rct=False,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Generating setting: {out_dir}")
    print("=" * 80)

    data_truth, ate_true, y1_truth, y0_truth, truth_diag = generate_dataset(
        n=n_truth,
        seed=42,
        d=d,
        rct=True,
        truth=True,
        overlap=overlap,
        outcome_mode=outcome_mode,
        verbose=False,
    )
    save_without_probs(data_truth, out_dir / "data_truth.csv")

    data_obs = generate_dataset(
        n=n_obs,
        seed=1,
        d=d,
        rct=False,
        truth=False,
        overlap=overlap,
        outcome_mode=outcome_mode,
        verbose=False,
    )
    save_without_probs(data_obs, out_dir / "data.csv")

    # Important: seed_data_rct=False makes positivity settings affect the training data.
    data_seed = generate_dataset(
        n=n_seed,
        seed=2,
        d=d,
        rct=seed_data_rct,
        truth=False,
        overlap=overlap,
        outcome_mode=outcome_mode,
        verbose=False,
    )
    save_without_probs(data_seed, out_dir / "data_seed.csv")

    data_test = generate_dataset(
        n=n_test,
        seed=3,
        d=d,
        rct=True,
        truth=False,
        overlap=overlap,
        outcome_mode=outcome_mode,
        verbose=False,
    )
    save_without_probs(data_test, out_dir / "data_test.csv")

    truth = {
        "seed": 42,
        "n_truth": int(n_truth),
        "n_obs": int(n_obs),
        "n_seed": int(n_seed),
        "n_test": int(n_test),
        "d": int(d),
        "overlap": overlap,
        "outcome_mode": outcome_mode,
        "seed_data_rct": bool(seed_data_rct),
        "test_data_rct": True,
        "ate_true": float(ate_true),
        "y1_truth": float(y1_truth),
        "y0_truth": float(y0_truth),
        "truth_diagnostics": truth_diag,
    }

    with open(out_dir / "truth.json", "w") as f:
        json.dump(truth, f, indent=4)

    print(f"Saved: {out_dir}")
    print(json.dumps(truth, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=str,
        default="simulator_vary_data",
        help="Output directory relative to this script.",
    )
    parser.add_argument(
        "--seed-data-rct",
        action="store_true",
        help="If set, data_seed.csv uses randomized treatment. Default is observational seed data.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent / args.output_root

    settings = [
        # Low-dimensional moderate-overlap setting.
        {
            "name": "d6_n1000_overlap_moderate_complex",
            "d": 6,
            "n_seed": 1000,
            "n_test": 1000,
            "n_obs": 200,
            "n_truth": 50000,
            "overlap": "moderate",
            "outcome_mode": "complex",
        },

        # Low-dimensional poor-overlap setting.
        {
            "name": "d6_n1000_overlap_poor_complex",
            "d": 6,
            "n_seed": 1000,
            "n_test": 1000,
            "n_obs": 200,
            "n_truth": 50000,
            "overlap": "poor",
            "outcome_mode": "complex",
        },

        # High-dimensional poor-overlap setting.
        # Covariates are highly dependent, and outcome depends meaningfully on W1-W15.
        {
            "name": "d20_n1000_overlap_poor_complex",
            "d": 20,
            "n_seed": 1000,
            "n_test": 1000,
            "n_obs": 200,
            "n_truth": 50000,
            "overlap": "poor",
            "outcome_mode": "complex",
        },

        # Smaller seed sample size.
        # Same high-dimensional covariate and outcome structure as d20_n1000.
        {
            "name": "d20_n500_overlap_poor_complex",
            "d": 20,
            "n_seed": 500,
            "n_test": 1000,
            "n_obs": 200,
            "n_truth": 50000,
            "overlap": "poor",
            "outcome_mode": "complex",
        },

        # High-dimensional baseline outcome, but constant treatment effect.
        # This separates baseline outcome complexity from treatment-effect heterogeneity.
        {
            "name": "d20_n1000_overlap_poor_simple",
            "d": 20,
            "n_seed": 1000,
            "n_test": 1000,
            "n_obs": 200,
            "n_truth": 50000,
            "overlap": "poor",
            "outcome_mode": "simple",
        },
    ]

    for cfg in settings:
        save_setting(
            out_dir=base_dir / cfg["name"],
            d=cfg["d"],
            n_seed=cfg["n_seed"],
            n_test=cfg["n_test"],
            n_obs=cfg["n_obs"],
            n_truth=cfg["n_truth"],
            overlap=cfg["overlap"],
            outcome_mode=cfg["outcome_mode"],
            seed_data_rct=args.seed_data_rct,
        )


if __name__ == "__main__":
    main()