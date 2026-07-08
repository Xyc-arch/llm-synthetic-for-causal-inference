#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTG_DIR = PROJECT_ROOT / "actg175"
PRIVACY_DIR = PROJECT_ROOT / "privacy"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algs.tmle_continuous import estimate_tmle_continuous_df

AGG_LLM_JSON = ACTG_DIR / "results" / "aggregate_simulator_llm.json"
SIM_JSON = ACTG_DIR / "results" / "simulation_engine_synth_only.json"

CLEAN_CANDIDATES = [
    ACTG_DIR / "data" / "actg175_clean.csv",
    PRIVACY_DIR / "data" / "actg175_clean.csv",
]

RAW_ACTG_PATH = ACTG_DIR / "actg175.csv"
OUTPUT_CLEAN_PATH = ACTG_DIR / "data" / "actg175_clean.csv"

OUT_JSON = ACTG_DIR / "results" / "bayesian_tmle_curve_extrapolation.json"

# =========================
# ACTG setup
# =========================

COVARIATES = [
    "age", "wtkg", "hemo", "homo", "drugs", "karnof",
    "oprior", "z30", "zprior", "preanti", "race", "gender",
    "str2", "strat", "symptom", "cd40", "cd80",
]

CONT_VARS = ["age", "wtkg", "karnof", "preanti", "cd40", "cd80"]

OUTCOME_COL = "cd420"
TREATMENT_COL = "A"

# More grid points, fewer reps, as requested.
REAL_SUBSAMPLE_SIZES = list(range(50, 601, 50))
REAL_REPS_PER_SIZE = 20

# Synthetic-informed alpha prior.
ALPHA_MIN = -2 / 3
ALPHA_MAX = -1 / 2
ALPHA_PRIOR_SD = 0.04

# Weak priors for real-curve regression:
# mean_tmle(m) = psi_infty + beta * m^alpha + noise.
PSI_PRIOR_SD = 1000.0
BETA_PRIOR_SD = 10000.0

N_ALPHA_GRID = 5000
N_POSTERIOR_DRAWS = 20000
RANDOM_SEED = 12345

# Optional: compute full-data TMLE only for comparison, not for fitting.
COMPUTE_FULL_REAL_TMLE_FOR_REPORT = True


# =========================
# Data utilities
# =========================

def clean_actg_from_raw(raw_path: Path, out_path: Path) -> Path:
    if not raw_path.exists():
        raise FileNotFoundError(
            f"No cleaned ACTG file found and raw file is missing: {raw_path}"
        )

    df = pd.read_csv(raw_path)

    if "arms" not in df.columns:
        raise ValueError("Column 'arms' not found in raw ACTG175 data.")

    data_sub = df[df["arms"].isin([1, 2])].copy()
    data_sub["A"] = (data_sub["arms"] == 2).astype(int)

    needed = COVARIATES + [TREATMENT_COL, OUTCOME_COL]
    missing = [c for c in needed if c not in data_sub.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw ACTG data: {missing}")

    data_clean = data_sub[needed].dropna().copy()

    scaler = StandardScaler()
    data_clean.loc[:, CONT_VARS] = scaler.fit_transform(data_clean[CONT_VARS])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_clean.to_csv(out_path, index=False)

    print(f"Created cleaned ACTG file: {out_path}")
    print(f"Cleaned ACTG shape: {data_clean.shape}")

    return out_path


def find_or_create_clean_actg() -> Path:
    for path in CLEAN_CANDIDATES:
        if path.exists():
            print(f"Using cleaned ACTG file: {path}")
            return path

    return clean_actg_from_raw(RAW_ACTG_PATH, OUTPUT_CLEAN_PATH)


def load_real_actg() -> pd.DataFrame:
    clean_path = find_or_create_clean_actg()
    df = pd.read_csv(clean_path)

    needed = COVARIATES + [TREATMENT_COL, OUTCOME_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Cleaned ACTG file missing columns: {missing}")

    df = df[needed].dropna().reset_index(drop=True)

    for c in CONT_VARS + [OUTCOME_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    discrete_cols = [c for c in COVARIATES if c not in CONT_VARS] + [TREATMENT_COL]
    for c in discrete_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").round().astype(int)

    df = df.dropna().reset_index(drop=True)

    return df


def estimate_tmle(df: pd.DataFrame) -> float:
    return float(
        estimate_tmle_continuous_df(
            data=df.copy(),
            covariates=COVARIATES,
            outcome_col=OUTCOME_COL,
            treatment_col=TREATMENT_COL,
            random_state=42,
        )
    )


def sample_without_replacement(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n > len(df):
        raise ValueError(f"Requested n={n}, but data has only {len(df)} rows.")

    rng = np.random.default_rng(seed)
    idx = rng.choice(df.index.to_numpy(), size=n, replace=False)
    return df.loc[idx].reset_index(drop=True)


# =========================
# Synthetic alpha fit
# =========================

def load_synthetic_tmle_bias_curve():
    """
    Load synthetic LLM TMLE bias curve:
        bias_syn(n) = mean TMLE at n - full synthetic TMLE reference.
    """

    if AGG_LLM_JSON.exists():
        with open(AGG_LLM_JSON, "r") as f:
            obj = json.load(f)

        by_n = obj["by_sample_size"]

        n_values = []
        bias_values = []

        for n_str in sorted(by_n.keys(), key=lambda x: int(x)):
            if "tmle_continuous" not in by_n[n_str]:
                continue
            n_values.append(int(n_str))
            bias_values.append(
                float(by_n[n_str]["tmle_continuous"]["bias_vs_full_tmle"])
            )

        print(f"Loaded synthetic TMLE bias curve from: {AGG_LLM_JSON}")
        return np.asarray(n_values, dtype=float), np.asarray(bias_values, dtype=float)

    if SIM_JSON.exists():
        with open(SIM_JSON, "r") as f:
            obj = json.load(f)

        by_n = obj["synthetic_engine"]["llm"]["by_sample_size"]

        n_values = []
        bias_values = []

        for n_str in sorted(by_n.keys(), key=lambda x: int(x)):
            if "tmle_continuous" not in by_n[n_str]:
                continue

            summary = by_n[n_str]["tmle_continuous"]["against_synthetic_reference_truth"]
            n_values.append(int(n_str))
            bias_values.append(float(summary["bias"]))

        print(f"Loaded synthetic TMLE bias curve from: {SIM_JSON}")
        return np.asarray(n_values, dtype=float), np.asarray(bias_values, dtype=float)

    raise FileNotFoundError(
        f"Missing both:\n{AGG_LLM_JSON}\n{SIM_JSON}"
    )


def fit_synthetic_alpha_beta(n_values, bias_values):
    """
    Fit:
        bias(n) = beta * n^alpha

    Joint fit:
    - grid over alpha
    - closed-form least squares beta for each alpha
    """

    alpha_grid = np.linspace(ALPHA_MIN, ALPHA_MAX, N_ALPHA_GRID)

    best = None
    fit_rows = []

    for alpha in alpha_grid:
        x = n_values ** alpha
        beta = float(np.sum(x * bias_values) / np.sum(x * x))
        pred = beta * x
        residuals = bias_values - pred
        sse = float(np.sum(residuals ** 2))

        row = {
            "alpha": float(alpha),
            "beta": beta,
            "sse": sse,
        }
        fit_rows.append(row)

        if best is None or sse < best["sse"]:
            best = row

    alpha_hat = float(best["alpha"])
    beta_hat = float(best["beta"])

    fitted = beta_hat * (n_values ** alpha_hat)

    return {
        "alpha_hat": alpha_hat,
        "beta_hat": beta_hat,
        "sse": float(best["sse"]),
        "sample_sizes": n_values.astype(int).tolist(),
        "observed_bias": bias_values.tolist(),
        "fitted_bias": fitted.tolist(),
        "residuals": (bias_values - fitted).tolist(),
    }


# =========================
# Real subsampling curve
# =========================

def run_real_subsampling_curve(df: pd.DataFrame):
    """
    For each m, repeatedly subsample real data and estimate TMLE.
    We do not use full-data TMLE as truth.
    We only estimate the curve:
        mean_tmle(m) vs m.
    """

    N = len(df)
    usable_sizes = [m for m in REAL_SUBSAMPLE_SIZES if m < N]

    if not usable_sizes:
        raise ValueError(f"No usable subsample sizes. Real N={N}")

    results = {}

    for m in usable_sizes:
        estimates = []

        print(f"Real subsampling: m={m}, reps={REAL_REPS_PER_SIZE}")

        for r in range(REAL_REPS_PER_SIZE):
            seed = 700000 + 1000 * m + r
            sub = sample_without_replacement(df, n=m, seed=seed)
            est = estimate_tmle(sub)
            estimates.append(est)

        arr = np.asarray(estimates, dtype=float)

        mean_est = float(np.mean(arr))
        sd_est = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        se_mean = sd_est / np.sqrt(len(arr)) if len(arr) > 1 else 1.0

        # Prevent a point from having infinite weight.
        se_mean = max(se_mean, 1e-8)

        results[str(m)] = {
            "m": int(m),
            "n_reps": int(len(arr)),
            "mean_tmle": mean_est,
            "sd_tmle": sd_est,
            "se_mean_tmle": se_mean,
        }

    return results


# =========================
# Bayesian curve extrapolation
# =========================

def log_mvn_density(y, mean, cov):
    """
    Stable log density of multivariate normal N(mean, cov).
    """

    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return -np.inf

    resid = y - mean
    sol = np.linalg.solve(cov, resid)
    quad = float(resid.T @ sol)

    k = len(y)
    return -0.5 * (k * np.log(2.0 * np.pi) + logdet + quad)


def bayesian_curve_extrapolation(
    real_curve,
    alpha_prior_mean,
    alpha_prior_sd=ALPHA_PRIOR_SD,
    alpha_min=ALPHA_MIN,
    alpha_max=ALPHA_MAX,
    psi_prior_sd=PSI_PRIOR_SD,
    beta_prior_sd=BETA_PRIOR_SD,
    n_alpha_grid=N_ALPHA_GRID,
    n_draws=N_POSTERIOR_DRAWS,
    seed=RANDOM_SEED,
):
    """
    Bayesian model:

        y_k = psi_infty + beta * m_k^alpha + eps_k
        eps_k ~ N(0, se_k^2)

    Priors:
        alpha ~ TruncatedNormal(alpha_prior_mean, alpha_prior_sd^2)
        psi_infty ~ Normal(mean(y), psi_prior_sd^2)
        beta ~ Normal(0, beta_prior_sd^2)

    This does not use full-data TMLE as truth.
    """

    rng = np.random.default_rng(seed)

    m_values = []
    y_values = []
    se_values = []

    for m_str in sorted(real_curve.keys(), key=lambda x: int(x)):
        row = real_curve[m_str]
        m_values.append(float(row["m"]))
        y_values.append(float(row["mean_tmle"]))
        se_values.append(float(row["se_mean_tmle"]))

    m_values = np.asarray(m_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    se_values = np.asarray(se_values, dtype=float)

    alpha_grid = np.linspace(alpha_min, alpha_max, n_alpha_grid)

    # Prior for theta = [psi_infty, beta].
    theta_prior_mean = np.asarray([float(np.mean(y_values)), 0.0])
    theta_prior_cov = np.diag([psi_prior_sd ** 2, beta_prior_sd ** 2])
    theta_prior_prec = np.linalg.inv(theta_prior_cov)

    obs_cov = np.diag(se_values ** 2)

    log_weights = []
    theta_post_means = []
    theta_post_covs = []

    for alpha in alpha_grid:
        x = m_values ** alpha
        X = np.column_stack([np.ones_like(x), x])

        # p(alpha), up to truncation constant.
        log_alpha_prior = -0.5 * ((alpha - alpha_prior_mean) / alpha_prior_sd) ** 2

        # Marginal likelihood integrating out theta:
        # y | alpha ~ N(X mu0, obs_cov + X Sigma0 X')
        marginal_mean = X @ theta_prior_mean
        marginal_cov = obs_cov + X @ theta_prior_cov @ X.T
        log_marginal = log_mvn_density(y_values, marginal_mean, marginal_cov)

        # Posterior theta | alpha, y.
        obs_prec = np.diag(1.0 / (se_values ** 2))
        post_prec = theta_prior_prec + X.T @ obs_prec @ X
        post_cov = np.linalg.inv(post_prec)
        post_mean = post_cov @ (
            theta_prior_prec @ theta_prior_mean + X.T @ obs_prec @ y_values
        )

        log_weights.append(float(log_alpha_prior + log_marginal))
        theta_post_means.append(post_mean)
        theta_post_covs.append(post_cov)

    log_weights = np.asarray(log_weights, dtype=float)
    log_weights = log_weights - np.max(log_weights)

    weights = np.exp(log_weights)
    weights = weights / np.sum(weights)

    alpha_idx = rng.choice(
        np.arange(len(alpha_grid)),
        size=n_draws,
        replace=True,
        p=weights,
    )

    alpha_draws = alpha_grid[alpha_idx]
    psi_draws = np.zeros(n_draws)
    beta_draws = np.zeros(n_draws)

    for i, idx in enumerate(alpha_idx):
        draw = rng.multivariate_normal(
            mean=theta_post_means[idx],
            cov=theta_post_covs[idx],
        )
        psi_draws[i] = draw[0]
        beta_draws[i] = draw[1]

    def summarize(arr):
        return {
            "mean": float(np.mean(arr)),
            "sd": float(np.std(arr, ddof=1)),
            "q025": float(np.quantile(arr, 0.025)),
            "q50": float(np.quantile(arr, 0.50)),
            "q975": float(np.quantile(arr, 0.975)),
        }

    # Posterior predictive mean curve at observed m.
    fitted_curve = {}
    for m in m_values:
        pred_draws = psi_draws + beta_draws * (m ** alpha_draws)
        fitted_curve[str(int(m))] = summarize(pred_draws)

    return {
        "model": "mean_tmle(m) = psi_infty + beta * m^alpha + error",
        "alpha_prior": {
            "center_from_synthetic": float(alpha_prior_mean),
            "sd": float(alpha_prior_sd),
            "range": [float(alpha_min), float(alpha_max)],
        },
        "priors": {
            "psi_infty_prior_mean": float(theta_prior_mean[0]),
            "psi_infty_prior_sd": float(psi_prior_sd),
            "beta_prior_mean": 0.0,
            "beta_prior_sd": float(beta_prior_sd),
        },
        "posterior": {
            "alpha": summarize(alpha_draws),
            "beta": summarize(beta_draws),
            "psi_infty": summarize(psi_draws),
        },
        "fitted_curve": fitted_curve,
    }


# =========================
# Main
# =========================

def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # 1. Synthetic alpha prior.
    syn_n, syn_bias = load_synthetic_tmle_bias_curve()
    syn_fit = fit_synthetic_alpha_beta(syn_n, syn_bias)

    alpha_syn = float(syn_fit["alpha_hat"])

    print("\nSynthetic TMLE bias power-law fit")
    print(f"alpha_hat = {syn_fit['alpha_hat']:.6f}")
    print(f"beta_hat  = {syn_fit['beta_hat']:.6f}")
    print(f"sse       = {syn_fit['sse']:.6f}")

    # 2. Real ACTG subsampling curve.
    real_df = load_real_actg()
    N_real = len(real_df)

    print(f"\nReal ACTG full sample size: N={N_real}")

    real_curve = run_real_subsampling_curve(real_df)

    # 3. Bayesian extrapolation of real curve.
    bayes = bayesian_curve_extrapolation(
        real_curve=real_curve,
        alpha_prior_mean=alpha_syn,
        alpha_prior_sd=ALPHA_PRIOR_SD,
        alpha_min=ALPHA_MIN,
        alpha_max=ALPHA_MAX,
        psi_prior_sd=PSI_PRIOR_SD,
        beta_prior_sd=BETA_PRIOR_SD,
        n_alpha_grid=N_ALPHA_GRID,
        n_draws=N_POSTERIOR_DRAWS,
        seed=RANDOM_SEED,
    )

    # 4. Optional full-data TMLE for comparison only.
    full_real_tmle = None
    full_real_tmle_minus_psi = None

    if COMPUTE_FULL_REAL_TMLE_FOR_REPORT:
        print("\nComputing full-data TMLE for comparison only...")
        full_real_tmle = estimate_tmle(real_df)
        psi_mean = bayes["posterior"]["psi_infty"]["mean"]
        full_real_tmle_minus_psi = full_real_tmle - psi_mean

        print(f"full_real_tmle = {full_real_tmle:.6f}")
        print(f"posterior psi_infty mean = {psi_mean:.6f}")
        print(f"full_real_tmle - psi_infty = {full_real_tmle_minus_psi:.6f}")

    result = {
        "config": {
            "project_root": str(PROJECT_ROOT),
            "actg_dir": str(ACTG_DIR),
            "aggregate_llm_json": str(AGG_LLM_JSON),
            "simulation_json": str(SIM_JSON),
            "output_json": str(OUT_JSON),
            "covariates": COVARIATES,
            "outcome_col": OUTCOME_COL,
            "treatment_col": TREATMENT_COL,
            "real_subsample_sizes": REAL_SUBSAMPLE_SIZES,
            "real_reps_per_size": REAL_REPS_PER_SIZE,
            "alpha_range": [float(ALPHA_MIN), float(ALPHA_MAX)],
            "alpha_prior_sd": float(ALPHA_PRIOR_SD),
            "psi_prior_sd": float(PSI_PRIOR_SD),
            "beta_prior_sd": float(BETA_PRIOR_SD),
            "n_alpha_grid": int(N_ALPHA_GRID),
            "n_posterior_draws": int(N_POSTERIOR_DRAWS),
            "random_seed": int(RANDOM_SEED),
            "uses_full_real_tmle_as_truth": False,
            "full_real_tmle_used_for_report_only": bool(COMPUTE_FULL_REAL_TMLE_FOR_REPORT),
        },
        "synthetic_alpha_fit": syn_fit,
        "real_data": {
            "n_full": int(N_real),
            "subsampling_curve": real_curve,
            "full_real_tmle_for_comparison_only": full_real_tmle,
            "full_real_tmle_minus_posterior_psi_mean": full_real_tmle_minus_psi,
        },
        "bayesian_curve_extrapolation": bayes,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=4)

    print(f"\nSaved Bayesian curve extrapolation to: {OUT_JSON}")

    print("\nPosterior summary")
    print(f"alpha mean:     {bayes['posterior']['alpha']['mean']:.6f}")
    print(f"alpha 95%:      [{bayes['posterior']['alpha']['q025']:.6f}, {bayes['posterior']['alpha']['q975']:.6f}]")
    print(f"beta mean:      {bayes['posterior']['beta']['mean']:.6f}")
    print(f"psi_infty mean: {bayes['posterior']['psi_infty']['mean']:.6f}")
    print(f"psi_infty 95%:  [{bayes['posterior']['psi_infty']['q025']:.6f}, {bayes['posterior']['psi_infty']['q975']:.6f}]")


if __name__ == "__main__":
    main()