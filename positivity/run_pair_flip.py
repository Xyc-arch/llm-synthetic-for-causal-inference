import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algs.aipw import estimate_aipw_df
from algs.ipw import estimate_ipw_df
from algs.outcome_regression import estimate_outcome_regression_df
from algs.tmle import estimate_tmle_df

COVARIATES = ["W1", "W2", "W3", "W4", "W5", "W6"]
OUTCOME_COL = "Y"
TREATMENT_COL = "A"
THRESHOLD = 0.001
CLIP_MIN = 1e-6
FLIP_RATES = [0.05, 0.10, 0.20]

TRUTH_PATH = os.path.join(PROJECT_ROOT, "truth.json")

ORIG_DATASETS = {
    f"data_{i}": os.path.join(PROJECT_ROOT, "positivity", "data", f"data_{i}.csv")
    for i in [1, 2, 3, 4, 5]
}

SOURCES = {
    "llm": {
        "syn_cov": os.path.join(PROJECT_ROOT, "llm_data", "syn_clean.csv"),
        "syn_hyb": os.path.join(PROJECT_ROOT, "llm_data", "syn_hybrid.csv"),
    },
    "gan": {
        "syn_cov": os.path.join(PROJECT_ROOT, "gan_data", "syn_clean.csv"),
        "syn_hyb": os.path.join(PROJECT_ROOT, "gan_data", "syn_hybrid.csv"),
    },
}

RESULTS_DIR = os.path.join(PROJECT_ROOT, "positivity", "results")
AUG_DIR = os.path.join(PROJECT_ROOT, "positivity", "augmented_qhyb_flip")


def load_truth():
    with open(TRUTH_PATH, "r") as f:
        truth = json.load(f)
    return float(truth["ate_true"]), truth


def propensity_truncation_level(n):
    if n <= 1:
        return 0.25
    delta_n = 5.0 / (np.sqrt(n) * np.log(n))
    delta_n = max(CLIP_MIN, float(delta_n))
    return min(delta_n, 0.25)


def fit_ps_on_orig(orig):
    rf_g = RandomForestClassifier(random_state=42)
    rf_g.fit(orig[COVARIATES], orig[TREATMENT_COL])
    return rf_g


def fit_q_on_hybrid(hyb):
    rf_q = RandomForestClassifier(random_state=42)
    rf_q.fit(hyb[COVARIATES + [TREATMENT_COL]], hyb[OUTCOME_COL])
    return rf_q


def compute_ps(orig, rf_g):
    delta_n = propensity_truncation_level(len(orig))
    ps_hat_raw = rf_g.predict_proba(orig[COVARIATES])[:, 1]
    ps_hat_trunc = np.clip(ps_hat_raw, delta_n, 1 - delta_n)

    out = orig.copy()
    out["ps_hat_raw"] = ps_hat_raw
    out["ps_hat"] = ps_hat_trunc
    out["g_trunc_level"] = float(delta_n)
    return out


def standardize_for_matching(orig, syn):
    means = orig[COVARIATES].mean()
    stds = orig[COVARIATES].std().replace(0, 1.0)

    orig_std = orig.copy()
    syn_std = syn.copy()

    orig_std[COVARIATES] = (orig[COVARIATES] - means) / stds
    syn_std[COVARIATES] = (syn[COVARIATES] - means) / stds

    return orig_std, syn_std


def flip_hybrid_labels_on_the_fly(syn_hyb, flip_rate, seed):
    df = syn_hyb.copy()
    df[OUTCOME_COL] = df[OUTCOME_COL].round().astype(int)
    n = len(df)
    n_flip = int(round(flip_rate * n))
    rng = np.random.default_rng(seed)
    flip_idx = rng.choice(df.index.to_numpy(), size=n_flip, replace=False)
    df.loc[flip_idx, OUTCOME_COL] = 1 - df.loc[flip_idx, OUTCOME_COL]
    return df, int(n_flip)


def predict_y_from_q(rf_q, df):
    probs = np.clip(
        rf_q.predict_proba(df[COVARIATES + [TREATMENT_COL]])[:, 1],
        CLIP_MIN,
        1 - CLIP_MIN,
    )
    y = np.random.binomial(1, probs)
    return probs, y


def build_matched_rows(orig, syn_cov, threshold):
    orig_std, syn_std = standardize_for_matching(orig, syn_cov)

    matched_rows = []
    used_idx = set()

    # Use raw propensity for rare-region detection and matching
    extreme_low = orig[orig["ps_hat_raw"] < threshold].copy()
    extreme_high = orig[orig["ps_hat_raw"] > (1 - threshold)].copy()

    for _, row in orig_std[orig_std["ps_hat_raw"] < threshold].iterrows():
        candidates = syn_std[~syn_std.index.isin(used_idx)]
        if candidates.empty:
            continue
        distances = np.sqrt(((candidates[COVARIATES] - row[COVARIATES]) ** 2).sum(axis=1))
        best_idx = distances.idxmin()

        new_row = syn_cov.loc[best_idx, COVARIATES].copy()
        new_row[TREATMENT_COL] = 1
        matched_rows.append(new_row.to_dict())
        used_idx.add(best_idx)

    for _, row in orig_std[orig_std["ps_hat_raw"] > (1 - threshold)].iterrows():
        candidates = syn_std[~syn_std.index.isin(used_idx)]
        if candidates.empty:
            continue
        distances = np.sqrt(((candidates[COVARIATES] - row[COVARIATES]) ** 2).sum(axis=1))
        best_idx = distances.idxmin()

        new_row = syn_cov.loc[best_idx, COVARIATES].copy()
        new_row[TREATMENT_COL] = 0
        matched_rows.append(new_row.to_dict())
        used_idx.add(best_idx)

    matched_df = (
        pd.DataFrame(matched_rows)
        if len(matched_rows) > 0
        else pd.DataFrame(columns=COVARIATES + [TREATMENT_COL])
    )

    diagnostics = {
        "n_orig": int(len(orig)),
        "g_trunc_level": float(orig["g_trunc_level"].iloc[0]),
        "threshold": float(threshold),
        "extreme_low_count": int(len(extreme_low)),
        "extreme_high_count": int(len(extreme_high)),
        "matched_low_added_as_treated": int(len(extreme_low)),
        "matched_high_added_as_control": int(len(extreme_high)),
        "used_synthetic_rows": int(len(used_idx)),
        "min_ps_hat_raw": float(orig["ps_hat_raw"].min()),
        "max_ps_hat_raw": float(orig["ps_hat_raw"].max()),
        "min_ps_hat_trunc": float(orig["ps_hat"].min()),
        "max_ps_hat_trunc": float(orig["ps_hat"].max()),
    }

    return matched_df, diagnostics


def make_augmented_dataset(orig_path, syn_cov_path, syn_hyb_path, out_csv, flip_rate, threshold=THRESHOLD):
    orig = pd.read_csv(orig_path).copy()
    syn_cov = pd.read_csv(syn_cov_path).copy()
    syn_hyb = pd.read_csv(syn_hyb_path).copy()

    orig[OUTCOME_COL] = orig[OUTCOME_COL].round().astype(int)
    orig[TREATMENT_COL] = orig[TREATMENT_COL].round().astype(int)

    syn_cov = syn_cov[COVARIATES].copy()
    syn_hyb = syn_hyb[COVARIATES + [TREATMENT_COL, OUTCOME_COL]].copy()
    syn_hyb[TREATMENT_COL] = syn_hyb[TREATMENT_COL].round().astype(int)
    syn_hyb[OUTCOME_COL] = syn_hyb[OUTCOME_COL].round().astype(int)

    flip_seed = int(round(flip_rate * 1000)) + 42
    syn_hyb_flipped, n_flipped = flip_hybrid_labels_on_the_fly(
        syn_hyb,
        flip_rate=flip_rate,
        seed=flip_seed,
    )

    rf_g = fit_ps_on_orig(orig)
    rf_q_hyb = fit_q_on_hybrid(syn_hyb_flipped)

    orig_with_ps = compute_ps(orig, rf_g)
    matched_df, diag_match = build_matched_rows(orig_with_ps, syn_cov, threshold=threshold)

    if len(matched_df) == 0:
        augmented = orig[COVARIATES + [TREATMENT_COL, OUTCOME_COL]].copy()
    else:
        probs, y = predict_y_from_q(rf_q_hyb, matched_df.copy())
        matched_df["y_prob_qhyb_flip"] = probs
        matched_df[OUTCOME_COL] = y.astype(int)

        augmented = pd.concat(
            [
                orig[COVARIATES + [TREATMENT_COL, OUTCOME_COL]],
                matched_df[COVARIATES + [TREATMENT_COL, OUTCOME_COL]],
            ],
            ignore_index=True,
        )

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    augmented.to_csv(out_csv, index=False)

    diagnostics = {
        "orig_file": orig_path,
        "syn_cov_file": syn_cov_path,
        "syn_hyb_file": syn_hyb_path,
        "output_file": out_csv,
        "flip_rate": float(flip_rate),
        "flip_seed": int(flip_seed),
        "n_flipped_in_hybrid_training": int(n_flipped),
        "n_augmented": int(len(augmented)),
        "n_added": int(len(augmented) - len(orig)),
    }
    diagnostics.update(diag_match)

    return augmented, diagnostics


def evaluate_estimators(df, ate_true):
    estimators = {
        "aipw": estimate_aipw_df,
        "ipw": estimate_ipw_df,
        "outcome_regression": estimate_outcome_regression_df,
        "tmle": estimate_tmle_df,
    }

    delta_n = propensity_truncation_level(len(df))

    out = {}
    for name, fn in estimators.items():
        kwargs = {
            "covariates": COVARIATES,
            "outcome_col": OUTCOME_COL,
            "treatment_col": TREATMENT_COL,
            "random_state": 42,
        }

        if name in {"aipw", "ipw", "tmle"}:
            kwargs["clip_min"] = delta_n

        est = fn(df.copy(), **kwargs)
        bias = float(est - ate_true)

        out[name] = {
            "estimate": float(est),
            "ate_true": float(ate_true),
            "bias": bias,
            "abs_bias": float(abs(bias)),
            "g_trunc_level": float(delta_n),
        }
    return out


def main():
    ate_true, truth = load_truth()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(AUG_DIR, exist_ok=True)

    all_results = {
        "truth": truth,
        "threshold": THRESHOLD,
        "flip_rates": FLIP_RATES,
        "g_bounds_rule": "delta_n = 5 / (sqrt(n) * log(n)); g used in estimators is in [delta_n, 1-delta_n]",
        "matching_rule": "matching/extreme-region detection uses raw propensity ps_hat_raw before truncation",
        "datasets": {},
    }

    for data_name, orig_path in ORIG_DATASETS.items():
        all_results["datasets"][data_name] = {}

        for syn_name, paths in SOURCES.items():
            all_results["datasets"][data_name][syn_name] = {}

            for flip_rate in FLIP_RATES:
                flip_tag = f"flip_{int(round(flip_rate * 100))}"
                out_csv = os.path.join(AUG_DIR, f"{data_name}_{syn_name}_{flip_tag}_pair_qhyb.csv")

                augmented_df, diagnostics = make_augmented_dataset(
                    orig_path=orig_path,
                    syn_cov_path=paths["syn_cov"],
                    syn_hyb_path=paths["syn_hyb"],
                    out_csv=out_csv,
                    flip_rate=flip_rate,
                    threshold=THRESHOLD,
                )

                estimator_results = evaluate_estimators(augmented_df, ate_true)

                all_results["datasets"][data_name][syn_name][flip_tag] = {
                    "diagnostics": diagnostics,
                    "estimators": estimator_results,
                }

                print(f"Finished {data_name} | {syn_name} | {flip_tag}")

    out_json = os.path.join(RESULTS_DIR, "pair_qhyb_flip_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"Saved results to {out_json}")


if __name__ == "__main__":
    main()