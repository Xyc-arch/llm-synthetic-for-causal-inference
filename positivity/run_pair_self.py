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
TRUTH_PATH = os.path.join(PROJECT_ROOT, "truth.json")

ORIG_DATASETS = {
    f"data_{i}": os.path.join(PROJECT_ROOT, "positivity", "data", f"data_{i}.csv")
    for i in [1, 2, 3, 4, 5]
}

SYN_SOURCES = {
    "llm": os.path.join(PROJECT_ROOT, "llm_data", "syn_clean.csv"),
    "gan": os.path.join(PROJECT_ROOT, "gan_data", "syn_clean.csv"),
}

RESULTS_DIR = os.path.join(PROJECT_ROOT, "positivity", "results")
AUG_DIR = os.path.join(PROJECT_ROOT, "positivity", "augmented_self_supervised")


def load_truth():
    with open(TRUTH_PATH, "r") as f:
        truth = json.load(f)
    return float(truth["ate_true"]), truth


def propensity_truncation_level(n):
    if n <= 1:
        return 0.25
    delta_n = 5.0 / (np.sqrt(n) * np.log(n))
    delta_n = max(1e-6, float(delta_n))
    return min(delta_n, 0.25)


def fit_models_on_orig(orig):
    rf_g = RandomForestClassifier(random_state=42)
    rf_g.fit(orig[COVARIATES], orig[TREATMENT_COL])

    rf_q = RandomForestClassifier(random_state=42)
    rf_q.fit(orig[COVARIATES + [TREATMENT_COL]], orig[OUTCOME_COL])

    return rf_g, rf_q


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


def predict_y_from_q(rf_q, df):
    probs = rf_q.predict_proba(df[COVARIATES + [TREATMENT_COL]])[:, 1]
    y = np.random.binomial(1, probs)
    return probs, y


def make_self_supervised_pair(orig_path, syn_path, out_csv, threshold=THRESHOLD):
    orig = pd.read_csv(orig_path).copy()
    syn = pd.read_csv(syn_path).copy()

    orig[OUTCOME_COL] = orig[OUTCOME_COL].round().astype(int)
    orig[TREATMENT_COL] = orig[TREATMENT_COL].round().astype(int)

    syn = syn[COVARIATES].copy()

    rf_g, rf_q = fit_models_on_orig(orig)
    orig = compute_ps(orig, rf_g)
    delta_n = float(orig["g_trunc_level"].iloc[0])

    # Use RAW propensity scores for identifying rare/extreme regions
    extreme_low = orig[orig["ps_hat_raw"] < threshold].copy()
    extreme_high = orig[orig["ps_hat_raw"] > (1 - threshold)].copy()

    orig_std, syn_std = standardize_for_matching(orig, syn)

    matched_rows = []
    used_idx = set()

    # Match using RAW propensity score regime membership
    for _, row in orig_std[orig_std["ps_hat_raw"] < threshold].iterrows():
        candidates = syn_std[~syn_std.index.isin(used_idx)]
        if candidates.empty:
            continue
        distances = np.sqrt(((candidates[COVARIATES] - row[COVARIATES]) ** 2).sum(axis=1))
        best_idx = distances.idxmin()

        new_row = syn.loc[best_idx, COVARIATES].copy()
        new_row[TREATMENT_COL] = 1
        matched_rows.append(new_row.to_dict())
        used_idx.add(best_idx)

    for _, row in orig_std[orig_std["ps_hat_raw"] > (1 - threshold)].iterrows():
        candidates = syn_std[~syn_std.index.isin(used_idx)]
        if candidates.empty:
            continue
        distances = np.sqrt(((candidates[COVARIATES] - row[COVARIATES]) ** 2).sum(axis=1))
        best_idx = distances.idxmin()

        new_row = syn.loc[best_idx, COVARIATES].copy()
        new_row[TREATMENT_COL] = 0
        matched_rows.append(new_row.to_dict())
        used_idx.add(best_idx)

    if len(matched_rows) > 0:
        matched_df = pd.DataFrame(matched_rows)
        y_prob, y_draw = predict_y_from_q(rf_q, matched_df)
        matched_df["y_prob_orig_q"] = y_prob
        matched_df[OUTCOME_COL] = y_draw.astype(int)

        augmented = pd.concat(
            [
                orig[COVARIATES + [TREATMENT_COL, OUTCOME_COL]],
                matched_df[COVARIATES + [TREATMENT_COL, OUTCOME_COL]],
            ],
            ignore_index=True,
        )
    else:
        matched_df = pd.DataFrame(columns=COVARIATES + [TREATMENT_COL, OUTCOME_COL, "y_prob_orig_q"])
        augmented = orig[COVARIATES + [TREATMENT_COL, OUTCOME_COL]].copy()

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    augmented.to_csv(out_csv, index=False)

    diagnostics = {
        "orig_file": orig_path,
        "syn_file": syn_path,
        "output_file": out_csv,
        "g_trunc_level": delta_n,
        "n_orig": int(len(orig)),
        "n_augmented": int(len(augmented)),
        "n_added": int(len(augmented) - len(orig)),
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
        "g_bounds_rule": "delta_n = 5 / (sqrt(n) * log(n)); g used in estimators is in [delta_n, 1-delta_n]",
        "matching_rule": "matching/extreme-region detection uses raw propensity ps_hat_raw before truncation",
        "datasets": {},
    }

    for data_name, orig_path in ORIG_DATASETS.items():
        all_results["datasets"][data_name] = {}

        for syn_name, syn_path in SYN_SOURCES.items():
            out_csv = os.path.join(AUG_DIR, f"{data_name}_{syn_name}_self_supervised_pair.csv")

            augmented_df, diagnostics = make_self_supervised_pair(
                orig_path=orig_path,
                syn_path=syn_path,
                out_csv=out_csv,
                threshold=THRESHOLD,
            )

            estimator_results = evaluate_estimators(augmented_df, ate_true)

            all_results["datasets"][data_name][syn_name] = {
                "diagnostics": diagnostics,
                "estimators": estimator_results,
            }

            print(f"Finished {data_name} with {syn_name}")

    out_json = os.path.join(RESULTS_DIR, "self_supervised_pair_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"Saved results to {out_json}")


if __name__ == "__main__":
    main()