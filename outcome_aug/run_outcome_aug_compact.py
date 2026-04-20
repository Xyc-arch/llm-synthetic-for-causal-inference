import os
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logit, expit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

PROJECT_ROOT = "/home/ubuntu/syn_causal"
DATA_DIR = os.path.join(PROJECT_ROOT, "outcome_aug", "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outcome_aug", "results")

COVARIATES = ["W1", "W2", "W3", "W4", "W5", "W6"]
A_COL = "A"
Y_COL = "Y"

DATASETS = {
    f"data_{i}": os.path.join(DATA_DIR, f"data_{i}.csv")
    for i in [1, 2, 3, 4, 5]
}

SYN_SOURCES = {
    "llm": os.path.join(PROJECT_ROOT, "llm_data", "syn_hybrid.csv"),
    "gan": os.path.join(PROJECT_ROOT, "gan_data", "syn_hybrid.csv"),
}

TRUTH_PATH = os.path.join(DATA_DIR, "truth.json")

# Keep only 5% and 10%
FLIP_RATES = [0.05, 0.10]

USE_KNOWN_RCT_G = False
CLIP_MIN = 1e-6
SOFT_Y_COL = "Y_soft_corrected"

# Speed controls
N_ESTIMATORS = 100
MIN_SAMPLES_LEAF = 5
N_JOBS = -1
RUN_SOFT_REFIT_BRANCH = True


def clip_probs(x, lower=CLIP_MIN):
    upper = 1.0 - lower
    return np.clip(np.asarray(x, dtype=float), lower, upper)


def propensity_truncation_level(n):
    if n <= 1:
        return 0.25
    delta_n = 5.0 / (np.sqrt(n) * np.log(n))
    delta_n = max(CLIP_MIN, float(delta_n))
    return min(delta_n, 0.25)


def truncate_g(g, n):
    delta_n = propensity_truncation_level(n)
    return clip_probs(g, lower=delta_n)


def load_truth():
    with open(TRUTH_PATH, "r") as f:
        return json.load(f)


def flip_labels(df, flip_rate, seed):
    out = df.copy()
    out[Y_COL] = out[Y_COL].round().astype(int)

    n = len(out)
    n_flip = int(round(flip_rate * n))

    rng = np.random.default_rng(seed)
    flip_idx = rng.choice(out.index.to_numpy(), size=n_flip, replace=False)
    out.loc[flip_idx, Y_COL] = 1 - out.loc[flip_idx, Y_COL]

    return out, n_flip


def precompute_flipped_syn_cache(syn_cache):
    flipped = {}
    for syn_name, syn_df in syn_cache.items():
        flipped[syn_name] = {}
        for flip_rate in FLIP_RATES:
            flip_seed = int(round(flip_rate * 1000)) + 42
            syn_flip, n_flipped = flip_labels(syn_df, flip_rate=flip_rate, seed=flip_seed)
            key = f"{syn_name}_flip_{int(round(flip_rate * 100))}"
            flipped[syn_name][key] = {
                "df": syn_flip,
                "flip_rate": float(flip_rate),
                "flip_seed": int(flip_seed),
                "n_flipped": int(n_flipped),
            }
    return flipped


# -----------------------------
# Propensity model
# -----------------------------
def fit_g_model(df, random_state=42):
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_jobs=N_JOBS,
        random_state=random_state,
    )
    model.fit(df[COVARIATES], df[A_COL].round().astype(int))
    return model


def get_g_probs_from_model(df, g_model):
    n = len(df)
    if USE_KNOWN_RCT_G:
        g = np.full(n, 0.5, dtype=float)
        return truncate_g(g, n)
    g = g_model.predict_proba(df[COVARIATES])[:, 1]
    return truncate_g(g, n)


# -----------------------------
# Outcome models
# -----------------------------
def fit_q_model_binary(df, random_state=42):
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_jobs=N_JOBS,
        random_state=random_state,
    )
    model.fit(df[COVARIATES + [A_COL]], df[Y_COL].round().astype(int))
    return model


def predict_q_binary(model, df):
    x_obs = df[COVARIATES + [A_COL]].copy()
    q_aw = clip_probs(model.predict_proba(x_obs)[:, 1])

    x1 = df[COVARIATES].copy()
    x1[A_COL] = 1
    q1 = clip_probs(model.predict_proba(x1)[:, 1])

    x0 = df[COVARIATES].copy()
    x0[A_COL] = 0
    q0 = clip_probs(model.predict_proba(x0)[:, 1])

    return q_aw, q1, q0


def fit_q_model_soft(df, soft_y_col=SOFT_Y_COL, random_state=42):
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_jobs=N_JOBS,
        random_state=random_state,
    )
    model.fit(df[COVARIATES + [A_COL]], df[soft_y_col].astype(float))
    return model


def predict_q_soft(model, df):
    x_obs = df[COVARIATES + [A_COL]].copy()
    q_aw = clip_probs(model.predict(x_obs))

    x1 = df[COVARIATES].copy()
    x1[A_COL] = 1
    q1 = clip_probs(model.predict(x1))

    x0 = df[COVARIATES].copy()
    x0[A_COL] = 0
    q0 = clip_probs(model.predict(x0))

    return q_aw, q1, q0


# -----------------------------
# Estimators
# -----------------------------
def estimate_or_from_q(q1, q0):
    return float(np.mean(q1 - q0))


def estimate_aipw_from_q_g(df, q_aw, q1, q0, g):
    a = df[A_COL].to_numpy().astype(int)
    y = df[Y_COL].to_numpy().astype(int)
    g = truncate_g(g, len(df))

    val = np.mean(
        q1 - q0
        + a * (y - q_aw) / g
        - (1 - a) * (y - q_aw) / (1 - g)
    )
    return float(val)


# -----------------------------
# TMLE targeting
# -----------------------------
def tmle_target(df, q_aw, q1, q0, g):
    a = df[A_COL].to_numpy().astype(int)
    y = df[Y_COL].to_numpy().astype(int)
    g = truncate_g(g, len(df))

    h_aw = a / g - (1 - a) / (1 - g)
    h1 = 1 / g
    h0 = -1 / (1 - g)

    model = sm.GLM(
        y,
        h_aw.reshape(-1, 1),
        family=sm.families.Binomial(),
        offset=logit(clip_probs(q_aw)),
    )
    fit = model.fit()
    eps = float(fit.params[0])

    q1_star = clip_probs(expit(logit(clip_probs(q1)) + eps * h1))
    q0_star = clip_probs(expit(logit(clip_probs(q0)) + eps * h0))
    q_aw_star = clip_probs(expit(logit(clip_probs(q_aw)) + eps * h_aw))

    psi_tmle = float(np.mean(q1_star - q0_star))

    return {
        "epsilon": eps,
        "q_aw_star": q_aw_star,
        "q1_star": q1_star,
        "q0_star": q0_star,
        "tmle": psi_tmle,
        "g_trunc_level": propensity_truncation_level(len(df)),
    }


def apply_tmle_update_with_fixed_epsilon(df, q_aw, q1, q0, g, epsilon):
    a = df[A_COL].to_numpy().astype(int)
    g = truncate_g(g, len(df))

    h_aw = a / g - (1 - a) / (1 - g)
    h1 = 1 / g
    h0 = -1 / (1 - g)

    q1_star = clip_probs(expit(logit(clip_probs(q1)) + epsilon * h1))
    q0_star = clip_probs(expit(logit(clip_probs(q0)) + epsilon * h0))
    q_aw_star = clip_probs(expit(logit(clip_probs(q_aw)) + epsilon * h_aw))

    return {
        "q_aw_star": q_aw_star,
        "q1_star": q1_star,
        "q0_star": q0_star,
    }


# -----------------------------
# Evaluation branches
# -----------------------------
def evaluate_orig_only(orig_df, g_model_orig):
    q_model = fit_q_model_binary(orig_df)
    g_orig = get_g_probs_from_model(orig_df, g_model_orig)

    q_aw_orig, q1_orig, q0_orig = predict_q_binary(q_model, orig_df)

    or_raw = estimate_or_from_q(q1_orig, q0_orig)
    aipw_raw = estimate_aipw_from_q_g(orig_df, q_aw_orig, q1_orig, q0_orig, g_orig)

    tmle_res = tmle_target(orig_df, q_aw_orig, q1_orig, q0_orig, g_orig)

    or_corr = float(np.mean(tmle_res["q1_star"] - tmle_res["q0_star"]))
    aipw_corr = estimate_aipw_from_q_g(
        orig_df,
        tmle_res["q_aw_star"],
        tmle_res["q1_star"],
        tmle_res["q0_star"],
        g_orig,
    )

    return {
        "raw_plugin": {
            "or": or_raw,
            "aipw": aipw_raw,
        },
        "corrected_plugin": {
            "or": or_corr,
            "aipw": aipw_corr,
            "tmle": tmle_res["tmle"],
            "epsilon": tmle_res["epsilon"],
            "g_trunc_level": tmle_res["g_trunc_level"],
        },
    }


def evaluate_pooled_q(orig_df, syn_df, g_model_orig):
    pooled_df = pd.concat(
        [orig_df[COVARIATES + [A_COL, Y_COL]], syn_df[COVARIATES + [A_COL, Y_COL]]],
        ignore_index=True,
    )

    q_model_pooled = fit_q_model_binary(pooled_df)

    g_orig = get_g_probs_from_model(orig_df, g_model_orig)
    q_aw_orig, q1_orig, q0_orig = predict_q_binary(q_model_pooled, orig_df)

    or_nocorr = estimate_or_from_q(q1_orig, q0_orig)
    aipw_nocorr = estimate_aipw_from_q_g(orig_df, q_aw_orig, q1_orig, q0_orig, g_orig)

    tmle_res_orig = tmle_target(orig_df, q_aw_orig, q1_orig, q0_orig, g_orig)

    or_corr_orig = float(np.mean(tmle_res_orig["q1_star"] - tmle_res_orig["q0_star"]))
    aipw_corr_orig = estimate_aipw_from_q_g(
        orig_df,
        tmle_res_orig["q_aw_star"],
        tmle_res_orig["q1_star"],
        tmle_res_orig["q0_star"],
        g_orig,
    )

    # pooled average with orig epsilon
    g_pooled = get_g_probs_from_model(pooled_df, g_model_orig)
    q_aw_pooled, q1_pooled, q0_pooled = predict_q_binary(q_model_pooled, pooled_df)

    transported_corr = apply_tmle_update_with_fixed_epsilon(
        pooled_df,
        q_aw_pooled,
        q1_pooled,
        q0_pooled,
        g_pooled,
        epsilon=tmle_res_orig["epsilon"],
    )

    or_corr_pooled_avg = float(np.mean(transported_corr["q1_star"] - transported_corr["q0_star"]))
    aipw_corr_pooled_avg = estimate_aipw_from_q_g(
        pooled_df,
        transported_corr["q_aw_star"],
        transported_corr["q1_star"],
        transported_corr["q0_star"],
        g_pooled,
    )

    out = {
        "without_correction_plugin": {
            "or": or_nocorr,
            "aipw": aipw_nocorr,
        },
        "with_correction_plugin_orig_avg": {
            "or": or_corr_orig,
            "aipw": aipw_corr_orig,
            "tmle": tmle_res_orig["tmle"],
            "epsilon": tmle_res_orig["epsilon"],
            "g_trunc_level": tmle_res_orig["g_trunc_level"],
        },
        "with_correction_plugin_pooled_avg": {
            "or": or_corr_pooled_avg,
            "aipw": aipw_corr_pooled_avg,
        },
    }

    if RUN_SOFT_REFIT_BRANCH:
        syn_only = syn_df[COVARIATES + [A_COL, Y_COL]].copy()
        q_aw_syn, q1_syn, q0_syn = predict_q_binary(q_model_pooled, syn_only)
        g_syn = get_g_probs_from_model(syn_only, g_model_orig)

        syn_corrected = apply_tmle_update_with_fixed_epsilon(
            syn_only,
            q_aw_syn,
            q1_syn,
            q0_syn,
            g_syn,
            epsilon=tmle_res_orig["epsilon"],
        )

        orig_soft = orig_df[COVARIATES + [A_COL, Y_COL]].copy()
        orig_soft[SOFT_Y_COL] = orig_soft[Y_COL].astype(float)

        syn_soft = syn_only[COVARIATES + [A_COL, Y_COL]].copy()
        syn_soft[SOFT_Y_COL] = syn_corrected["q_aw_star"]

        corrected_soft_train = pd.concat([orig_soft, syn_soft], ignore_index=True)

        q_model_soft = fit_q_model_soft(corrected_soft_train, soft_y_col=SOFT_Y_COL)

        q_aw_soft_orig, q1_soft_orig, q0_soft_orig = predict_q_soft(q_model_soft, orig_df)
        or_soft_orig_avg = estimate_or_from_q(q1_soft_orig, q0_soft_orig)
        aipw_soft_orig_avg = estimate_aipw_from_q_g(
            orig_df,
            q_aw_soft_orig,
            q1_soft_orig,
            q0_soft_orig,
            g_orig,
        )

        q_aw_soft_pooled, q1_soft_pooled, q0_soft_pooled = predict_q_soft(q_model_soft, pooled_df)
        or_soft_pooled_avg = estimate_or_from_q(q1_soft_pooled, q0_soft_pooled)
        aipw_soft_pooled_avg = estimate_aipw_from_q_g(
            pooled_df,
            q_aw_soft_pooled,
            q1_soft_pooled,
            q0_soft_pooled,
            g_pooled,
        )

        out["refit_on_corrected_synth_y"] = {
            "orig_avg": {
                "or": or_soft_orig_avg,
                "aipw": aipw_soft_orig_avg,
            },
            "pooled_avg": {
                "or": or_soft_pooled_avg,
                "aipw": aipw_soft_pooled_avg,
            },
        }

    return out


def summarize_estimates(estimates, truth):
    arr = np.array(estimates, dtype=float)
    errors = arr - truth

    return {
        "n_datasets": int(len(arr)),
        "estimates": arr.tolist(),
        "mean": float(np.mean(arr)),
        "bias": float(np.mean(arr) - truth),
        "abs_bias": float(abs(np.mean(arr) - truth)),
        "var": float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "mse": float(np.mean(errors ** 2)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    truth = load_truth()
    ate_true = float(truth["ate_true"])

    syn_cache = {}
    for name, path in SYN_SOURCES.items():
        df = pd.read_csv(path)
        df[A_COL] = df[A_COL].round().astype(int)
        df[Y_COL] = df[Y_COL].round().astype(int)
        syn_cache[name] = df[COVARIATES + [A_COL, Y_COL]].copy()

    flipped_syn_cache = precompute_flipped_syn_cache(syn_cache)

    collected = {
        "orig": {
            "raw_plugin": {"or": [], "aipw": []},
            "corrected_plugin": {"or": [], "aipw": [], "tmle": []},
        }
    }

    for syn_name in SYN_SOURCES:
        for flip_rate in FLIP_RATES:
            key = f"{syn_name}_flip_{int(round(flip_rate * 100))}"
            collected[key] = {
                "without_correction_plugin": {"or": [], "aipw": []},
                "with_correction_plugin_orig_avg": {"or": [], "aipw": [], "tmle": []},
                "with_correction_plugin_pooled_avg": {"or": [], "aipw": []},
            }
            if RUN_SOFT_REFIT_BRANCH:
                collected[key]["refit_on_corrected_synth_y"] = {
                    "orig_avg": {"or": [], "aipw": []},
                    "pooled_avg": {"or": [], "aipw": []},
                }

    for dataset_name, dataset_path in DATASETS.items():
        orig_df = pd.read_csv(dataset_path)
        orig_df[A_COL] = orig_df[A_COL].round().astype(int)
        orig_df[Y_COL] = orig_df[Y_COL].round().astype(int)

        # fit g once per original dataset
        g_model_orig = fit_g_model(orig_df)

        orig_res = evaluate_orig_only(orig_df, g_model_orig)

        for est_name, val in orig_res["raw_plugin"].items():
            collected["orig"]["raw_plugin"][est_name].append(val)

        for est_name, val in orig_res["corrected_plugin"].items():
            if est_name in collected["orig"]["corrected_plugin"]:
                collected["orig"]["corrected_plugin"][est_name].append(val)

        for syn_name in SYN_SOURCES:
            for flip_rate in FLIP_RATES:
                key = f"{syn_name}_flip_{int(round(flip_rate * 100))}"
                syn_flip = flipped_syn_cache[syn_name][key]["df"]

                pooled_res = evaluate_pooled_q(orig_df, syn_flip, g_model_orig)

                for est_name, val in pooled_res["without_correction_plugin"].items():
                    collected[key]["without_correction_plugin"][est_name].append(val)

                for est_name, val in pooled_res["with_correction_plugin_orig_avg"].items():
                    if est_name in collected[key]["with_correction_plugin_orig_avg"]:
                        collected[key]["with_correction_plugin_orig_avg"][est_name].append(val)

                for est_name, val in pooled_res["with_correction_plugin_pooled_avg"].items():
                    collected[key]["with_correction_plugin_pooled_avg"][est_name].append(val)

                if RUN_SOFT_REFIT_BRANCH:
                    for avg_name in ["orig_avg", "pooled_avg"]:
                        for est_name, val in pooled_res["refit_on_corrected_synth_y"][avg_name].items():
                            collected[key]["refit_on_corrected_synth_y"][avg_name][est_name].append(val)

                print(
                    f"finished {dataset_name} | {key} | "
                    f"g_trunc={pooled_res['with_correction_plugin_orig_avg']['g_trunc_level']:.6f}"
                )

    compact = {
        "truth": truth,
        "flip_rates": FLIP_RATES,
        "use_known_rct_g": USE_KNOWN_RCT_G,
        "g_bounds_rule": "delta_n = 5 / (sqrt(n) * log(n)); g in [delta_n, 1-delta_n]",
        "rf_settings": {
            "n_estimators": N_ESTIMATORS,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "n_jobs": N_JOBS,
        },
        "soft_y_branch_enabled": RUN_SOFT_REFIT_BRANCH,
        "summary": {},
    }

    compact["summary"]["orig"] = {
        "raw_plugin": {
            est_name: summarize_estimates(vals, ate_true)
            for est_name, vals in collected["orig"]["raw_plugin"].items()
        },
        "corrected_plugin": {
            est_name: summarize_estimates(vals, ate_true)
            for est_name, vals in collected["orig"]["corrected_plugin"].items()
        },
    }

    for syn_name in SYN_SOURCES:
        for flip_rate in FLIP_RATES:
            key = f"{syn_name}_flip_{int(round(flip_rate * 100))}"
            compact["summary"][key] = {
                "without_correction_plugin": {
                    est_name: summarize_estimates(vals, ate_true)
                    for est_name, vals in collected[key]["without_correction_plugin"].items()
                },
                "with_correction_plugin_orig_avg": {
                    est_name: summarize_estimates(vals, ate_true)
                    for est_name, vals in collected[key]["with_correction_plugin_orig_avg"].items()
                },
                "with_correction_plugin_pooled_avg": {
                    est_name: summarize_estimates(vals, ate_true)
                    for est_name, vals in collected[key]["with_correction_plugin_pooled_avg"].items()
                },
            }

            if RUN_SOFT_REFIT_BRANCH:
                compact["summary"][key]["refit_on_corrected_synth_y"] = {
                    "orig_avg": {
                        est_name: summarize_estimates(vals, ate_true)
                        for est_name, vals in collected[key]["refit_on_corrected_synth_y"]["orig_avg"].items()
                    },
                    "pooled_avg": {
                        est_name: summarize_estimates(vals, ate_true)
                        for est_name, vals in collected[key]["refit_on_corrected_synth_y"]["pooled_avg"].items()
                    },
                }

    out_path = os.path.join(RESULTS_DIR, "outcome_aug_compact_transport_corrected_y_fast.json")
    with open(out_path, "w") as f:
        json.dump(compact, f, indent=4)

    print(f"saved compact results to {out_path}")


if __name__ == "__main__":
    main()