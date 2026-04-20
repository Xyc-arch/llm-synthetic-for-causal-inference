import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.special import logit, expit
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

COVARIATES = ["W1", "W2", "W3", "W4", "W5", "W6"]
OUTCOME_COL = "Y"
TREATMENT_COL = "A"
THRESHOLD = 0.001
CLIP_MIN = 1e-6

TRUTH_PATH = os.path.join(PROJECT_ROOT, "truth.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "positivity", "results")
PAIR_DIR = os.path.join(PROJECT_ROOT, "positivity", "augmented_tmle_analysis")

ORIG_DATASETS = {
    f"data_{i}": os.path.join(PROJECT_ROOT, "positivity", "data", f"data_{i}.csv")
    for i in [1, 2, 3, 4, 5]
}

PAIR_SOURCES = {
    "llm_self": os.path.join(PROJECT_ROOT, "positivity", "augmented_self_supervised"),
    "gan_self": os.path.join(PROJECT_ROOT, "positivity", "augmented_self_supervised"),
    "llm_qhyb": os.path.join(PROJECT_ROOT, "positivity", "augmented_qhyb_pair"),
    "gan_qhyb": os.path.join(PROJECT_ROOT, "positivity", "augmented_qhyb_pair"),
}


def load_truth():
    with open(TRUTH_PATH, "r") as f:
        truth = json.load(f)
    return float(truth["ate_true"]), truth


def trunc_level(n):
    return 5.0 / (np.sqrt(n) * np.log(n))


def clip_probs(x, low=CLIP_MIN):
    return np.clip(x, low, 1 - low)


def fit_ps(df, random_state=42):
    rf = RandomForestClassifier(random_state=random_state)
    rf.fit(df[COVARIATES], df[TREATMENT_COL].round().astype(int))
    g = clip_probs(rf.predict_proba(df[COVARIATES])[:, 1])
    return rf, g


def fit_q(df, random_state=42):
    rf = RandomForestClassifier(random_state=random_state)
    rf.fit(df[COVARIATES + [TREATMENT_COL]], df[OUTCOME_COL].round().astype(int))
    return rf


def tmle_estimate_df(df, truncate=False, random_state=42):
    data = df.copy()
    data[OUTCOME_COL] = data[OUTCOME_COL].round().astype(int)
    data[TREATMENT_COL] = data[TREATMENT_COL].round().astype(int)

    q_model = fit_q(data, random_state=random_state)
    _, g = fit_ps(data, random_state=random_state)

    if truncate:
        cn = trunc_level(len(data))
        g = np.clip(g, cn, 1 - cn)
    else:
        cn = None

    X_obs = data[COVARIATES + [TREATMENT_COL]].copy()
    QAW = clip_probs(q_model.predict_proba(X_obs)[:, 1])

    X1 = data[COVARIATES].copy()
    X1[TREATMENT_COL] = 1
    Q1W = clip_probs(q_model.predict_proba(X1)[:, 1])

    X0 = data[COVARIATES].copy()
    X0[TREATMENT_COL] = 0
    Q0W = clip_probs(q_model.predict_proba(X0)[:, 1])

    A = data[TREATMENT_COL].to_numpy()
    Y = data[OUTCOME_COL].to_numpy()

    HAW = A / g - (1 - A) / (1 - g)
    H1W = 1 / g
    H0W = -1 / (1 - g)

    model = sm.GLM(Y, HAW.reshape(-1, 1), family=sm.families.Binomial(), offset=logit(QAW))
    result = model.fit()
    eps = float(result.params[0])

    Q1W_star = clip_probs(expit(logit(Q1W) + eps * H1W))
    Q0W_star = clip_probs(expit(logit(Q0W) + eps * H0W))
    psi = float(np.mean(Q1W_star - Q0W_star))

    diagnostics = {
        "n": int(len(data)),
        "truncate": bool(truncate),
        "trunc_level": None if cn is None else float(cn),
        "epsilon": eps,
        "min_g": float(np.min(g)),
        "max_g": float(np.max(g)),
        "mean_g": float(np.mean(g)),
        "count_g_lt_cn": 0 if cn is None else int(np.sum(g <= cn + 1e-15)),
        "count_g_gt_1mcn": 0 if cn is None else int(np.sum(g >= 1 - cn - 1e-15)),
        "count_g_lt_0.001": int(np.sum(g < 0.001)),
        "count_g_gt_0.999": int(np.sum(g > 0.999)),
        "mean_abs_H": float(np.mean(np.abs(HAW))),
        "max_abs_H": float(np.max(np.abs(HAW))),
        "mean_QAW": float(np.mean(QAW)),
        "min_QAW": float(np.min(QAW)),
        "max_QAW": float(np.max(QAW)),
    }

    return psi, diagnostics


def evaluate_tmle_file(path, ate_true):
    df = pd.read_csv(path)

    psi_raw, diag_raw = tmle_estimate_df(df, truncate=False)
    psi_trunc, diag_trunc = tmle_estimate_df(df, truncate=True)

    return {
        "file": path,
        "tmle": {
            "estimate": psi_raw,
            "ate_true": ate_true,
            "bias": float(psi_raw - ate_true),
            "abs_bias": float(abs(psi_raw - ate_true)),
            "diagnostics": diag_raw,
        },
        "tmle_truncated": {
            "estimate": psi_trunc,
            "ate_true": ate_true,
            "bias": float(psi_trunc - ate_true),
            "abs_bias": float(abs(psi_trunc - ate_true)),
            "diagnostics": diag_trunc,
        },
    }


def paired_file_path(dataset_name, kind):
    if kind == "llm_self":
        return os.path.join(PAIR_SOURCES[kind], f"{dataset_name}_llm_self_supervised_pair.csv")
    if kind == "gan_self":
        return os.path.join(PAIR_SOURCES[kind], f"{dataset_name}_gan_self_supervised_pair.csv")
    if kind == "llm_qhyb":
        return os.path.join(PAIR_SOURCES[kind], f"{dataset_name}_llm_pair_qhyb.csv")
    if kind == "gan_qhyb":
        return os.path.join(PAIR_SOURCES[kind], f"{dataset_name}_gan_pair_qhyb.csv")
    raise ValueError(kind)


def summarize(results):
    buckets = {
        "orig_tmle": [],
        "orig_tmle_truncated": [],
        "llm_self_tmle": [],
        "llm_self_tmle_truncated": [],
        "gan_self_tmle": [],
        "gan_self_tmle_truncated": [],
        "llm_qhyb_tmle": [],
        "llm_qhyb_tmle_truncated": [],
        "gan_qhyb_tmle": [],
        "gan_qhyb_tmle_truncated": [],
    }

    for dataset_name, block in results["datasets"].items():
        buckets["orig_tmle"].append(block["orig"]["tmle"]["abs_bias"])
        buckets["orig_tmle_truncated"].append(block["orig"]["tmle_truncated"]["abs_bias"])
        buckets["llm_self_tmle"].append(block["llm_self"]["tmle"]["abs_bias"])
        buckets["llm_self_tmle_truncated"].append(block["llm_self"]["tmle_truncated"]["abs_bias"])
        buckets["gan_self_tmle"].append(block["gan_self"]["tmle"]["abs_bias"])
        buckets["gan_self_tmle_truncated"].append(block["gan_self"]["tmle_truncated"]["abs_bias"])
        buckets["llm_qhyb_tmle"].append(block["llm_qhyb"]["tmle"]["abs_bias"])
        buckets["llm_qhyb_tmle_truncated"].append(block["llm_qhyb"]["tmle_truncated"]["abs_bias"])
        buckets["gan_qhyb_tmle"].append(block["gan_qhyb"]["tmle"]["abs_bias"])
        buckets["gan_qhyb_tmle_truncated"].append(block["gan_qhyb"]["tmle_truncated"]["abs_bias"])

    return {k: float(np.mean(v)) for k, v in buckets.items()}


def main():
    ate_true, truth = load_truth()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {
        "truth": truth,
        "threshold": THRESHOLD,
        "datasets": {},
    }

    for dataset_name, orig_path in ORIG_DATASETS.items():
        results["datasets"][dataset_name] = {
            "orig": evaluate_tmle_file(orig_path, ate_true),
            "llm_self": evaluate_tmle_file(paired_file_path(dataset_name, "llm_self"), ate_true),
            "gan_self": evaluate_tmle_file(paired_file_path(dataset_name, "gan_self"), ate_true),
            "llm_qhyb": evaluate_tmle_file(paired_file_path(dataset_name, "llm_qhyb"), ate_true),
            "gan_qhyb": evaluate_tmle_file(paired_file_path(dataset_name, "gan_qhyb"), ate_true),
        }
        print(f"Finished {dataset_name}")

    results["avg_abs_bias_summary"] = summarize(results)

    out_path = os.path.join(RESULTS_DIR, "tmle_truncation_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()