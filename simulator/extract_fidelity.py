#!/usr/bin/env python3
import os
import json
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "simulator", "results")

INPUT_CSV = os.path.join(
    RESULTS_DIR,
    "simulation_engine_global_hybrid_self_ref_summary.csv",
)

OUTPUT_JSON = os.path.join(
    RESULTS_DIR,
    "real_fidelity_table_compact.json",
)

SOURCE_ORDER = ["llm", "gan"]
ESTIMATOR_ORDER = ["ipw", "tmle", "aipw", "outcome_regression"]

SOURCE_LABELS = {
    "llm": "LLM",
    "gan": "GAN",
}

ESTIMATOR_LABELS = {
    "ipw": "IPW",
    "tmle": "TMLE",
    "aipw": "AIPW",
    "outcome_regression": "OR",
}


def f4(x):
    return round(float(x), 4)


def f6(x):
    return round(float(x), 6)


def pm(m, se):
    return f"{float(m):.6f} ± {float(se):.6f}"


def sign_correct(real_bias, syn_bias):
    real_bias = float(real_bias)
    syn_bias = float(syn_bias)

    if real_bias == 0 or syn_bias == 0:
        return real_bias == syn_bias

    return (real_bias > 0 and syn_bias > 0) or (real_bias < 0 and syn_bias < 0)


def get_one(df, source, estimator):
    sub = df[
        (df["source"].astype(str) == source)
        & (df["estimator"].astype(str) == estimator)
    ]

    if len(sub) != 1:
        raise ValueError(f"Expected one row for source={source}, estimator={estimator}; got {len(sub)}")

    return sub.iloc[0]


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    rows = []

    for source in SOURCE_ORDER:
        for estimator in ESTIMATOR_ORDER:
            r = get_one(df, source, estimator)

            real_bias = float(r["real_group_bias"])
            syn_bias = float(r["synthetic_vs_real_bias"])

            rows.append({
                "Source": SOURCE_LABELS[source],
                "Estimator": ESTIMATOR_LABELS[estimator],
                "Sign correct": "Yes" if sign_correct(real_bias, syn_bias) else "No",
                "Real bias": f4(real_bias),
                "Syn. bias": f4(syn_bias),
                "Real var": f6(r["real_group_var"]),
                "Syn. var": f6(r["synthetic_vs_real_var"]),
                "Real RMSE": f4(r["real_group_rmse"]),
                "Syn. RMSE": f4(r["synthetic_vs_real_rmse"]),
                "Real MSE": pm(r["real_group_mse"], r["real_group_mse_se"]),
                "Syn. MSE": pm(r["synthetic_vs_real_mse"], r["synthetic_vs_real_mse_se"]),
            })

    out = {
        "columns": [
            "Source",
            "Estimator",
            "Sign correct",
            "Real bias",
            "Syn. bias",
            "Real var",
            "Syn. var",
            "Real RMSE",
            "Syn. RMSE",
            "Real MSE",
            "Syn. MSE",
        ],
        "rows": rows,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved compact table JSON to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()