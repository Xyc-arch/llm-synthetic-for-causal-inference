#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"

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


def get_one(df, setting, source, estimator):
    sub = df[
        (df["setting"].astype(str) == setting)
        & (df["source"].astype(str) == source)
        & (df["estimator"].astype(str) == estimator)
    ]

    if len(sub) != 1:
        raise ValueError(
            f"Expected one row for setting={setting}, source={source}, "
            f"estimator={estimator}; got {len(sub)}"
        )

    return sub.iloc[0]


def build_table_for_setting(df, setting):
    rows = []

    for source in SOURCE_ORDER:
        for estimator in ESTIMATOR_ORDER:
            r = get_one(df, setting, source, estimator)

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

    return {
        "setting": setting,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--reference-mode",
        type=str,
        choices=["self_ref", "estimator_specific"],
        default="self_ref",
    )
    parser.add_argument("--setting", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    suffix = "self_ref" if args.reference_mode == "self_ref" else "estimator_specific"

    input_csv = results_dir / f"all_settings_simulation_engine_{suffix}_summary.csv"
    output_json = results_dir / f"real_fidelity_tables_{suffix}_compact.json"

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    df = pd.read_csv(input_csv)

    required_cols = {
        "setting",
        "source",
        "estimator",
        "real_group_bias",
        "synthetic_vs_real_bias",
        "real_group_var",
        "synthetic_vs_real_var",
        "real_group_rmse",
        "synthetic_vs_real_rmse",
        "real_group_mse",
        "real_group_mse_se",
        "synthetic_vs_real_mse",
        "synthetic_vs_real_mse_se",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {input_csv}: {sorted(missing)}")

    if args.setting is not None:
        settings = [args.setting]
    else:
        settings = sorted(df["setting"].astype(str).unique().tolist())

    out = {
        "input_csv": str(input_csv),
        "reference_mode": args.reference_mode,
        "tables": {
            setting: build_table_for_setting(df, setting)
            for setting in settings
        },
    }

    with open(output_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved compact real-fidelity tables to: {output_json}")


if __name__ == "__main__":
    main()