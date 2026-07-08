#!/usr/bin/env python3
import os
import json
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "positivity", "results")

INPUT_CSV = os.path.join(RESULTS_DIR, "all_results_by_n_mse_summary.csv")
OUTPUT_JSON = os.path.join(RESULTS_DIR, "positivity_mse_by_n_table.json")

ESTIMATORS = ["ipw", "aipw", "outcome_regression", "tmle"]

ESTIMATOR_LABELS = {
    "ipw": "IPW",
    "aipw": "AIPW",
    "outcome_regression": "OR",
    "tmle": "TMLE",
}

# Only include these scenarios in the JSON table.
SCENARIO_ORDER = [
    ("orig", None, None),

    ("pair_qhyb", "gan", None),
    ("pair_qhyb", "llm", None),

    ("self_supervised", "gan", None),
    ("self_supervised", "llm", None),

    ("pair_qhyb_flip", "gan", "flip_5"),
    ("pair_qhyb_flip", "llm", "flip_5"),

    ("pair_qhyb_flip", "gan", "flip_10"),
    ("pair_qhyb_flip", "llm", "flip_10"),
]


def clean_value(x):
    if pd.isna(x) or str(x).lower() == "nan":
        return None
    return str(x)


def format_mse_pm_se(mse, se):
    if pd.isna(mse):
        return "--"

    mse = float(mse)
    se = 0.0 if pd.isna(se) else float(se)

    return f"{mse:.4f} ± {se:.4f}"


def scenario_label(experiment, source=None, flip=None):
    source = clean_value(source)
    flip = clean_value(flip)

    source_label = {
        "gan": "GAN",
        "llm": "LLM",
    }.get(source, source.upper() if source else "")

    if experiment == "orig":
        return "Original"

    if experiment == "pair_qhyb":
        return f"Pair Hybrid {source_label}"

    if experiment == "self_supervised":
        return f"Pair Self-Supervised {source_label}"

    if experiment == "pair_qhyb_flip":
        flip_label = {
            "flip_5": "5%",
            "flip_10": "10%",
            "flip_20": "20%",
        }.get(flip, flip.replace("_", " ") if flip else "")

        return f"Pair Hybrid Flip {flip_label} {source_label}"

    return experiment


def scenario_json_key(experiment, source=None, flip=None):
    source = clean_value(source)
    flip = clean_value(flip)

    parts = [experiment]

    if source is not None:
        parts.append(source)

    if flip is not None:
        parts.append(flip)

    return "_".join(parts)


def match_rows(df, sample_size, experiment, source, flip, estimator):
    source = clean_value(source)
    flip = clean_value(flip)

    mask = (
        (df["sample_size"].astype(int) == int(sample_size))
        & (df["experiment"].astype(str) == str(experiment))
        & (df["estimator"].astype(str) == str(estimator))
    )

    if source is None:
        mask = mask & df["source"].isna()
    else:
        mask = mask & (df["source"].astype(str) == str(source))

    if flip is None:
        mask = mask & df["flip"].isna()
    else:
        mask = mask & (df["flip"].astype(str) == str(flip))

    return df[mask]


def build_json_for_n(df, sample_size):
    rows = []

    for experiment, source, flip in SCENARIO_ORDER:
        label = scenario_label(experiment, source, flip)
        key = scenario_json_key(experiment, source, flip)

        row = {
            "key": key,
            "label": label,
            "experiment": experiment,
            "source": clean_value(source),
            "flip": clean_value(flip),
            "estimators": {},
        }

        for est in ESTIMATORS:
            match = match_rows(
                df=df,
                sample_size=sample_size,
                experiment=experiment,
                source=source,
                flip=flip,
                estimator=est,
            )

            est_label = ESTIMATOR_LABELS[est]

            if len(match) == 0:
                row["estimators"][est] = {
                    "label": est_label,
                    "mse": None,
                    "mse_se": None,
                    "formatted": "--",
                }
                continue

            m = match.iloc[0]

            row["estimators"][est] = {
                "label": est_label,
                "mse": round(float(m["mse"]), 6),
                "mse_se": round(float(m["mse_se"]), 6),
                "formatted": format_mse_pm_se(m["mse"], m["mse_se"]),
            }

            optional_cols = [
                "mean_estimate",
                "se_estimate",
                "mean_bias",
                "se_bias",
                "mean_abs_bias",
                "se_abs_bias",
                "rmse",
                "n_reps",
                "ate_true",
            ]

            for col in optional_cols:
                if col in m and not pd.isna(m[col]):
                    if col == "n_reps":
                        row["estimators"][est][col] = int(m[col])
                    else:
                        row["estimators"][est][col] = round(float(m[col]), 6)

        rows.append(row)

    return {
        "sample_size": int(sample_size),
        "columns": [
            {"key": est, "label": ESTIMATOR_LABELS[est]}
            for est in ESTIMATORS
        ],
        "rows": rows,
    }


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Missing input CSV: {INPUT_CSV}\n"
            "Run collect.py first to create all_results_by_n_mse_summary.csv."
        )

    df = pd.read_csv(INPUT_CSV)

    required = {
        "sample_size",
        "experiment",
        "source",
        "flip",
        "estimator",
        "mse",
        "mse_se",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_CSV}: {sorted(missing)}")

    # Normalize missing values for source and flip.
    df["source"] = df["source"].where(~df["source"].isna(), None)
    df["flip"] = df["flip"].where(~df["flip"].isna(), None)

    sample_sizes = sorted(df["sample_size"].dropna().astype(int).unique().tolist())

    output = {
        "input_csv": INPUT_CSV,
        "sample_sizes": sample_sizes,
        "estimators": ESTIMATOR_LABELS,
        "scenario_order": [
            {
                "experiment": exp,
                "source": clean_value(src),
                "flip": clean_value(flip),
                "label": scenario_label(exp, src, flip),
            }
            for exp, src, flip in SCENARIO_ORDER
        ],
        "tables": {
            f"n{n}": build_json_for_n(df, n)
            for n in sample_sizes
        },
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved JSON table data to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()