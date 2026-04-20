import os
import json
import math
import pandas as pd

PROJECT_ROOT = "/home/ubuntu/syn_causal"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "positivity", "results")

FILES = {
    "orig": os.path.join(RESULTS_DIR, "orig_estimators.json"),
    "pair_qhyb": os.path.join(RESULTS_DIR, "pair_qhyb_results.json"),
    "self_supervised": os.path.join(RESULTS_DIR, "self_supervised_pair_results.json"),
    "pair_qhyb_flip": os.path.join(RESULTS_DIR, "pair_qhyb_flip_results.json"),
}

ESTIMATOR_KEYS = ["aipw", "ipw", "outcome_regression", "tmle"]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def add_row(rows, experiment, dataset, estimator, bias, source=None, flip=None, file_name=None):
    rows.append(
        {
            "experiment": experiment,
            "dataset": dataset,
            "source": source,
            "flip": flip,
            "estimator": estimator,
            "bias": float(bias),
            "bias_sq": float(bias) ** 2,
            "file_name": file_name,
        }
    )


def parse_orig(payload, file_name):
    rows = []
    for dataset, dataset_obj in payload["datasets"].items():
        estimators = dataset_obj["estimators"]
        for est in ESTIMATOR_KEYS:
            add_row(
                rows=rows,
                experiment="orig",
                dataset=dataset,
                estimator=est,
                bias=estimators[est]["bias"],
                source=None,
                flip=None,
                file_name=file_name,
            )
    return rows


def parse_pair(payload, experiment_name, file_name):
    rows = []
    for dataset, dataset_obj in payload["datasets"].items():
        for source, source_obj in dataset_obj.items():
            estimators = source_obj["estimators"]
            for est in ESTIMATOR_KEYS:
                add_row(
                    rows=rows,
                    experiment=experiment_name,
                    dataset=dataset,
                    estimator=est,
                    bias=estimators[est]["bias"],
                    source=source,
                    flip=None,
                    file_name=file_name,
                )
    return rows


def parse_pair_flip(payload, file_name):
    rows = []
    for dataset, dataset_obj in payload["datasets"].items():
        for source, source_obj in dataset_obj.items():
            for flip_tag, flip_obj in source_obj.items():
                estimators = flip_obj["estimators"]
                for est in ESTIMATOR_KEYS:
                    add_row(
                        rows=rows,
                        experiment="pair_qhyb_flip",
                        dataset=dataset,
                        estimator=est,
                        bias=estimators[est]["bias"],
                        source=source,
                        flip=flip_tag,
                        file_name=file_name,
                    )
    return rows


def collect_all_rows():
    rows = []

    if os.path.exists(FILES["orig"]):
        payload = load_json(FILES["orig"])
        rows.extend(parse_orig(payload, os.path.basename(FILES["orig"])))

    if os.path.exists(FILES["pair_qhyb"]):
        payload = load_json(FILES["pair_qhyb"])
        rows.extend(parse_pair(payload, "pair_qhyb", os.path.basename(FILES["pair_qhyb"])))

    if os.path.exists(FILES["self_supervised"]):
        payload = load_json(FILES["self_supervised"])
        rows.extend(parse_pair(payload, "self_supervised", os.path.basename(FILES["self_supervised"])))

    if os.path.exists(FILES["pair_qhyb_flip"]):
        payload = load_json(FILES["pair_qhyb_flip"])
        rows.extend(parse_pair_flip(payload, os.path.basename(FILES["pair_qhyb_flip"])))

    return pd.DataFrame(rows)


def summarize_mse(df):
    summary = (
        df.groupby(["experiment", "source", "flip", "estimator"], dropna=False)
        .agg(
            n_seeds=("dataset", "nunique"),
            mse=("bias_sq", "mean"),
            rmse=("bias_sq", lambda x: math.sqrt(float(x.mean()))),
            mean_bias=("bias", "mean"),
            mean_abs_bias=("bias", lambda x: float(x.abs().mean())),
        )
        .reset_index()
        .sort_values(["experiment", "source", "flip", "estimator"], na_position="first")
        .reset_index(drop=True)
    )
    return summary


def main():
    df = collect_all_rows()

    if df.empty:
        raise ValueError("No rows collected. Check file paths and JSON structure.")

    # sanity check
    print("Collected rows:", len(df))
    print(df.head(20).to_string(index=False))

    summary = summarize_mse(df)

    long_csv = os.path.join(RESULTS_DIR, "all_results_long_bias_sq.csv")
    summary_csv = os.path.join(RESULTS_DIR, "all_results_mse_summary.csv")

    df.to_csv(long_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    print("\nSaved long table to:", long_csv)
    print("Saved summary to:", summary_csv)
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()