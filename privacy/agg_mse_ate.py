import os
import json
from collections import OrderedDict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(PROJECT_ROOT, "privacy", "results", "privacy_estimators.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "privacy", "results", "privacy_estimators_compact.json")


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def build_compact(results):
    compact = OrderedDict()
    compact["ate_true"] = results["truth"]["ate_true"]
    compact["subsample_n"] = results["subsample_n"]
    compact["seeds"] = results["seeds"]
    compact["datasets"] = OrderedDict()

    for dataset_name, dataset_results in results["datasets"].items():
        compact["datasets"][dataset_name] = OrderedDict()
        for est_name, metrics in dataset_results.items():
            compact["datasets"][dataset_name][est_name] = OrderedDict([
                ("mean", round(metrics["mean"], 6)),
                ("bias", round(metrics["bias"], 6)),
                ("abs_bias", round(metrics["abs_bias"], 6)),
                ("std", round(metrics["std"], 6)),
                ("mse", round(metrics["mse"], 6)),
                ("rmse", round(metrics["rmse"], 6)),
            ])

    return compact


def main():
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("INPUT_PATH   =", INPUT_PATH)
    print("OUTPUT_PATH  =", OUTPUT_PATH)

    results = load_results(INPUT_PATH)
    compact = build_compact(results)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(compact, f, indent=2)

    print(f"Saved compact aggregate JSON to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()