#!/usr/bin/env python3
import argparse
import json
import re
from itertools import combinations
from pathlib import Path


DEFAULT_JSON = Path("results/real_fidelity_tables_self_ref_compact.json")

ESTIMATOR_ORDER = ["IPW", "TMLE", "AIPW", "OR"]
SOURCE_ORDER = ["LLM", "GAN"]


def parse_mse(x):
    """
    Parse strings like '0.004820 ± 0.001744' or numeric values.
    """
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not m:
        raise ValueError(f"Cannot parse MSE from: {x}")
    return float(m.group(0))


def latex_escape(s):
    return str(s).replace("_", r"\_")


def setting_label(setting):
    labels = {
        "d6_n1000_overlap_moderate_complex": r"$d=6$, moderate overlap, complex outcome",
        "d6_n1000_overlap_poor_complex": r"$d=6$, poor overlap, complex outcome",
        "d20_n1000_overlap_poor_complex": r"$d=20$, poor overlap, complex outcome",
        "d20_n500_overlap_poor_complex": r"$d=20$, poor overlap, complex outcome, $n_{\mathrm{seed}}=500$",
        "d20_n1000_overlap_poor_simple": r"$d=20$, poor overlap, simple outcome",
    }
    return labels.get(setting, latex_escape(setting))


def get_rows_by_source(table):
    by_source = {src: {} for src in SOURCE_ORDER}
    for row in table["rows"]:
        src = row["Source"]
        est = row["Estimator"]
        if src in by_source:
            by_source[src][est] = row
    return by_source


def rank_estimators(mse_by_est):
    """
    Return rank string from lowest MSE to highest MSE.
    """
    ordered = sorted(
        ESTIMATOR_ORDER,
        key=lambda est: (mse_by_est[est], ESTIMATOR_ORDER.index(est)),
    )
    return r" $<$ ".join(ordered)


def pairwise_accuracy(real_mse, syn_mse):
    """
    Compare pairwise ordering among estimators.
    Correct if real and synthetic agree on which estimator has lower MSE.
    Ties are treated as agreement only if both sides tie exactly.
    """
    correct = 0
    total = 0

    for a, b in combinations(ESTIMATOR_ORDER, 2):
        real_cmp = (real_mse[a] > real_mse[b]) - (real_mse[a] < real_mse[b])
        syn_cmp = (syn_mse[a] > syn_mse[b]) - (syn_mse[a] < syn_mse[b])

        if real_cmp == syn_cmp:
            correct += 1
        total += 1

    return correct, total, 100.0 * correct / total


def build_rank_rows(data):
    out_rows = []

    for setting, table in data["tables"].items():
        by_source = get_rows_by_source(table)

        for src in SOURCE_ORDER:
            if src not in by_source:
                continue

            rows = by_source[src]

            missing = [est for est in ESTIMATOR_ORDER if est not in rows]
            if missing:
                raise ValueError(f"Missing estimators for setting={setting}, source={src}: {missing}")

            real_mse = {
                est: parse_mse(rows[est]["Real MSE"])
                for est in ESTIMATOR_ORDER
            }
            syn_mse = {
                est: parse_mse(rows[est]["Syn. MSE"])
                for est in ESTIMATOR_ORDER
            }

            correct, total, acc = pairwise_accuracy(real_mse, syn_mse)

            out_rows.append({
                "setting": setting,
                "source": src,
                "correct": correct,
                "total": total,
                "accuracy": acc,
                "real_rank": rank_estimators(real_mse),
                "syn_rank": rank_estimators(syn_mse),
            })

    return out_rows


def print_latex_table(rank_rows):
    print(r"\begin{table*}[t!]")
    print(r"\centering")
    print(r"\caption{Estimator MSE ranking agreement between real and synthetic finite-sample behavior. For each setting and source, the table reports the real-data MSE rank, synthetic-data MSE rank, and pairwise rank agreement across the four estimators. Lower MSE is better. Pairwise accuracy is computed over the six estimator pairs per setting.}")
    print(r"\label{tab:sim_engine_rank_agreement}")
    print(r"\scriptsize")
    print(r"\setlength{\tabcolsep}{4pt}")
    print(r"\renewcommand{\arraystretch}{1.15}")
    print(r"\resizebox{\textwidth}{!}{%")
    print(r"\begin{tabular}{llcccll}")
    print(r"\hline")
    print(r"Setting & Source & Correct pairs & Accuracy & Real MSE rank & Synthetic MSE rank \\")
    print(r"\hline")

    overall = {src: {"correct": 0, "total": 0} for src in SOURCE_ORDER}

    current_setting = None
    for row in rank_rows:
        setting = row["setting"]
        src = row["source"]

        if current_setting is not None and setting != current_setting:
            print(r"\hline")
            print()

        setting_text = setting_label(setting) if setting != current_setting else ""
        current_setting = setting

        correct = row["correct"]
        total = row["total"]
        acc = row["accuracy"]

        overall[src]["correct"] += correct
        overall[src]["total"] += total

        print(
            f"{setting_text} & {src} & {correct}/{total} & {acc:.1f}\\% "
            f"& {row['real_rank']} & {row['syn_rank']} \\\\"
        )

    print(r"\hline")
    print()

    for src in SOURCE_ORDER:
        correct = overall[src]["correct"]
        total = overall[src]["total"]
        if total == 0:
            continue
        acc = 100.0 * correct / total
        print(
            rf"\textbf{{Overall}} & \textbf{{{src}}} & "
            rf"\textbf{{{correct}/{total}}} & \textbf{{{acc:.1f}\%}} & -- & -- \\"
        )

    print(r"\hline")
    print(r"\end{tabular}%")
    print(r"}")
    print(r"\end{table*}")


def print_debug(rank_rows):
    for row in rank_rows:
        print(
            f"{row['setting']} | {row['source']} | "
            f"{row['correct']}/{row['total']} | {row['accuracy']:.1f}% | "
            f"real: {row['real_rank']} | synthetic: {row['syn_rank']}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_JSON,
        help="Path to real_fidelity_tables_*_compact.json",
    )
    parser.add_argument(
        "--format",
        choices=["latex", "debug"],
        default="latex",
    )
    args = parser.parse_args()

    if not args.json.exists():
        raise FileNotFoundError(f"Missing JSON file: {args.json}")

    with open(args.json, "r") as f:
        data = json.load(f)

    rank_rows = build_rank_rows(data)

    if args.format == "latex":
        print_latex_table(rank_rows)
    else:
        print_debug(rank_rows)


if __name__ == "__main__":
    main()