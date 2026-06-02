#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


ESTIMATORS = ["ipw", "aipw", "outcome_regression", "tmle"]

EST_LABELS = {
    "ipw": "IPW",
    "aipw": "AIPW",
    "outcome_regression": "OR",
    "tmle": "TMLE",
}

GENERATORS = ["llm", "gan"]


def fmt_mean_se(mean, se, digits=3):
    return f"{mean:.{digits}f} ± {se:.{digits}f}"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_metric(datasets, gen, variant, est, metric):
    key = f"{gen}_syn_{variant}"
    if key not in datasets:
        return None
    if est not in datasets[key]:
        return None
    item = datasets[key][est]
    if "error" in item:
        return None
    return item.get(metric)


def print_setting_summary(setting_name, setting):
    print("\n" + "=" * 120)
    print(f"SETTING: {setting_name}")
    print(
        f"d={setting.get('d')} | "
        f"n_seed={setting.get('n_seed')} | "
        f"overlap={setting.get('overlap')} | "
        f"outcome={setting.get('outcome_mode')} | "
        f"ATE true={setting.get('ate_true'):.6f}"
    )
    print("=" * 120)

    datasets = setting["datasets"]

    for gen in GENERATORS:
        print(f"\n--- {gen.upper()} clean vs hybrid ---")
        header = (
            f"{'Estimator':<10}"
            f"{'Clean ATE ± SE':>20}"
            f"{'Hybrid ATE ± SE':>22}"
            f"{'Clean RMSE':>14}"
            f"{'Hybrid RMSE':>14}"
            f"{'RMSE Δ':>12}"
            f"{'RMSE %Δ':>12}"
        )
        print(header)
        print("-" * len(header))

        for est in ESTIMATORS:
            clean_mean = get_metric(datasets, gen, "clean", est, "mean")
            clean_se = get_metric(datasets, gen, "clean", est, "se")
            clean_rmse = get_metric(datasets, gen, "clean", est, "rmse")

            hyb_mean = get_metric(datasets, gen, "hybrid", est, "mean")
            hyb_se = get_metric(datasets, gen, "hybrid", est, "se")
            hyb_rmse = get_metric(datasets, gen, "hybrid", est, "rmse")

            if None in [clean_mean, clean_se, clean_rmse, hyb_mean, hyb_se, hyb_rmse]:
                continue

            rmse_delta = clean_rmse - hyb_rmse
            rmse_pct = 100.0 * rmse_delta / clean_rmse if clean_rmse > 0 else float("nan")

            print(
                f"{EST_LABELS[est]:<10}"
                f"{fmt_mean_se(clean_mean, clean_se):>20}"
                f"{fmt_mean_se(hyb_mean, hyb_se):>22}"
                f"{clean_rmse:>14.3f}"
                f"{hyb_rmse:>14.3f}"
                f"{rmse_delta:>12.3f}"
                f"{rmse_pct:>11.1f}%"
            )


def print_overall_summary(results):
    rows = []

    for setting_name, setting in results["settings"].items():
        datasets = setting["datasets"]

        for gen in GENERATORS:
            for est in ESTIMATORS:
                clean_rmse = get_metric(datasets, gen, "clean", est, "rmse")
                hyb_rmse = get_metric(datasets, gen, "hybrid", est, "rmse")

                if clean_rmse is None or hyb_rmse is None:
                    continue

                delta = clean_rmse - hyb_rmse
                pct = 100.0 * delta / clean_rmse if clean_rmse > 0 else float("nan")

                rows.append({
                    "setting": setting_name,
                    "generator": gen,
                    "estimator": est,
                    "clean_rmse": clean_rmse,
                    "hybrid_rmse": hyb_rmse,
                    "delta": delta,
                    "pct": pct,
                })

    print("\n" + "=" * 120)
    print("OVERALL RMSE IMPROVEMENT SUMMARY")
    print("=" * 120)

    for gen in GENERATORS:
        gen_rows = [r for r in rows if r["generator"] == gen]
        if not gen_rows:
            continue

        improved = [r for r in gen_rows if r["delta"] > 0]
        worsened = [r for r in gen_rows if r["delta"] < 0]

        avg_delta = sum(r["delta"] for r in gen_rows) / len(gen_rows)
        avg_pct = sum(r["pct"] for r in gen_rows) / len(gen_rows)

        print(f"\n{gen.upper()}:")
        print(f"  comparisons       : {len(gen_rows)}")
        print(f"  improved          : {len(improved)}")
        print(f"  worsened          : {len(worsened)}")
        print(f"  avg RMSE reduction: {avg_delta:.3f}")
        print(f"  avg % reduction   : {avg_pct:.1f}%")

    print("\nBy estimator:")
    header = (
        f"{'Generator':<10}"
        f"{'Estimator':<10}"
        f"{'N':>5}"
        f"{'Improved':>10}"
        f"{'Avg Δ RMSE':>14}"
        f"{'Avg %Δ':>12}"
    )
    print(header)
    print("-" * len(header))

    for gen in GENERATORS:
        for est in ESTIMATORS:
            sub = [r for r in rows if r["generator"] == gen and r["estimator"] == est]
            if not sub:
                continue

            improved = sum(1 for r in sub if r["delta"] > 0)
            avg_delta = sum(r["delta"] for r in sub) / len(sub)
            avg_pct = sum(r["pct"] for r in sub) / len(sub)

            print(
                f"{gen.upper():<10}"
                f"{EST_LABELS[est]:<10}"
                f"{len(sub):>5}"
                f"{improved:>10}"
                f"{avg_delta:>14.3f}"
                f"{avg_pct:>11.1f}%"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="vary_results/vary_estimators_compact.json",
    )
    parser.add_argument(
        "--only-setting",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    path = Path(args.input)
    results = load_json(path)

    if args.only_setting:
        setting = results["settings"][args.only_setting]
        print_setting_summary(args.only_setting, setting)
    else:
        for setting_name, setting in results["settings"].items():
            print_setting_summary(setting_name, setting)

        print_overall_summary(results)


if __name__ == "__main__":
    main()