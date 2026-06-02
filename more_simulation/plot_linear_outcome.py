#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter


DATASETS = [
    "llm_syn_clean",
    "llm_syn_hybrid",
    "gan_syn_clean",
    "gan_syn_hybrid",
]

DISPLAY_LABELS = {
    "llm_syn_clean": "LLM\nFully generative",
    "llm_syn_hybrid": "LLM\nHybrid",
    "gan_syn_clean": "GAN\nFully generative",
    "gan_syn_hybrid": "GAN\nHybrid",
}

# Linear outcome runner:
#   IPW: RF g, continuous Y
#   AIPW: RF g + linear Q, continuous Y
#   OR: linear Q, continuous Y
#   TMLE: logistic g + logistic Q, binary Y
ESTIMATORS = ["ipw", "aipw", "outcome_regression", "tmle"]

ESTIMATOR_LABELS = {
    "ipw": "IPW",
    "aipw": "AIPW",
    "outcome_regression": "OR",
    "tmle": "TMLE",
}

BAR_COLORS = {
    "llm_syn_clean": "#9ecae1",
    "llm_syn_hybrid": "#3182bd",
    "gan_syn_clean": "#fdd0a2",
    "gan_syn_hybrid": "#e6550d",
}

GENERATOR_PANELS = {
    "LLM": ["llm_syn_clean", "llm_syn_hybrid"],
    "GAN": ["gan_syn_clean", "gan_syn_hybrid"],
}


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r") as f:
        return json.load(f)


def style_axis(ax):
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def style_y_ticks(ax, nbins=6):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, min_n_ticks=4))

    formatter = ScalarFormatter(useMathText=False)
    formatter.set_scientific(False)
    formatter.set_useOffset(False)
    ax.yaxis.set_major_formatter(formatter)

    ax.tick_params(axis="y", labelsize=9)


def add_labels(ax, bars, fmt="{:.3f}", fontsize=8, y_offset=3):
    for b in bars:
        h = b.get_height()
        ax.annotate(
            fmt.format(h),
            (b.get_x() + b.get_width() / 2, h),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def nice_upper_limit(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]

    if not values:
        return 1.0

    max_v = max(values)

    if max_v <= 0:
        return 1.0

    upper = max_v * 1.18

    if upper < 0.01:
        return round(upper + 0.002, 3)
    if upper < 0.05:
        return round(upper + 0.005, 3)
    if upper < 0.1:
        return round(upper + 0.01, 2)
    if upper < 0.5:
        return round(upper + 0.05, 2)
    if upper < 1.0:
        return round(upper + 0.1, 1)

    return round(upper + 0.2, 1)


def get_ate_setting(ate_all, setting_name):
    if "settings" in ate_all:
        return ate_all["settings"][setting_name]
    return ate_all


def get_available_settings(ate_all):
    if "settings" in ate_all:
        return sorted(ate_all["settings"].keys())
    return ["single_setting"]


def print_mse_results(setting_name, ate_setting):
    if "datasets" not in ate_setting:
        raise KeyError(f"Missing datasets in ATE results for {setting_name}")

    ate_ds = ate_setting["datasets"]

    print()
    print("ATE MSE results")
    print(f"Setting: {setting_name}")
    print("-" * 86)
    print(
        f"{'Generator':<8} "
        f"{'Dataset':<18} "
        f"{'Estimator':<20} "
        f"{'MSE':>12} "
        f"{'MSE SE':>12}"
    )
    print("-" * 86)

    for gen_name, ds_pair in GENERATOR_PANELS.items():
        for ds in ds_pair:
            if ds not in ate_ds:
                raise KeyError(f"Missing {ds} in ATE results for {setting_name}")

            dataset_label = "Fully generative" if ds.endswith("clean") else "Hybrid"

            for est in ESTIMATORS:
                if est not in ate_ds[ds]:
                    raise KeyError(
                        f"Missing estimator {est} for {ds}, setting={setting_name}"
                    )

                mse = float(ate_ds[ds][est]["mse"])
                mse_se = float(ate_ds[ds][est].get("mse_se", 0.0))

                print(
                    f"{gen_name:<8} "
                    f"{dataset_label:<18} "
                    f"{ESTIMATOR_LABELS[est]:<20} "
                    f"{mse:>12.6f} "
                    f"{mse_se:>12.6f}"
                )

    print("-" * 86)


def plot_ate_mse_for_setting(setting_name, ate_setting, out_dir):
    if "datasets" not in ate_setting:
        raise KeyError(f"Missing datasets in ATE results for {setting_name}")

    ate_ds = ate_setting["datasets"]

    x = np.arange(len(ESTIMATORS))
    width = 0.36

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.0),
        sharey=False,
        constrained_layout=True,
    )

    for ax, (gen_name, ds_pair) in zip(axes, GENERATOR_PANELS.items()):
        full_gen_ds, hybrid_ds = ds_pair

        for ds in ds_pair:
            if ds not in ate_ds:
                raise KeyError(f"Missing {ds} in ATE results for {setting_name}")

        full_gen_vals = []
        hybrid_vals = []
        full_gen_errs = []
        hybrid_errs = []

        for est in ESTIMATORS:
            if est not in ate_ds[full_gen_ds]:
                raise KeyError(
                    f"Missing estimator {est} for {full_gen_ds}, setting={setting_name}"
                )
            if est not in ate_ds[hybrid_ds]:
                raise KeyError(
                    f"Missing estimator {est} for {hybrid_ds}, setting={setting_name}"
                )

            full_gen_vals.append(float(ate_ds[full_gen_ds][est]["mse"]))
            hybrid_vals.append(float(ate_ds[hybrid_ds][est]["mse"]))

            full_gen_errs.append(float(ate_ds[full_gen_ds][est].get("mse_se", 0.0)))
            hybrid_errs.append(float(ate_ds[hybrid_ds][est].get("mse_se", 0.0)))

        bars_full_gen = ax.bar(
            x - width / 2,
            full_gen_vals,
            width,
            yerr=full_gen_errs,
            capsize=3,
            label="Fully generative",
            color=BAR_COLORS[full_gen_ds],
        )

        bars_hybrid = ax.bar(
            x + width / 2,
            hybrid_vals,
            width,
            yerr=hybrid_errs,
            capsize=3,
            label="Hybrid",
            color=BAR_COLORS[hybrid_ds],
        )

        ax.set_xlabel(gen_name)
        ax.set_xticks(x)
        ax.set_xticklabels([ESTIMATOR_LABELS[e] for e in ESTIMATORS], fontsize=10)
        ax.set_ylabel("ATE MSE")
        ax.legend(frameon=False, fontsize=9)

        all_vals_with_errs = [
            v + e for v, e in zip(full_gen_vals, full_gen_errs)
        ] + [
            v + e for v, e in zip(hybrid_vals, hybrid_errs)
        ]

        ax.set_ylim(0, nice_upper_limit(all_vals_with_errs))

        style_axis(ax)
        style_y_ticks(ax, nbins=6)

        add_labels(ax, bars_full_gen, fmt="{:.3f}", fontsize=7, y_offset=3)
        add_labels(ax, bars_hybrid, fmt="{:.3f}", fontsize=7, y_offset=3)

    out_path = out_dir / "ate_mse_llm_gan_panels_linear.png"
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ate-file",
        type=str,
        default="vary_results_linear/vary_estimators_linear_compact.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plot_linear_outcome",
    )
    parser.add_argument("--setting", type=str, default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    ate_file = script_dir / args.ate_file
    output_dir = script_dir / args.output_dir

    ate_all = load_json(ate_file)

    settings = get_available_settings(ate_all)

    if args.setting is not None:
        if args.setting not in settings:
            raise ValueError(
                f"Requested setting {args.setting} not found. "
                f"Available settings: {settings}"
            )
        settings = [args.setting]

    output_dir.mkdir(parents=True, exist_ok=True)

    for setting_name in settings:
        print("=" * 100)
        print(f"Plotting setting: {setting_name}")
        print("=" * 100)

        setting_out_dir = output_dir / setting_name
        setting_out_dir.mkdir(parents=True, exist_ok=True)

        ate_setting = get_ate_setting(ate_all, setting_name)

        print_mse_results(
            setting_name=setting_name,
            ate_setting=ate_setting,
        )

        plot_ate_mse_for_setting(
            setting_name=setting_name,
            ate_setting=ate_setting,
            out_dir=setting_out_dir,
        )

    print(f"Done. Plots saved under: {output_dir}")


if __name__ == "__main__":
    main()