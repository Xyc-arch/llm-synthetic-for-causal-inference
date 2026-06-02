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


def get_tstr_setting(tstr_all, setting_name):
    if "settings" in tstr_all:
        return tstr_all["settings"][setting_name]["datasets"]
    return tstr_all


def get_dcr_setting(dcr_all, setting_name):
    if "settings" in dcr_all:
        return dcr_all["settings"][setting_name]["datasets"]
    return dcr_all


def get_ate_setting(ate_all, setting_name):
    if "settings" in ate_all:
        return ate_all["settings"][setting_name]
    return ate_all


def get_available_settings(tstr_all, dcr_all, ate_all):
    setting_sets = []

    if "settings" in tstr_all:
        setting_sets.append(set(tstr_all["settings"].keys()))

    if "settings" in dcr_all:
        setting_sets.append(set(dcr_all["settings"].keys()))

    if "settings" in ate_all:
        setting_sets.append(set(ate_all["settings"].keys()))

    if not setting_sets:
        return ["single_setting"]

    common = set.intersection(*setting_sets)
    return sorted(common)


def get_metric_mean_se(record):
    if "mean" in record:
        return float(record["mean"]), float(record.get("se", 0.0))

    if "repeated_mean_summary" in record:
        s = record["repeated_mean_summary"]
        return float(s["mean"]), float(s.get("se", 0.0))

    if "summary" in record:
        s = record["summary"]
        return float(s["mean"]), 0.0

    raise KeyError(f"Cannot find mean/se in record keys: {list(record.keys())}")


def plot_tstr_dcr_for_setting(setting_name, tstr_ds, dcr_ds, out_dir):
    labels = [DISPLAY_LABELS[d] for d in DATASETS]

    tstr_vals = []
    tstr_errs = []
    dcr_vals = []
    dcr_errs = []

    for ds in DATASETS:
        if ds not in tstr_ds:
            raise KeyError(f"Missing {ds} in TSTR results for {setting_name}")
        if ds not in dcr_ds:
            raise KeyError(f"Missing {ds} in DCR results for {setting_name}")

        t_mean, t_se = get_metric_mean_se(tstr_ds[ds])
        d_mean, d_se = get_metric_mean_se(dcr_ds[ds])

        tstr_vals.append(t_mean)
        tstr_errs.append(t_se)
        dcr_vals.append(d_mean)
        dcr_errs.append(d_se)

    colors = [BAR_COLORS[d] for d in DATASETS]
    x = np.arange(len(DATASETS))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)

    bars1 = axes[0].bar(
        x,
        tstr_vals,
        yerr=tstr_errs,
        capsize=3,
        color=colors,
        width=0.68,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("TSTR AUC")
    axes[0].set_ylim(
        0.0,
        min(1.0, nice_upper_limit([v + e for v, e in zip(tstr_vals, tstr_errs)])),
    )
    style_axis(axes[0])
    style_y_ticks(axes[0])
    add_labels(axes[0], bars1, fmt="{:.3f}", fontsize=8)

    bars2 = axes[1].bar(
        x,
        dcr_vals,
        yerr=dcr_errs,
        capsize=3,
        color=colors,
        width=0.68,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("Mean DCR")
    axes[1].set_ylim(0.0, nice_upper_limit([v + e for v, e in zip(dcr_vals, dcr_errs)]))
    style_axis(axes[1])
    style_y_ticks(axes[1])
    add_labels(axes[1], bars2, fmt="{:.3f}", fontsize=8)

    out_path = out_dir / "tstr_dcr.png"
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_path}")


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
                raise KeyError(f"Missing estimator {est} for {full_gen_ds}, setting={setting_name}")
            if est not in ate_ds[hybrid_ds]:
                raise KeyError(f"Missing estimator {est} for {hybrid_ds}, setting={setting_name}")

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

    out_path = out_dir / "ate_mse_llm_gan_panels.png"
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tstr-file", type=str, default="vary_results/tstr.json")
    parser.add_argument("--dcr-file", type=str, default="vary_results_dcr/dcr_aggregate.json")
    parser.add_argument("--ate-file", type=str, default="vary_results/vary_estimators_compact.json")
    parser.add_argument("--output-dir", type=str, default="plot_vary")
    parser.add_argument("--setting", type=str, default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    tstr_file = script_dir / args.tstr_file
    dcr_file = script_dir / args.dcr_file
    ate_file = script_dir / args.ate_file
    output_dir = script_dir / args.output_dir

    tstr_all = load_json(tstr_file)
    dcr_all = load_json(dcr_file)
    ate_all = load_json(ate_file)

    settings = get_available_settings(tstr_all, dcr_all, ate_all)

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

        tstr_ds = get_tstr_setting(tstr_all, setting_name)
        dcr_ds = get_dcr_setting(dcr_all, setting_name)
        ate_setting = get_ate_setting(ate_all, setting_name)

        plot_tstr_dcr_for_setting(
            setting_name=setting_name,
            tstr_ds=tstr_ds,
            dcr_ds=dcr_ds,
            out_dir=setting_out_dir,
        )

        plot_ate_mse_for_setting(
            setting_name=setting_name,
            ate_setting=ate_setting,
            out_dir=setting_out_dir,
        )

    print(f"Done. Plots saved under: {output_dir}")


if __name__ == "__main__":
    main()