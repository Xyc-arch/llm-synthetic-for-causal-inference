#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
PLOT_DIR = SCRIPT_DIR / "plot"

TSTR_FILE = RESULTS_DIR / "tstr.json"
DCR_FILE = RESULTS_DIR / "dcr_aggregate.json"
ATE_FILE = RESULTS_DIR / "privacy_estimators_compact.json"

OUT_TSTR_DCR = PLOT_DIR / "privacy_tstr_dcr_compact.png"
OUT_ATE_MSE = PLOT_DIR / "privacy_ate_mse_llm_gan_panels.png"


DATASETS = [
    "llm_syn_clean",
    "llm_syn_hybrid",
    "gan_syn_clean",
    "gan_syn_hybrid",
]

DISPLAY_LABELS = {
    "llm_syn_clean": "LLM\nFull generative",
    "llm_syn_hybrid": "LLM\nHybrid",
    "gan_syn_clean": "GAN\nFull generative",
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

PAIR_LABELS = {
    "llm_syn_clean": "Full generative",
    "llm_syn_hybrid": "Hybrid",
    "gan_syn_clean": "Full generative",
    "gan_syn_hybrid": "Hybrid",
}


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r") as f:
        return json.load(f)


def get_dcr_mean_and_se(dcr, ds):
    """
    New run_dcr.py stores repeated synthetic-row subsampling uncertainty in:
        repeated_mean_summary: {mean, se, ...}

    Fallback supports the older dcr_aggregate.json format:
        summary: {mean, ...}
    """
    if "repeated_mean_summary" in dcr[ds]:
        return (
            float(dcr[ds]["repeated_mean_summary"]["mean"]),
            float(dcr[ds]["repeated_mean_summary"].get("se", 0.0)),
        )

    return float(dcr[ds]["summary"]["mean"]), 0.0


def build_metrics():
    tstr = load_json(TSTR_FILE)
    dcr = load_json(DCR_FILE)
    ate = load_json(ATE_FILE)

    metrics = {}

    for ds in DATASETS:
        dcr_mean, dcr_se = get_dcr_mean_and_se(dcr, ds)

        metrics[ds] = {
            "label": DISPLAY_LABELS[ds],

            # TSTR AUC:
            # New tstr.json stores mean/se; fallback to old "auc" format if needed.
            "tstr_auc": float(tstr[ds].get("mean", tstr[ds]["auc"])),
            "tstr_auc_se": float(tstr[ds].get("se", 0.0)),

            # DCR:
            # New dcr_aggregate.json stores mean/se over synthetic-row subsampling seeds.
            "dcr_mean": dcr_mean,
            "dcr_mean_se": dcr_se,

            # Causal fidelity:
            # ATE MSE and SE over evaluation/subsampling seeds.
            "ate_mse": {
                est: float(ate["datasets"][ds][est]["mse"])
                for est in ESTIMATORS
            },
            "ate_mse_se": {
                est: float(ate["datasets"][ds][est].get("mse_se", 0.0))
                for est in ESTIMATORS
            },
        }

    return metrics


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


def plot_tstr_and_dcr(metrics):
    labels = [metrics[d]["label"] for d in DATASETS]

    tstr_vals = [metrics[d]["tstr_auc"] for d in DATASETS]
    tstr_errs = [metrics[d]["tstr_auc_se"] for d in DATASETS]

    dcr_vals = [metrics[d]["dcr_mean"] for d in DATASETS]
    dcr_errs = [metrics[d]["dcr_mean_se"] for d in DATASETS]

    colors = [BAR_COLORS[d] for d in DATASETS]
    x = np.arange(len(DATASETS))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)

    bars1 = axes[0].bar(
        x,
        tstr_vals,
        yerr=tstr_errs,
        capsize=3,
        color=colors,
        width=0.68,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("TSTR AUC")
    axes[0].set_title("Predictive Utility")
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
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Mean DCR")
    axes[1].set_title("Privacy Distance")
    style_axis(axes[1])
    style_y_ticks(axes[1])
    add_labels(axes[1], bars2, fmt="{:.3f}", fontsize=8)

    fig.savefig(OUT_TSTR_DCR, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_TSTR_DCR}")


def nice_upper_limit(values):
    max_v = max(values) if values else 0.0

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

    return round(upper + 0.1, 1)


def plot_ate_mse_llm_gan_panels(metrics):
    """
    ATE MSE plot with separate panels for LLM and GAN.

    Each panel has:
        x-axis: estimator
        bars: Full generative vs Hybrid
        y-axis: ATE MSE
    """
    x = np.arange(len(ESTIMATORS))
    width = 0.36

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.2),
        sharey=False,
        constrained_layout=True,
    )

    for ax, (gen_name, ds_pair) in zip(axes, GENERATOR_PANELS.items()):
        full_ds, hybrid_ds = ds_pair

        full_vals = [metrics[full_ds]["ate_mse"][est] for est in ESTIMATORS]
        hybrid_vals = [metrics[hybrid_ds]["ate_mse"][est] for est in ESTIMATORS]

        full_errs = [metrics[full_ds]["ate_mse_se"][est] for est in ESTIMATORS]
        hybrid_errs = [metrics[hybrid_ds]["ate_mse_se"][est] for est in ESTIMATORS]

        bars_full = ax.bar(
            x - width / 2,
            full_vals,
            width,
            yerr=full_errs,
            capsize=3,
            label="Full generative",
            color=BAR_COLORS[full_ds],
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

        ax.set_title(f"{gen_name}: ATE MSE", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([ESTIMATOR_LABELS[e] for e in ESTIMATORS], fontsize=10)
        ax.set_ylabel("ATE MSE")
        ax.legend(frameon=False, fontsize=9)

        all_vals_with_errs = [
            v + e for v, e in zip(full_vals, full_errs)
        ] + [
            v + e for v, e in zip(hybrid_vals, hybrid_errs)
        ]

        ax.set_ylim(0, nice_upper_limit(all_vals_with_errs))

        style_axis(ax)
        style_y_ticks(ax, nbins=6)

        add_labels(ax, bars_full, fmt="{:.3f}", fontsize=7, y_offset=3)
        add_labels(ax, bars_hybrid, fmt="{:.3f}", fontsize=7, y_offset=3)

    fig.suptitle("Causal Fidelity: ATE MSE by Generator", fontsize=13)

    fig.savefig(OUT_ATE_MSE, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_ATE_MSE}")


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = build_metrics()

    plot_tstr_and_dcr(metrics)
    plot_ate_mse_llm_gan_panels(metrics)


if __name__ == "__main__":
    main()