#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
PLOT_DIR = SCRIPT_DIR / "plot"

TSTR_FILE = RESULTS_DIR / "tstr.json"
DCR_FILE = RESULTS_DIR / "dcr_aggregate.json"
ATE_FILE = RESULTS_DIR / "privacy_estimators_compact.json"

OUT_TSTR_DCR = PLOT_DIR / "privacy_tstr_dcr_compact.png"
OUT_ATE_MSE = PLOT_DIR / "privacy_ate_mse_faceted_zoom.png"

DATASETS = [
    "llm_syn_clean",
    "llm_syn_hybrid",
    "gan_syn_clean",
    "gan_syn_hybrid",
]

DISPLAY_LABELS = {
    "llm_syn_clean": "LLM",
    "llm_syn_hybrid": "LLM+Hybrid",
    "gan_syn_clean": "GAN",
    "gan_syn_hybrid": "GAN+Hybrid",
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


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r") as f:
        return json.load(f)


def build_metrics():
    tstr = load_json(TSTR_FILE)
    dcr = load_json(DCR_FILE)
    ate = load_json(ATE_FILE)

    metrics = {}
    for ds in DATASETS:
        metrics[ds] = {
            "label": DISPLAY_LABELS[ds],
            "tstr_auc": float(tstr[ds]["auc"]),
            "dcr_mean": float(dcr[ds]["summary"]["mean"]),
            "ate_mse": {est: float(ate["datasets"][ds][est]["mse"]) for est in ESTIMATORS},
        }
    return metrics


def style_axis(ax):
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_labels(ax, bars, fmt="{:.3f}", fontsize=8):
    for b in bars:
        h = b.get_height()
        ax.annotate(
            fmt.format(h),
            (b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def plot_tstr_and_dcr(metrics):
    labels = [metrics[d]["label"] for d in DATASETS]
    tstr_vals = [metrics[d]["tstr_auc"] for d in DATASETS]
    dcr_vals = [metrics[d]["dcr_mean"] for d in DATASETS]
    colors = [BAR_COLORS[d] for d in DATASETS]
    x = np.arange(len(DATASETS))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)

    bars1 = axes[0].bar(x, tstr_vals, color=colors, width=0.68)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("TSTR AUC")
    axes[0].set_title("Predictive Utility")
    style_axis(axes[0])
    add_labels(axes[0], bars1, fmt="{:.3f}")

    bars2 = axes[1].bar(x, dcr_vals, color=colors, width=0.68)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Mean DCR")
    axes[1].set_title("Privacy Distance")
    style_axis(axes[1])
    add_labels(axes[1], bars2, fmt="{:.3f}")

    fig.savefig(OUT_TSTR_DCR, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_TSTR_DCR}")


def plot_ate_mse_faceted(metrics):
    labels = [metrics[d]["label"] for d in DATASETS]
    colors = [BAR_COLORS[d] for d in DATASETS]
    x = np.arange(len(DATASETS))

    y_cap = 0.02

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True, constrained_layout=True)

    for ax, est in zip(axes, ESTIMATORS):
        raw_vals = [metrics[d]["ate_mse"][est] for d in DATASETS]
        plot_vals = [min(v, y_cap) for v in raw_vals]

        bars = ax.bar(x, plot_vals, color=colors, width=0.68)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(ESTIMATOR_LABELS[est], fontsize=11)
        ax.set_ylim(0, y_cap)
        style_axis(ax)

        for bar, raw_v, plot_v in zip(bars, raw_vals, plot_vals):
            x_center = bar.get_x() + bar.get_width() / 2

            if raw_v > y_cap:
                ax.annotate(
                    f">{y_cap:.2f}",
                    (x_center, y_cap),
                    xytext=(0, -16),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=8,
                    fontweight="bold",
                )
            else:
                ax.annotate(
                    f"{raw_v:.3f}",
                    (x_center, plot_v),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    axes[0].set_ylabel("ATE MSE")
    # fig.suptitle("Causal Fidelity: ATE MSE", fontsize=13, y=1.03)

    fig.savefig(OUT_ATE_MSE, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_ATE_MSE}")


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = build_metrics()
    plot_tstr_and_dcr(metrics)
    plot_ate_mse_faceted(metrics)


if __name__ == "__main__":
    main()