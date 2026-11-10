"""Plot the 14-pair punctuation ablation summary used in Project Update 2."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = REPO_ROOT / "results" / "punctuation_ablation_summary.csv"
OUTPUT_PATH = REPO_ROOT / "docs" / "figures" / "punctuation_ablation.png"


def _plot_panel(ax, frame, title, ylim):
    x = range(len(frame))
    width = 0.24
    keep_bars = ax.bar(
        [position - width for position in x],
        frame["Keep"],
        width,
        label="A: Keep / inherit LID",
        color="#4C78A8",
    )
    drop_bars = ax.bar(
        list(x),
        frame["Drop"],
        width,
        label="B: Drop punctuation",
        color="#F28E2B",
    )
    neutral_bars = ax.bar(
        [position + width for position in x],
        frame["Neutral"],
        width,
        label="C: Keep / neutral LID",
        color="#59A14F",
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(list(x), frame["Metric"])
    ax.set_ylim(0, ylim)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    for bars in (keep_bars, drop_bars, neutral_bars):
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)

    for position, drop_delta, neutral_delta in zip(
        x,
        frame["Drop - Keep"],
        frame["Neutral - Keep"],
    ):
        ax.text(
            position,
            ylim * 0.04,
            f"B−A {drop_delta:+.3f} | C−A {neutral_delta:+.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#444444",
        )


def main():
    summary = pd.read_csv(SUMMARY_PATH)
    expected = {
        "Anticipatory F1",
        "Duration Accuracy",
        "Switch Rate",
        "Universality σ",
    }
    if set(summary["Metric"]) != expected:
        raise ValueError("punctuation ablation summary has unexpected metrics")

    performance = summary.iloc[:2]
    diagnostics = summary[summary["Metric"] == "Switch Rate"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 6),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    _plot_panel(axes[0], performance, "Prediction performance", 0.64)
    _plot_panel(axes[1], diagnostics, "Dataset diagnostic", 0.11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.88))
    fig.suptitle(
        "Punctuation Policies: Naive Baseline (14 Language Pairs)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "Same data caps and seed. C keeps punctuation as input but assigns a neutral LID.",
        ha="center",
        fontsize=9,
        color="#555555",
    )

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.78, wspace=0.20)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
