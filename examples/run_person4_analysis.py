"""Generate Person 4's tables and figures from the latest available runs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.person4_analysis import build_metadata, build_model_summary, build_pair_metrics, discover_runs


REPO_ROOT = Path(__file__).resolve().parents[1]


def _status_title(metadata):
    suffix = "" if metadata["result_status"] == "final" else " (provisional: 8-epoch files not all available)"
    return suffix


def _style(ax):
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def plot_model_comparison(summary, output, metadata):
    frame = summary.copy()
    labels = [f"{row.Backbone}\n{row.Approach}" for row in frame.itertuples()]
    x = range(len(frame))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=200)
    for ax, column, title in zip(
        axes,
        ["Anticipatory F1", "Duration Accuracy"],
        ["Anticipatory F1", "Duration accuracy"],
    ):
        bars = ax.bar([position - width / 2 for position in x], frame[column], width, color="#4C78A8")
        ax.set_xticks(list(x), labels, rotation=20, ha="right")
        ax.set_ylim(0, 0.72)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Score")
        _style(ax)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    fig.suptitle("Person 4: Model comparison" + _status_title(metadata), fontweight="bold", y=1.02)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_pair_f1(pairs, output, metadata):
    frame = pairs[pairs["Approach"].str.startswith("C:")].copy()
    if frame.empty:
        frame = pairs.copy()
    pivot = frame.pivot(index="Language pair", columns="Backbone", values="Anticipatory F1").sort_values(
        by="mBERT" if "mBERT" in frame["Backbone"].unique() else frame["Backbone"].iloc[0], ascending=False
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.3), dpi=200)
    colors = {"XLM-R": "#4C78A8", "mBERT": "#F28E2B"}
    for backbone in pivot.columns:
        ax.plot(pivot.index, pivot[backbone], marker="o", linewidth=2, label=backbone, color=colors.get(backbone))
    naive_path = REPO_ROOT / "results" / "naive_baseline_14pairs_neutral.json"
    if naive_path.exists():
        naive = json.loads(naive_path.read_text(encoding="utf-8"))["by_language_pair"]
        naive_values = [naive[pair]["anticipatory_f1"] for pair in pivot.index if pair in naive]
        if len(naive_values) == len(pivot.index):
            ax.plot(pivot.index, naive_values, marker="x", linestyle="--", color="#777777", label="Naive baseline")
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Anticipatory F1")
    ax.set_title("Person 4: Pair-level Anticipatory F1 (Approach C)" + _status_title(metadata), fontweight="bold")
    ax.tick_params(axis="x", rotation=55)
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_universality(summary, output, metadata):
    frame = summary.copy()
    frame["Label"] = frame["Backbone"] + "\n" + frame["Approach"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=200)
    bars = ax.bar(frame["Label"], frame["Universality σ"], color=["#4C78A8", "#F28E2B"] * 2)
    ax.set_ylabel("Pair-level F1 standard deviation (σ)")
    ax.set_title("Person 4: Universality across 14 language pairs" + _status_title(metadata), fontweight="bold")
    ax.set_ylim(0, max(0.12, frame["Universality σ"].max() * 1.35))
    ax.bar_label(bars, fmt="%.3f", padding=3)
    _style(ax)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "docs" / "figures"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    runs = discover_runs(results_dir)
    metadata = build_metadata(runs, results_dir)
    model_summary = build_model_summary(runs, results_dir)
    pair_metrics = build_pair_metrics(runs)

    model_summary.to_csv(results_dir / "person4_model_summary.csv", index=False)
    pair_metrics.to_csv(results_dir / "person4_pair_metrics.csv", index=False)
    (results_dir / "person4_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    plot_model_comparison(model_summary, output_dir / "person4_model_comparison.png", metadata)
    plot_pair_f1(pair_metrics, output_dir / "person4_pair_f1.png", metadata)
    plot_universality(model_summary, output_dir / "person4_universality.png", metadata)

    print(model_summary.to_string(index=False))
    print(f"saved {results_dir / 'person4_model_summary.csv'}")
    print(f"saved {results_dir / 'person4_pair_metrics.csv'}")
    print(f"status: {metadata['result_status']}")


if __name__ == "__main__":
    main()
