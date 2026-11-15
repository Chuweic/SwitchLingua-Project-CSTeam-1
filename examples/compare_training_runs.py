"""Summarize ``src.train`` runs: comparison table and convergence figure.

Reads every ``results/train_*.json`` file (or another ``--pattern``), writes
``results/training_comparison.csv``, and plots training loss, validation loss,
and validation Anticipatory F1 to ``docs/figures/training_convergence.png``.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
APPROACH = {"keep": "A", "drop": "B", "neutral": "C"}
BACKBONE_LABEL = {"xlm-roberta-base": "XLM-R", "bert-base-multilingual-cased": "mBERT"}
BACKBONE_COLOR = {"XLM-R": "#4C78A8", "mBERT": "#F28E2B"}
MODE_STYLE = {"keep": "-", "neutral": "--", "drop": ":"}


def _label(run):
    backbone = BACKBONE_LABEL.get(run["run"]["backbone"], run["run"]["backbone"])
    mode = run["run"]["punctuation_mode"]
    return backbone, mode, f"{backbone} · {APPROACH.get(mode, mode)} ({mode})"


def _naive_f1(results_dir, mode):
    path = results_dir / f"naive_baseline_14pairs_{mode}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["overall"]["anticipatory_f1"]


def build_table(runs, results_dir):
    rows = []
    for path, run in runs:
        backbone, mode, _ = _label(run)
        overall = run["overall"]
        naive = _naive_f1(results_dir, mode)
        cross = run.get("cross_eval")
        rows.append({
            "Backbone": backbone,
            "Approach": f"{APPROACH.get(mode, mode)}: {mode}",
            "Anticipatory F1": overall["anticipatory_f1"],
            "Precision": overall["precision"],
            "Recall": overall["recall"],
            "Duration Accuracy": overall["duration_accuracy"],
            "Universality σ": run["universality_std"],
            "Naive F1 (same policy)": naive,
            "F1 gain over naive": None if naive is None else overall["anticipatory_f1"] - naive,
            "Cross-eval labels": cross["label_punctuation_mode"] if cross else None,
            "Cross-eval F1": cross["overall"]["anticipatory_f1"] if cross else None,
            "Cross-eval Duration Accuracy": cross["overall"]["duration_accuracy"] if cross else None,
            "Epochs": len(run["history"]["epochs"]),
            "Best epoch": run["history"]["best_epoch"],
            "Threshold": run["threshold"],
            "Learning rate": f"{run['run']['hyperparameters']['learning_rate']:g}",
            "Train minutes": run["history"]["epochs"][-1]["seconds"] / 60,
            "File": path.name,
        })
    return pd.DataFrame(rows).sort_values(["Approach", "Backbone", "File"]).reset_index(drop=True)


def plot_convergence(runs, output):
    fig, (loss_ax, validation_ax, f1_ax) = plt.subplots(1, 3, figsize=(16, 4.6), dpi=200)
    for _, run in runs:
        backbone, mode, label = _label(run)
        style = dict(color=BACKBONE_COLOR.get(backbone, "#59A14F"), linestyle=MODE_STYLE.get(mode, "-"), linewidth=2)
        steps = run["history"]["steps"]
        epochs = run["history"]["epochs"]
        if steps:
            loss_ax.plot([entry["step"] for entry in steps], [entry["train_loss"] for entry in steps], label=label, **style)
        validation_ax.plot(
            [entry["epoch"] for entry in epochs],
            [entry["validation_loss"] for entry in epochs],
            marker="o",
            label=label,
            **style,
        )
        f1_ax.plot(
            [entry["epoch"] for entry in epochs],
            [entry["validation_anticipatory_f1"] for entry in epochs],
            marker="o",
            label=label,
            **style,
        )

    loss_ax.set_title("Training loss (switch + duration)", fontsize=12, fontweight="bold", pad=10)
    loss_ax.set_xlabel("Optimizer step")
    loss_ax.set_ylabel("Loss")
    validation_ax.set_title("Validation loss", fontsize=12, fontweight="bold", pad=10)
    validation_ax.set_xlabel("Epoch")
    validation_ax.set_ylabel("Loss")
    f1_ax.set_title("Validation Anticipatory F1", fontsize=12, fontweight="bold", pad=10)
    f1_ax.set_xlabel("Epoch")
    f1_ax.set_ylabel("F1 at validation-selected threshold")
    for ax in (validation_ax, f1_ax):
        ax.xaxis.get_major_locator().set_params(integer=True)
    for ax in (loss_ax, validation_ax, f1_ax):
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    f1_ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--pattern", default="train_*.json", help="Which result files to include.")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--figure", default=str(REPO_ROOT / "docs" / "figures" / "training_convergence.png"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    runs = [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(results_dir.glob(args.pattern))
    ]
    if not runs:
        raise SystemExit(f"no {args.pattern} files found in {results_dir}; run src.train first")

    table = build_table(runs, results_dir)
    output_csv = Path(args.output_csv) if args.output_csv else results_dir / "training_comparison.csv"
    table.to_csv(output_csv, index=False)
    plot_convergence(runs, Path(args.figure))

    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.4f}".format):
        print(table.drop(columns=["File"]).to_string(index=False))
    print("saved", output_csv)
    print("saved", args.figure)


if __name__ == "__main__":
    main()
