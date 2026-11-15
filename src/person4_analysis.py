"""Reproducible result selection and summaries for Person 4."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BACKBONES = {
    "XLM-R": ("xlmr", "xlm-roberta-base"),
    "mBERT": ("mbert", "bert-base-multilingual-cased"),
}
MODES = {"A": "keep", "C": "neutral"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_paths(results_dir: Path, short_name: str, mode: str) -> list[Path]:
    return [
        results_dir / f"converge_{short_name}_{mode}_8ep.json",
        results_dir / f"train_{short_name}_{mode}.json",
    ]


def _select_candidate(results_dir: Path, short_name: str, mode: str):
    for status, path in (("final", _candidate_paths(results_dir, short_name, mode)[0]),
                         ("provisional", _candidate_paths(results_dir, short_name, mode)[1])):
        if path.exists():
            return path, status
    return None, None


def discover_runs(results_dir: str | Path) -> dict[str, dict]:
    """Find the preferred run for every backbone/punctuation-policy pair.

    Eight-epoch neutral runs are preferred when present. Existing three-epoch
    runs remain usable but are explicitly marked provisional.
    """

    results_dir = Path(results_dir)
    selected = {}
    for backbone_label, (short_name, _) in BACKBONES.items():
        for approach, mode in MODES.items():
            path, status = _select_candidate(results_dir, short_name, mode)
            if path is None:
                continue
            selected[f"{backbone_label}:{approach}"] = {
                "path": path,
                "status": status,
                "data": _load(path),
                "backbone": backbone_label,
                "approach": approach,
                "punctuation_mode": mode,
            }
    if not selected:
        raise FileNotFoundError(f"no supported training result files found in {results_dir}")
    return selected


def _naive_f1(results_dir: Path, mode: str):
    path = results_dir / f"naive_baseline_14pairs_{mode}.json"
    if not path.exists():
        return None
    return _load(path)["overall"]["anticipatory_f1"]


def build_model_summary(runs: dict[str, dict], results_dir: str | Path) -> pd.DataFrame:
    results_dir = Path(results_dir)
    rows = []
    for item in runs.values():
        data = item["data"]
        overall = data["overall"]
        naive = _naive_f1(results_dir, item["punctuation_mode"])
        rows.append({
            "Backbone": item["backbone"],
            "Approach": f"{item['approach']}: {item['punctuation_mode']}",
            "Anticipatory F1": overall["anticipatory_f1"],
            "Precision": overall.get("precision"),
            "Recall": overall.get("recall"),
            "Duration Accuracy": overall.get("duration_accuracy"),
            "Universality σ": data.get("universality_std"),
            "Naive F1": naive,
            "F1 gain over naive": None if naive is None else overall["anticipatory_f1"] - naive,
            "Best epoch": data.get("history", {}).get("best_epoch"),
            "Result status": item["status"],
            "Source file": item["path"].name,
        })
    return pd.DataFrame(rows).sort_values(["Approach", "Backbone"]).reset_index(drop=True)


def build_pair_metrics(runs: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for item in runs.values():
        for pair, metrics in item["data"].get("by_language_pair", {}).items():
            rows.append({
                "Backbone": item["backbone"],
                "Approach": f"{item['approach']}: {item['punctuation_mode']}",
                "Language pair": pair,
                "Anticipatory F1": metrics.get("anticipatory_f1"),
                "Precision": metrics.get("precision"),
                "Recall": metrics.get("recall"),
                "Duration Accuracy": metrics.get("duration_accuracy"),
                "Result status": item["status"],
                "Source file": item["path"].name,
            })
    return pd.DataFrame(rows).sort_values(
        ["Approach", "Backbone", "Anticipatory F1"], ascending=[True, True, False]
    ).reset_index(drop=True)


def build_metadata(runs: dict[str, dict], results_dir: str | Path) -> dict:
    statuses = {item["status"] for item in runs.values()}
    return {
        "result_status": "final" if statuses == {"final"} else "provisional",
        "selected_files": {key: item["path"].name for key, item in runs.items()},
        "results_directory": str(Path(results_dir)),
        "language_pair_count": len(next(iter(runs.values()))["data"].get("by_language_pair", {})),
    }
