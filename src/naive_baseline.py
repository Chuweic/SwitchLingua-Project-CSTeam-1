"""Current-token-only baseline and CLI for Project Update 1."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from src.switchlingua_labels import (
    DEFAULT_TOKENIZER,
    IGNORE_INDEX,
    SELECTED_LANGUAGE_PAIRS,
    load_fast_tokenizer,
    records_from_parquet,
)


def _divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _binary_metrics(targets: Sequence[int], predictions: Sequence[int]) -> Dict[str, float]:
    tp = sum(y == 1 and p == 1 for y, p in zip(targets, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(targets, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(targets, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(targets, predictions))
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    return {
        "anticipatory_f1": _divide(2 * precision * recall, precision + recall),
        "precision": precision,
        "recall": recall,
        "accuracy": _divide(tp + tn, tp + fp + fn + tn),
        "support": len(targets),
        "positive_support": sum(targets),
    }


class CurrentTokenBaseline:
    """Estimate P(next-token switch | current token ID) from training only."""

    def fit(self, records: Sequence[Mapping[str, Any]]) -> "CurrentTokenBaseline":
        self.token_counts: Dict[int, Counter] = defaultdict(Counter)
        self.duration_counts: Dict[int, Counter] = defaultdict(Counter)
        self.global_counts: Counter = Counter()
        self.global_durations: Counter = Counter()
        for record in records:
            for token, switch, duration in zip(
                record["input_ids"], record["switch_labels"], record["duration_labels"]
            ):
                if int(switch) == IGNORE_INDEX:
                    continue
                self.token_counts[int(token)][int(switch)] += 1
                self.global_counts[int(switch)] += 1
                if int(duration) >= 0:  # excludes both -1 and -100
                    self.duration_counts[int(token)][int(duration)] += 1
                    self.global_durations[int(duration)] += 1
        if not self.global_counts:
            raise ValueError("training records contain no valid labels")
        self.global_probability = _divide(self.global_counts[1], sum(self.global_counts.values()))
        self.global_duration = self.global_durations.most_common(1)[0][0] if self.global_durations else 0
        self.threshold = 0.5
        return self

    def probability(self, token_id: int) -> float:
        counts = self.token_counts.get(int(token_id))
        if not counts:
            return self.global_probability
        return (counts[1] + self.global_probability) / (sum(counts.values()) + 1)

    def duration(self, token_id: int) -> int:
        counts = self.duration_counts.get(int(token_id))
        return counts.most_common(1)[0][0] if counts else self.global_duration

    def tune(self, records: Sequence[Mapping[str, Any]]) -> float:
        probabilities, targets = _switch_examples(self, records)
        candidates = []
        for step in range(1, 100):
            threshold = step / 100
            predictions = [int(value >= threshold) for value in probabilities]
            f1 = _binary_metrics(targets, predictions)["anticipatory_f1"]
            candidates.append((f1, -abs(threshold - 0.5), threshold))
        self.threshold = max(candidates)[2]
        return self.threshold


def _switch_examples(
    model: CurrentTokenBaseline, records: Iterable[Mapping[str, Any]]
) -> Tuple[List[float], List[int]]:
    probabilities: List[float] = []
    targets: List[int] = []
    for record in records:
        for token, label in zip(record["input_ids"], record["switch_labels"]):
            if int(label) != IGNORE_INDEX:
                probabilities.append(model.probability(int(token)))
                targets.append(int(label))
    return probabilities, targets


def _evaluate(model: CurrentTokenBaseline, records: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    probabilities, targets = _switch_examples(model, records)
    metrics = _binary_metrics(targets, [int(value >= model.threshold) for value in probabilities])
    duration_targets: List[int] = []
    duration_predictions: List[int] = []
    for record in records:
        for token, duration in zip(record["input_ids"], record["duration_labels"]):
            if int(duration) >= 0:
                duration_targets.append(int(duration))
                duration_predictions.append(model.duration(int(token)))
    metrics["duration_accuracy"] = _divide(
        sum(y == p for y, p in zip(duration_targets, duration_predictions)), len(duration_targets)
    )
    metrics["duration_support"] = len(duration_targets)
    metrics["switch_rate"] = _divide(sum(targets), len(targets))
    return metrics


def evaluate_by_pair(model: CurrentTokenBaseline, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("language_pair", "unknown"))].append(record)
    by_pair = {pair: _evaluate(model, values) for pair, values in sorted(grouped.items())}
    return {
        "overall": _evaluate(model, records),
        "by_language_pair": by_pair,
        "universality_std": statistics.pstdev(
            metrics["anticipatory_f1"] for metrics in by_pair.values()
        ) if len(by_pair) > 1 else 0.0,
        "threshold": model.threshold,
        "definition": "P(switch at t+1 | current token_id at t), trained on train only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default="results/naive_baseline.json")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--pairs", nargs="+", default=list(SELECTED_LANGUAGE_PAIRS))
    parser.add_argument("--train-rows-per-pair", type=int, default=500)
    parser.add_argument("--validation-rows-per-pair", type=int, default=150)
    parser.add_argument("--test-rows-per-pair", type=int, default=150)
    parser.add_argument("--utterances-per-row", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = load_fast_tokenizer(args.tokenizer)
    limits = {
        "train": args.train_rows_per_pair,
        "validation": args.validation_rows_per_pair,
        "test": args.test_rows_per_pair,
    }
    records = {
        split: records_from_parquet(
            str(Path(args.data_dir) / f"{split}.parquet"),
            tokenizer,
            language_pairs=args.pairs,
            max_rows_per_pair=limit,
            max_utterances_per_row=args.utterances_per_row,
            seed=args.seed,
        )
        for split, limit in limits.items()
    }
    model = CurrentTokenBaseline().fit(records["train"])
    model.tune(records["validation"])
    result = evaluate_by_pair(model, records["test"])
    result["run"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": args.tokenizer,
        "language_pairs": args.pairs,
        "weak_lid_labels": True,
        "seed": args.seed,
        "rows_per_pair": limits,
        "utterances_per_row": args.utterances_per_row,
        "record_counts": {split: len(values) for split, values in records.items()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
