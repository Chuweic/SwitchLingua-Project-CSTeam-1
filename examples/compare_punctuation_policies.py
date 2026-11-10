"""Compare inherited-LID and neutral-LID punctuation on the manual audit."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from src.switchlingua_labels import (
    IGNORE_INDEX,
    NEUTRAL_LID,
    build_predictive_labels,
    is_punctuation_or_symbol_token,
    tokenize_mixed_text,
)


GROUP_FIELDS = ["language_pair", "row_id", "utterance_index"]


def _divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _with_punctuation_lids(
    words: Sequence[str],
    lexical_lids: Sequence[str],
    *,
    neutral: bool,
) -> List[str]:
    """Insert punctuation LIDs into an audited lexical-token LID sequence."""

    labels: List[Optional[str]] = []
    lexical_index = 0
    for word in words:
        if is_punctuation_or_symbol_token(word):
            labels.append(NEUTRAL_LID if neutral else None)
            continue
        if lexical_index >= len(lexical_lids):
            raise ValueError("audit has fewer lexical LIDs than reconstructed words")
        labels.append(str(lexical_lids[lexical_index]))
        lexical_index += 1
    if lexical_index != len(lexical_lids):
        raise ValueError("audit has more lexical LIDs than reconstructed words")

    if neutral:
        return [str(label) for label in labels]

    # Reproduce Approach A: punctuation inherits the nearest previous lexical
    # LID, falling back to the next lexical LID for leading punctuation.
    for index, label in enumerate(labels):
        if label is not None:
            continue
        previous = next(
            (labels[position] for position in range(index - 1, -1, -1) if labels[position]),
            None,
        )
        following = next(
            (
                labels[position]
                for position in range(index + 1, len(labels))
                if labels[position]
            ),
            None,
        )
        labels[index] = previous or following
    if any(label is None for label in labels):
        raise ValueError("utterance contains no lexical token to label punctuation")
    return [str(label) for label in labels]


def _binary_metrics(
    targets: Sequence[int], predictions: Sequence[int]
) -> Dict[str, Union[float, int]]:
    tp = sum(target == 1 and prediction == 1 for target, prediction in zip(targets, predictions))
    fp = sum(target == 0 and prediction == 1 for target, prediction in zip(targets, predictions))
    fn = sum(target == 1 and prediction == 0 for target, prediction in zip(targets, predictions))
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    return {
        "support": len(targets),
        "positive_support": sum(targets),
        "predicted_positive_support": sum(predictions),
        "precision": precision,
        "recall": recall,
        "f1": _divide(2 * precision * recall, precision + recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _score_positions(
    automatic_switch: Sequence[int],
    gold_switch: Sequence[int],
    positions: Sequence[int],
) -> Dict[str, Union[float, int]]:
    valid = [
        position
        for position in positions
        if automatic_switch[position] != IGNORE_INDEX
        and gold_switch[position] != IGNORE_INDEX
    ]
    return _binary_metrics(
        [gold_switch[position] for position in valid],
        [automatic_switch[position] for position in valid],
    )


def compare_audit(frame: pd.DataFrame) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Score Approach A and C against completed manual lexical LIDs."""

    required = set(GROUP_FIELDS + ["text", "token_index", "token", "automatic_lid", "gold_lid"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"audit file is missing columns: {missing}")
    if (frame["gold_lid"].astype(str).str.strip() == "").any():
        raise ValueError("audit file contains blank gold_lid values")

    approach_data = {
        "keep": {
            "automatic_switch": [],
            "gold_switch": [],
            "automatic_duration": [],
            "gold_duration": [],
            "near_punctuation": [],
        },
        "neutral": {
            "automatic_switch": [],
            "gold_switch": [],
            "automatic_duration": [],
            "gold_duration": [],
            "near_punctuation": [],
        },
    }
    comparison = {
        "shared_valid_switch_targets": 0,
        "switch_value_changes_on_shared_targets": 0,
        "keep_only_switch_targets": 0,
        "neutral_only_switch_targets": 0,
        "shared_switch_duration_targets": 0,
        "duration_class_changes_on_shared_switches": 0,
    }

    for key, utterance in frame.sort_values(GROUP_FIELDS + ["token_index"]).groupby(
        GROUP_FIELDS,
        sort=False,
    ):
        language_pair = str(key[0])
        first_language, second_language = language_pair.split("-", 1)
        segmentation_language = next(
            (
                language
                for language in (first_language, second_language)
                if language in {"Chinese", "Cantonese", "Japanese"}
            ),
            None,
        )
        text = str(utterance.iloc[0]["text"])
        words = tokenize_mixed_text(text, language=segmentation_language)
        lexical_words = [
            word for word in words if not is_punctuation_or_symbol_token(word)
        ]
        audited_words = utterance["token"].astype(str).tolist()
        if lexical_words != audited_words:
            raise ValueError(
                f"audit tokens do not align with reconstructed text for {key}: "
                f"{audited_words!r} != {lexical_words!r}"
            )

        automatic_lexical = utterance["automatic_lid"].astype(str).tolist()
        gold_lexical = utterance["gold_lid"].astype(str).tolist()
        per_utterance = {}
        for approach, neutral in (("keep", False), ("neutral", True)):
            automatic_lids = _with_punctuation_lids(
                words, automatic_lexical, neutral=neutral
            )
            gold_lids = _with_punctuation_lids(words, gold_lexical, neutral=neutral)
            automatic_switch, automatic_duration = build_predictive_labels(
                automatic_lids
            )
            gold_switch, gold_duration = build_predictive_labels(gold_lids)
            near_punctuation = [
                is_punctuation_or_symbol_token(words[position])
                or is_punctuation_or_symbol_token(words[position + 1])
                for position in range(len(words) - 1)
            ] + [False]

            approach_data[approach]["automatic_switch"].extend(automatic_switch)
            approach_data[approach]["gold_switch"].extend(gold_switch)
            approach_data[approach]["automatic_duration"].extend(automatic_duration)
            approach_data[approach]["gold_duration"].extend(gold_duration)
            approach_data[approach]["near_punctuation"].extend(near_punctuation)
            per_utterance[approach] = (
                automatic_switch,
                automatic_duration,
            )

        keep_switch, keep_duration = per_utterance["keep"]
        neutral_switch, neutral_duration = per_utterance["neutral"]
        for position in range(len(words)):
            keep_valid = keep_switch[position] != IGNORE_INDEX
            neutral_valid = neutral_switch[position] != IGNORE_INDEX
            comparison["shared_valid_switch_targets"] += int(keep_valid and neutral_valid)
            comparison["keep_only_switch_targets"] += int(keep_valid and not neutral_valid)
            comparison["neutral_only_switch_targets"] += int(neutral_valid and not keep_valid)
            comparison["switch_value_changes_on_shared_targets"] += int(
                keep_valid
                and neutral_valid
                and keep_switch[position] != neutral_switch[position]
            )
            shared_duration = keep_duration[position] >= 0 and neutral_duration[position] >= 0
            comparison["shared_switch_duration_targets"] += int(shared_duration)
            comparison["duration_class_changes_on_shared_switches"] += int(
                shared_duration and keep_duration[position] != neutral_duration[position]
            )

    scored = {}
    rows = []
    for approach, values in approach_data.items():
        positions = list(range(len(values["gold_switch"])))
        overall = _score_positions(
            values["automatic_switch"], values["gold_switch"], positions
        )
        punctuation_positions = [
            position
            for position, near in enumerate(values["near_punctuation"])
            if near
        ]
        punctuation = _score_positions(
            values["automatic_switch"],
            values["gold_switch"],
            punctuation_positions,
        )
        duration_positions = [
            position
            for position, gold_duration in enumerate(values["gold_duration"])
            if gold_duration >= 0 and values["automatic_duration"][position] >= 0
        ]
        duration_correct = sum(
            values["automatic_duration"][position] == values["gold_duration"][position]
            for position in duration_positions
        )
        duration = {
            "support": len(duration_positions),
            "correct": duration_correct,
            "accuracy": _divide(duration_correct, len(duration_positions)),
        }
        scored[approach] = {
            "switch_labels": overall,
            "near_punctuation_switch_labels": punctuation,
            "duration_labels_at_gold_switches": duration,
        }
        rows.append(
            {
                "Approach": "A: keep/inherit" if approach == "keep" else "C: keep/neutral",
                "Boundary F1": overall["f1"],
                "Boundary Precision": overall["precision"],
                "Boundary Recall": overall["recall"],
                "Near-Punctuation F1": punctuation["f1"],
                "Duration Accuracy": duration["accuracy"],
                "Valid Switch Targets": overall["support"],
                "Gold Switches": overall["positive_support"],
            }
        )

    result = {
        "definition": {
            "A": "keep punctuation and inherit a nearby lexical LID",
            "C": "keep punctuation as neutral; ignore targets into neutral spans and anchor the lexical boundary on the final neutral token",
            "duration_C": "count lexical subwords only; neutral punctuation does not end or lengthen a language burst",
        },
        "audit_scope": {
            "language_pairs": int(frame["language_pair"].nunique()),
            "utterances": int(frame.groupby(GROUP_FIELDS).ngroups),
            "manually_labeled_lexical_tokens": int(len(frame)),
        },
        "approaches": scored,
        "automatic_A_vs_C": comparison,
    }
    return result, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default="results/lid_audit_14pairs.csv")
    parser.add_argument(
        "--output-json",
        default="results/punctuation_ac_label_comparison.json",
    )
    parser.add_argument(
        "--output-csv",
        default="results/punctuation_ac_label_comparison.csv",
    )
    args = parser.parse_args()

    result, summary = compare_audit(
        pd.read_csv(args.audit, keep_default_na=False)
    )
    json_path = Path(args.output_json)
    csv_path = Path(args.output_csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary.to_csv(csv_path, index=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Wrote {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
