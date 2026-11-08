"""Prepare and score a balanced manual audit of automatic word-level LIDs."""

import argparse
import json
from pathlib import Path
import random
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd

from src.switchlingua_labels import (
    SELECTED_LANGUAGE_PAIRS,
    PairRestrictedWordLID,
    apply_punctuation_policy,
)


AUDIT_COLUMNS = (
    "language_pair",
    "row_id",
    "utterance_index",
    "text",
    "token_index",
    "token",
    "automatic_lid",
    "gold_lid",
    "is_correct",
    "error_type",
    "notes",
)


def _nonempty_label(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    label = str(value).strip()
    return label or None


def build_lid_audit(
    table: pd.DataFrame,
    *,
    language_pairs: Sequence[str] = SELECTED_LANGUAGE_PAIRS,
    utterances_per_pair: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a deterministic, pair-balanced token audit template.

    Punctuation is excluded because it has no intrinsic language identity. The
    blank ``gold_lid`` and error-analysis columns are intended for manual entry.
    """

    if utterances_per_pair <= 0:
        raise ValueError("utterances_per_pair must be positive")
    required = {
        "row_id",
        "first_language",
        "second_language",
        "utterances",
        "language_pair",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"audit table is missing columns: {missing}")

    rng = random.Random(seed)
    lid = PairRestrictedWordLID()
    audit_rows = []
    for pair in language_pairs:
        candidates = []
        pair_table = table[table["language_pair"] == pair]
        for row in pair_table.itertuples(index=False):
            for utterance_index, text in enumerate(row.utterances):
                if isinstance(text, str) and text.strip():
                    candidates.append(
                        (
                            int(row.row_id),
                            utterance_index,
                            text,
                            row.first_language,
                            row.second_language,
                        )
                    )
        rng.shuffle(candidates)
        for row_id, utterance_index, text, first_language, second_language in candidates[
            :utterances_per_pair
        ]:
            words, automatic_lids = lid.annotate(text, first_language, second_language)
            words, automatic_lids = apply_punctuation_policy(
                words,
                automatic_lids,
                mode="drop",
            )
            for token_index, (token, automatic_lid) in enumerate(
                zip(words, automatic_lids)
            ):
                audit_rows.append(
                    {
                        "language_pair": pair,
                        "row_id": row_id,
                        "utterance_index": utterance_index,
                        "text": text,
                        "token_index": token_index,
                        "token": token,
                        "automatic_lid": automatic_lid,
                        "gold_lid": "",
                        "is_correct": "",
                        "error_type": "",
                        "notes": "",
                    }
                )
    return pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)


def prepare_lid_audit(
    parquet_path: str,
    *,
    language_pairs: Sequence[str] = SELECTED_LANGUAGE_PAIRS,
    utterances_per_pair: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Read a SwitchLingua split and return its manual LID audit template."""

    return build_lid_audit(
        pd.read_parquet(parquet_path),
        language_pairs=language_pairs,
        utterances_per_pair=utterances_per_pair,
        seed=seed,
    )


def _divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _score_rows(frame: pd.DataFrame) -> Dict[str, Any]:
    gold = frame["gold_lid"].map(_nonempty_label)
    automatic = frame["automatic_lid"].map(_nonempty_label)
    labeled = gold.notna()
    token_count = int(labeled.sum())
    token_correct = int(((gold == automatic) & labeled).sum())

    tp = fp = fn = 0
    group_fields = ["language_pair", "row_id", "utterance_index"]
    for _, utterance in frame.sort_values(group_fields + ["token_index"]).groupby(
        group_fields,
        sort=False,
    ):
        rows = list(utterance.to_dict(orient="records"))
        for current, following in zip(rows, rows[1:]):
            if int(following["token_index"]) != int(current["token_index"]) + 1:
                continue
            current_gold = _nonempty_label(current["gold_lid"])
            following_gold = _nonempty_label(following["gold_lid"])
            current_auto = _nonempty_label(current["automatic_lid"])
            following_auto = _nonempty_label(following["automatic_lid"])
            if None in (current_gold, following_gold, current_auto, following_auto):
                continue
            gold_switch = current_gold != following_gold
            automatic_switch = current_auto != following_auto
            tp += int(gold_switch and automatic_switch)
            fp += int(not gold_switch and automatic_switch)
            fn += int(gold_switch and not automatic_switch)

    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    return {
        "token_count": token_count,
        "token_correct": token_correct,
        "token_lid_accuracy": _divide(token_correct, token_count),
        "switch_boundary_f1": _divide(2 * precision * recall, precision + recall),
        "switch_boundary_precision": precision,
        "switch_boundary_recall": recall,
        "switch_boundary_tp": tp,
        "switch_boundary_fp": fp,
        "switch_boundary_fn": fn,
    }


def score_lid_audit(frame: pd.DataFrame) -> Dict[str, Any]:
    """Score completed gold LIDs overall and separately for every pair."""

    required = {
        "language_pair",
        "row_id",
        "utterance_index",
        "token_index",
        "automatic_lid",
        "gold_lid",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"audit file is missing columns: {missing}")
    if not frame["gold_lid"].map(_nonempty_label).notna().any():
        raise ValueError("audit file has no completed gold_lid values")

    by_pair = {
        str(pair): _score_rows(pair_frame)
        for pair, pair_frame in frame.groupby("language_pair", sort=True)
        if pair_frame["gold_lid"].map(_nonempty_label).notna().any()
    }
    return {
        "overall": _score_rows(frame),
        "by_language_pair": by_pair,
    }


def _write_json(data: Mapping[str, Any], output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create a balanced manual-audit CSV.")
    prepare.add_argument("--data", default="data/processed/validation.parquet")
    prepare.add_argument("--output", default="results/lid_audit_14pairs.csv")
    prepare.add_argument("--utterances-per-pair", type=int, default=10)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--pairs", nargs="+", default=list(SELECTED_LANGUAGE_PAIRS))

    score = subparsers.add_parser("score", help="Score a manually completed audit CSV.")
    score.add_argument("--audit", default="results/lid_audit_14pairs.csv")
    score.add_argument("--output", default="results/lid_audit_14pairs_metrics.json")

    args = parser.parse_args()
    if args.command == "prepare":
        audit = prepare_lid_audit(
            args.data,
            language_pairs=args.pairs,
            utterances_per_pair=args.utterances_per_pair,
            seed=args.seed,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(output, index=False)
        print(f"Wrote {len(audit)} token rows to {output}")
        return

    metrics = score_lid_audit(pd.read_csv(args.audit, keep_default_na=False))
    _write_json(metrics, args.output)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
