import pandas as pd
import pytest

from src.lid_audit import build_lid_audit, score_lid_audit


def test_build_audit_is_balanced_and_excludes_punctuation():
    table = pd.DataFrame(
        [
            {
                "row_id": 1,
                "first_language": "Arabic",
                "second_language": "English",
                "utterances": ["مرحبا, hello!"],
                "language_pair": "Arabic-English",
            },
            {
                "row_id": 2,
                "first_language": "German",
                "second_language": "French",
                "utterances": ["Guten Tag, bonjour!"],
                "language_pair": "German-French",
            },
        ]
    )

    audit = build_lid_audit(
        table,
        language_pairs=["Arabic-English"],
        utterances_per_pair=1,
        seed=7,
    )

    assert set(audit["language_pair"]) == {"Arabic-English"}
    assert audit["token"].tolist() == ["مرحبا", "hello"]
    assert audit["automatic_lid"].tolist() == ["ar", "en"]
    assert audit["gold_lid"].eq("").all()


def test_score_audit_reports_token_accuracy_and_switch_boundary_f1():
    audit = pd.DataFrame(
        {
            "language_pair": ["Spanish-English"] * 4,
            "row_id": [1] * 4,
            "utterance_index": [0] * 4,
            "token_index": [0, 1, 2, 3],
            "automatic_lid": ["es", "es", "en", "en"],
            "gold_lid": ["es", "es", "en", "en"],
        }
    )

    metrics = score_lid_audit(audit)

    assert metrics["overall"]["token_lid_accuracy"] == 1.0
    assert metrics["overall"]["switch_boundary_f1"] == 1.0
    assert metrics["by_language_pair"]["Spanish-English"]["switch_boundary_tp"] == 1


def test_score_audit_ignores_blank_gold_labels_without_joining_gaps():
    audit = pd.DataFrame(
        {
            "language_pair": ["Spanish-English"] * 3,
            "row_id": [1] * 3,
            "utterance_index": [0] * 3,
            "token_index": [0, 1, 2],
            "automatic_lid": ["es", "es", "en"],
            "gold_lid": ["es", "", "en"],
        }
    )

    metrics = score_lid_audit(audit)

    assert metrics["overall"]["token_count"] == 2
    assert metrics["overall"]["token_lid_accuracy"] == 1.0
    assert metrics["overall"]["switch_boundary_tp"] == 0


def test_score_audit_requires_at_least_one_gold_label():
    audit = pd.DataFrame(
        {
            "language_pair": ["Arabic-English"],
            "row_id": [1],
            "utterance_index": [0],
            "token_index": [0],
            "automatic_lid": ["ar"],
            "gold_lid": [""],
        }
    )

    with pytest.raises(ValueError, match="no completed"):
        score_lid_audit(audit)
