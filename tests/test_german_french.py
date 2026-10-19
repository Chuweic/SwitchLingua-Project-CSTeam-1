import pytest

from src.switchlingua_labels import (
    LANGUAGE_NAME_TO_CODE,
    SELECTED_LANGUAGE_PAIRS,
    PairRestrictedWordLID,
    encode_and_label,
)
from tests.test_token_labels import FakeTokenizer


def test_german_french_is_configured():
    assert "German-French" in SELECTED_LANGUAGE_PAIRS
    assert LANGUAGE_NAME_TO_CODE["German"] == "de"
    assert LANGUAGE_NAME_TO_CODE["French"] == "fr"


def test_german_french_real_word_lid():
    pytest.importorskip("lingua")

    lid = PairRestrictedWordLID()
    words, labels = lid.annotate(
        "Guten Morgen bonjour merci",
        "German",
        "French",
    )

    assert words == ["Guten", "Morgen", "bonjour", "merci"]
    assert labels == ["de", "de", "fr", "fr"]


def test_reversing_language_order_does_not_change_labels():
    pytest.importorskip("lingua")

    lid = PairRestrictedWordLID()
    text = "Guten Morgen bonjour merci"

    _, forward = lid.annotate(text, "German", "French")
    _, reverse = lid.annotate(text, "French", "German")

    assert forward == reverse


def test_german_french_switch_and_duration_labels():
    result = encode_and_label(
        FakeTokenizer(),
        words=["Guten", "Morgen", "bonjour", "merci"],
        word_language_ids=["de", "de", "fr", "fr"],
    )

    assert result["subword_language_ids"] == [
        None,
        "de", "de",
        "de", "de",
        "fr", "fr",
        "fr", "fr",
        None,
    ]

    assert result["switch_labels"] == [
        -100,
        0, 0,
        0, 1,
        0, 0,
        0, -100,
        -100,
    ]

    assert result["duration_labels"][4] == 1