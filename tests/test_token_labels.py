import pytest

from src import switchlingua_labels

IGNORE_INDEX = switchlingua_labels.IGNORE_INDEX
NON_SWITCH_DURATION = switchlingua_labels.NON_SWITCH_DURATION
NEUTRAL_LID = switchlingua_labels.NEUTRAL_LID


class FakeEncoding(dict):
    def __init__(self, input_ids, attention_mask, word_ids):
        super().__init__(input_ids=input_ids, attention_mask=attention_mask)
        self._word_ids = word_ids

    def word_ids(self):
        return self._word_ids


class FakeTokenizer:
    """Deterministic stand-in for a Hugging Face fast tokenizer."""

    def __call__(
        self,
        words,
        *,
        is_split_into_words,
        return_attention_mask,
        padding=False,
        max_length=None,
        truncation=False,
    ):
        assert is_split_into_words is True
        assert return_attention_mask is True
        assert truncation is (max_length is not None)

        input_ids = [101]
        word_ids = [None]
        for word_index, word in enumerate(words):
            pieces = [word] if len(word) <= 2 else [word[:2], word[2:]]
            for piece_index, _ in enumerate(pieces):
                input_ids.append(200 + word_index * 10 + piece_index)
                word_ids.append(word_index)
        input_ids.append(102)
        word_ids.append(None)

        if padding == "max_length":
            if max_length is None:
                raise ValueError("max_length is required for max_length padding")
            input_ids = input_ids[:max_length]
            word_ids = word_ids[:max_length]
            input_ids.extend([0] * (max_length - len(input_ids)))
            word_ids.extend([None] * (max_length - len(word_ids)))

        return FakeEncoding(input_ids, [1 if word_id is not None or token_id in (101, 102) else 0 for token_id, word_id in zip(input_ids, word_ids)], word_ids)


def encode_and_label(tokenizer, words, word_language_ids, **kwargs):
    return switchlingua_labels.encode_and_label(tokenizer, words, word_language_ids, **kwargs)


def labels_for_language_sequence(language_ids):
    words = [f"w{index}" for index in range(len(language_ids))]
    return encode_and_label_impl(FakeTokenizer(), words, language_ids)


def encode_and_label_impl(tokenizer, words, word_language_ids, **kwargs):
    return switchlingua_labels.encode_and_label(tokenizer, words, word_language_ids, **kwargs)


def test_split_word_pieces_inherit_language_and_do_not_switch_inside_word():
    result = encode_and_label_impl(
        tokenizer=FakeTokenizer(),
        words=["hello", "世界"],
        word_language_ids=["en", "zh"],
    )
    assert result["subword_language_ids"] == [None, "en", "en", "zh", None]
    assert result["switch_labels"] == [IGNORE_INDEX, 0, 1, IGNORE_INDEX, IGNORE_INDEX]


def test_duration_uses_upcoming_burst_length_and_bins_boundaries():
    one_token = labels_for_language_sequence(["a", "b"])
    six_tokens = labels_for_language_sequence(["a", "b", "b", "b", "b", "b", "b"])
    seven_tokens = labels_for_language_sequence(["a"] + ["b"] * 7)

    assert one_token["duration_labels"] == [IGNORE_INDEX, 0, IGNORE_INDEX, IGNORE_INDEX]
    assert six_tokens["duration_labels"] == [IGNORE_INDEX, 1, NON_SWITCH_DURATION, NON_SWITCH_DURATION, NON_SWITCH_DURATION, NON_SWITCH_DURATION, NON_SWITCH_DURATION, IGNORE_INDEX, IGNORE_INDEX]
    assert seven_tokens["duration_labels"] == [IGNORE_INDEX, 2] + [NON_SWITCH_DURATION] * 6 + [IGNORE_INDEX, IGNORE_INDEX]


def test_special_padding_and_last_position_are_ignored():
    result = encode_and_label_impl(
        FakeTokenizer(), ["hi"], ["en"], padding="max_length", max_length=5
    )
    assert result["switch_labels"] == [IGNORE_INDEX] * 5
    assert result["duration_labels"] == [IGNORE_INDEX] * 5


def test_mismatched_words_and_language_ids_raise_value_error():
    with pytest.raises(ValueError, match="same length"):
        encode_and_label_impl(FakeTokenizer(), ["hello"], ["en", "zh"])


def test_real_text_tokenizer_splits_script_changes_and_arabic_punctuation():
    assert switchlingua_labels.tokenize_mixed_text("研究team真係好!") == ["研究", "team", "真係好", "!"]
    assert switchlingua_labels.tokenize_mixed_text("الاصطناعي، خاصة") == ["الاصطناعي", "،", "خاصة"]


def test_combining_marks_stay_attached_to_their_base_token():
    assert switchlingua_labels.tokenize_mixed_text("أحياناً أحس") == ["أحياناً", "أحس"]
    assert switchlingua_labels.tokenize_mixed_text("cafe\u0301 noir") == ["cafe\u0301", "noir"]


def test_han_languages_use_word_segmentation_before_model_tokenization():
    assert switchlingua_labels.tokenize_mixed_text(
        "我喜欢机器学习", language="Chinese"
    ) == ["我", "喜欢", "机器", "学习"]
    assert switchlingua_labels.tokenize_mixed_text(
        "研究team真係好", language="Cantonese"
    ) == ["研究", "team", "真", "係", "好"]


def test_mixed_han_and_latin_text_keeps_language_aware_word_units():
    assert switchlingua_labels.tokenize_mixed_text(
        "我like机器", language="Chinese"
    ) == ["我", "like", "机器"]


def test_space_delimited_languages_keep_words_and_split_punctuation():
    assert switchlingua_labels.tokenize_mixed_text(
        "Das ist gut.", language="German"
    ) == ["Das", "ist", "gut", "."]


def test_latin_apostrophes_stay_inside_words():
    assert switchlingua_labels.tokenize_mixed_text(
        "l'homme aujourd'hui", language="French"
    ) == ["l'homme", "aujourd'hui"]


def test_default_scope_has_fourteen_english_pivot_pairs():
    pairs = switchlingua_labels.SELECTED_LANGUAGE_PAIRS

    assert len(pairs) == 14
    assert "German-French" not in pairs
    assert all(pair.endswith("-English") for pair in pairs)


def test_drop_punctuation_preserves_words_and_internal_apostrophes():
    words = ["Bonjour", ",", "l'homme", "!", "😊"]
    lids = ["fr", "fr", "fr", "fr", "fr"]

    kept_words, kept_lids = switchlingua_labels.apply_punctuation_policy(
        words,
        lids,
        mode="drop",
    )

    assert kept_words == ["Bonjour", "l'homme"]
    assert kept_lids == ["fr", "fr"]


def test_keep_punctuation_returns_aligned_copies():
    words = ["hello", "!", "مرحبا"]
    lids = ["en", "en", "ar"]

    kept_words, kept_lids = switchlingua_labels.apply_punctuation_policy(
        words,
        lids,
        mode="keep",
    )

    assert kept_words == words
    assert kept_lids == lids
    assert kept_words is not words
    assert kept_lids is not lids


def test_neutral_punctuation_keeps_input_and_marks_only_symbol_tokens():
    words = ["Bonjour", ",", "l'homme", "!", "😊"]
    lids = ["fr", "fr", "fr", "fr", "fr"]

    neutral_words, neutral_lids = switchlingua_labels.apply_punctuation_policy(
        words,
        lids,
        mode="neutral",
    )

    assert neutral_words == words
    assert neutral_lids == ["fr", NEUTRAL_LID, "fr", NEUTRAL_LID, NEUTRAL_LID]


def test_neutral_punctuation_anchors_one_cross_language_switch():
    words = ["hi", ",", "مرحبا"]
    lids = ["en", NEUTRAL_LID, "ar"]

    result = encode_and_label_impl(FakeTokenizer(), words, lids)

    assert result["subword_language_ids"] == [
        None,
        "en",
        NEUTRAL_LID,
        "ar",
        "ar",
        None,
    ]
    assert result["switch_labels"] == [
        IGNORE_INDEX,
        IGNORE_INDEX,
        1,
        0,
        IGNORE_INDEX,
        IGNORE_INDEX,
    ]
    assert result["duration_labels"] == [
        IGNORE_INDEX,
        IGNORE_INDEX,
        0,
        NON_SWITCH_DURATION,
        IGNORE_INDEX,
        IGNORE_INDEX,
    ]


def test_neutral_punctuation_does_not_create_same_language_switch():
    result = encode_and_label_impl(
        FakeTokenizer(),
        ["hi", ",", "yo"],
        ["en", NEUTRAL_LID, "en"],
    )

    assert 1 not in result["switch_labels"]
    comma_position = result["subword_language_ids"].index(NEUTRAL_LID)
    assert result["switch_labels"][comma_position] == 0
    assert result["duration_labels"][comma_position] == NON_SWITCH_DURATION


def test_consecutive_neutral_punctuation_creates_only_one_boundary_target():
    result = encode_and_label_impl(
        FakeTokenizer(),
        ["hi", ",", "!", "مرحبا"],
        ["en", NEUTRAL_LID, NEUTRAL_LID, "ar"],
    )

    assert result["switch_labels"].count(1) == 1
    neutral_positions = [
        index
        for index, language in enumerate(result["subword_language_ids"])
        if language == NEUTRAL_LID
    ]
    assert result["switch_labels"][neutral_positions[0]] == IGNORE_INDEX
    assert result["switch_labels"][neutral_positions[1]] == 1


def test_leading_and_trailing_neutral_punctuation_are_ignored():
    result = encode_and_label_impl(
        FakeTokenizer(),
        ["!", "hi", "."],
        [NEUTRAL_LID, "en", NEUTRAL_LID],
    )

    neutral_positions = [
        index
        for index, language in enumerate(result["subword_language_ids"])
        if language == NEUTRAL_LID
    ]
    assert all(result["switch_labels"][position] == IGNORE_INDEX for position in neutral_positions)


def test_neutral_punctuation_is_not_counted_in_duration_burst():
    result = encode_and_label_impl(
        FakeTokenizer(),
        ["hi", ",", "aa", "!", "bb", "?", "cc", ".", "dd", "yo"],
        [
            "en",
            NEUTRAL_LID,
            "ar",
            NEUTRAL_LID,
            "ar",
            NEUTRAL_LID,
            "ar",
            NEUTRAL_LID,
            "ar",
            "en",
        ],
    )

    first_switch = result["switch_labels"].index(1)
    assert result["duration_labels"][first_switch] == 1


def test_punctuation_policy_rejects_invalid_mode_and_misaligned_lids():
    with pytest.raises(ValueError, match="punctuation mode"):
        switchlingua_labels.apply_punctuation_policy(["hello"], ["en"], mode="other")
    with pytest.raises(ValueError, match="same length"):
        switchlingua_labels.apply_punctuation_policy(["hello", "!"], ["en"], mode="drop")


def test_labels_are_recomputed_after_punctuation_is_dropped():
    words = ["hello", "!", "مرحبا"]
    lids = ["en", "en", "ar"]
    filtered_words, filtered_lids = switchlingua_labels.apply_punctuation_policy(
        words,
        lids,
        mode="drop",
    )

    result = encode_and_label_impl(FakeTokenizer(), filtered_words, filtered_lids)

    assert filtered_words == ["hello", "مرحبا"]
    assert 1 in result["switch_labels"]
    switch_position = result["switch_labels"].index(1)
    assert result["duration_labels"][switch_position] == 0


def test_parquet_adapter_applies_punctuation_mode_before_labeling(monkeypatch):
    pandas = pytest.importorskip("pandas")
    table = pandas.DataFrame(
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
    monkeypatch.setattr(pandas, "read_parquet", lambda _: table)

    records = switchlingua_labels.records_from_parquet(
        "unused.parquet",
        FakeTokenizer(),
        punctuation_mode="drop",
    )

    assert len(records) == 1
    assert records[0]["language_pair"] == "Arabic-English"
    assert records[0]["words"] == ["مرحبا", "hello"]
    assert records[0]["removed_punctuation_count"] == 2
    assert records[0]["punctuation_mode"] == "drop"
    assert 1 in records[0]["switch_labels"]


def test_parquet_adapter_records_neutral_punctuation_metadata(monkeypatch):
    pandas = pytest.importorskip("pandas")
    table = pandas.DataFrame(
        [
            {
                "row_id": 1,
                "first_language": "Arabic",
                "second_language": "English",
                "utterances": ["مرحبا, hello!"],
                "language_pair": "Arabic-English",
            }
        ]
    )
    monkeypatch.setattr(pandas, "read_parquet", lambda _: table)

    records = switchlingua_labels.records_from_parquet(
        "unused.parquet",
        FakeTokenizer(),
        punctuation_mode="neutral",
    )

    assert len(records) == 1
    assert records[0]["words"] == ["مرحبا", ",", "hello", "!"]
    assert records[0]["word_language_ids"] == [
        "ar",
        NEUTRAL_LID,
        "en",
        NEUTRAL_LID,
    ]
    assert records[0]["removed_punctuation_count"] == 0
    assert records[0]["neutralized_punctuation_count"] == 2
    assert records[0]["punctuation_mode"] == "neutral"


def test_pair_restricted_lid_handles_script_distinct_text():
    words, lids = switchlingua_labels.PairRestrictedWordLID().annotate(
        "أنا like this.", "Arabic", "English"
    )
    assert words == ["أنا", "like", "this", "."]
    assert lids == ["ar", "en", "en", "en"]
