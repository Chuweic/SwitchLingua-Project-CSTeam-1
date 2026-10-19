import pytest

from src import switchlingua_labels

IGNORE_INDEX = switchlingua_labels.IGNORE_INDEX
NON_SWITCH_DURATION = switchlingua_labels.NON_SWITCH_DURATION


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


def test_pair_restricted_lid_handles_script_distinct_text():
    words, lids = switchlingua_labels.PairRestrictedWordLID().annotate(
        "أنا like this.", "Arabic", "English"
    )
    assert words == ["أنا", "like", "this", "."]
    assert lids == ["ar", "en", "en", "en"]
