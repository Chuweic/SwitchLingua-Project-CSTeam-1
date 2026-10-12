"""Align word-level language IDs to sub-words and create predictive labels."""

from typing import Any, Dict, List, Optional, Sequence


IGNORE_INDEX = -100
NON_SWITCH_DURATION = -1


def _duration_class(burst_length: int) -> int:
    if burst_length <= 0:
        raise ValueError("burst length must be positive")
    if burst_length <= 2:
        return 0
    if burst_length <= 6:
        return 1
    return 2


def _validate_inputs(words: Sequence[str], word_language_ids: Sequence[str]) -> None:
    if not words:
        raise ValueError("words must not be empty")
    if len(words) != len(word_language_ids):
        raise ValueError("words and word_language_ids must have the same length")
    if any(not isinstance(word, str) or not word for word in words):
        raise ValueError("words must contain non-empty strings")
    if any(not isinstance(language_id, str) or not language_id.strip() for language_id in word_language_ids):
        raise ValueError("word_language_ids must contain non-empty strings")


def _get_word_ids(encoding: Any) -> List[Optional[int]]:
    word_ids_method = getattr(encoding, "word_ids", None)
    if not callable(word_ids_method):
        raise TypeError("tokenizer output must provide a callable word_ids() method")
    return list(word_ids_method())


def encode_and_label(
    tokenizer: Any,
    words: Sequence[str],
    word_language_ids: Sequence[str],
    *,
    padding: Any = False,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Tokenize pre-segmented words and generate next-sub-word labels.

    ``word_language_ids`` supplies the language annotation for each input word;
    the tokenizer only supplies the word-to-sub-word alignment.  Duration is
    measured in consecutive sub-words in the upcoming language burst.
    """

    _validate_inputs(words, word_language_ids)
    if not callable(tokenizer):
        raise TypeError("tokenizer must be callable")

    encoding = tokenizer(
        list(words),
        is_split_into_words=True,
        return_attention_mask=True,
        padding=padding,
        max_length=max_length,
        truncation=max_length is not None,
    )
    word_ids = _get_word_ids(encoding)

    try:
        input_ids = list(encoding["input_ids"])
        attention_mask = list(encoding["attention_mask"])
    except (KeyError, TypeError) as exc:
        raise TypeError("tokenizer output must contain input_ids and attention_mask") from exc

    sequence_length = len(input_ids)
    if len(word_ids) != sequence_length or len(attention_mask) != sequence_length:
        raise ValueError("tokenizer fields must have the same sequence length")

    subword_language_ids: List[Optional[str]] = []
    valid_positions: List[int] = []
    for position, (word_id, mask_value) in enumerate(zip(word_ids, attention_mask)):
        is_valid = word_id is not None and bool(mask_value)
        if is_valid:
            if not isinstance(word_id, int) or word_id < 0 or word_id >= len(words):
                raise ValueError("tokenizer returned an out-of-range word ID")
            subword_language_ids.append(word_language_ids[word_id])
            valid_positions.append(position)
        else:
            subword_language_ids.append(None)

    switch_labels = [IGNORE_INDEX] * sequence_length
    duration_labels = [IGNORE_INDEX] * sequence_length

    for valid_index, position in enumerate(valid_positions[:-1]):
        next_position = valid_positions[valid_index + 1]
        current_language = subword_language_ids[position]
        next_language = subword_language_ids[next_position]
        assert current_language is not None
        assert next_language is not None

        is_switch = current_language != next_language
        switch_labels[position] = int(is_switch)
        if not is_switch:
            duration_labels[position] = NON_SWITCH_DURATION
            continue

        burst_length = 0
        for burst_position in valid_positions[valid_index + 1 :]:
            if subword_language_ids[burst_position] != next_language:
                break
            burst_length += 1
        duration_labels[position] = _duration_class(burst_length)

    result = dict(encoding)
    result.update(
        {
            "word_ids": word_ids,
            "subword_language_ids": subword_language_ids,
            "switch_labels": switch_labels,
            "duration_labels": duration_labels,
        }
    )
    return result
