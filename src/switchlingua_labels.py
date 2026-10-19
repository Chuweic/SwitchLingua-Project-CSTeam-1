"""Create word LIDs, align them to sub-words, and build predictive labels."""

import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple


IGNORE_INDEX = -100
NON_SWITCH_DURATION = -1
DEFAULT_TOKENIZER = "xlm-roberta-base"
SELECTED_LANGUAGE_PAIRS = (
    "Arabic-English",
    "Cantonese-English",
    "Chinese-English",
    "French-English",
    "German-English",
    "German-French",
    "Hindi-English",
    "Italian-English",
    "Japanese-English",
    "Korean-English",
    "Malay-English",
    "Philippines-English",
    "Russian-English",
    "Spanish-English",
    "Vietnamese-English",
)

LANGUAGE_NAME_TO_CODE = {
    "Arabic": "ar",
    "Cantonese": "yue",
    "Chinese": "zh",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Malay": "ms",
    "Philippines": "tl",
    "Russian": "ru",
    "Spanish": "es",
    "Vietnamese": "vi",
}

_CODE_TO_LINGUA_NAME = {
    "ar": "ARABIC",
    "yue": "CHINESE",
    "zh": "CHINESE",
    "en": "ENGLISH",
    "fr": "FRENCH",
    "de": "GERMAN",
    "hi": "HINDI",
    "it": "ITALIAN",
    "ja": "JAPANESE",
    "ko": "KOREAN",
    "ms": "MALAY",
    "tl": "TAGALOG",
    "ru": "RUSSIAN",
    "es": "SPANISH",
    "vi": "VIETNAMESE",
}

_SCRIPT_CODES = {
    "ARABIC": {"ar"},
    "DEVANAGARI": {"hi"},
    "CYRILLIC": {"ru"},
    "HAN": {"yue", "zh", "ja"},
    "HIRAGANA": {"ja"},
    "KATAKANA": {"ja"},
    "HANGUL": {"ko"},
}


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


# --- Real SwitchLingua word-LID and parquet adapter -------------------------


def _char_script(char: str) -> str:
    category = unicodedata.category(char)
    if category.startswith(("P", "S")):
        return "PUNCT"
    if category.startswith("M") or char in "'’":
        return "JOINER"
    if char.isdigit():
        return "DIGIT"
    code = ord(char)
    if 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F or 0x08A0 <= code <= 0x08FF:
        return "ARABIC"
    if 0x0900 <= code <= 0x097F:
        return "DEVANAGARI"
    if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
        return "HAN"
    if 0x3040 <= code <= 0x309F:
        return "HIRAGANA"
    if 0x30A0 <= code <= 0x30FF:
        return "KATAKANA"
    if 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:
        return "HANGUL"
    if 0x0400 <= code <= 0x052F:
        return "CYRILLIC"
    if char.isalpha() and "LATIN" in unicodedata.name(char, ""):
        return "LATIN"
    return "PUNCT"


def tokenize_mixed_text(text: str) -> List[str]:
    """Split at whitespace, punctuation, and Unicode-script changes."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    tokens: List[str] = []
    current: List[str] = []
    current_script: Optional[str] = None

    def flush() -> None:
        nonlocal current, current_script
        if current:
            tokens.append("".join(current))
        current = []
        current_script = None

    for char in text:
        if char.isspace():
            flush()
            continue
        script = _char_script(char)
        if script == "PUNCT":
            flush()
            tokens.append(char)
            continue
        if script == "JOINER" and current:
            current.append(char)
            continue
        compatible = (
            current_script is None
            or script == current_script
            or {script, current_script} <= {"LATIN", "DIGIT"}
        )
        if not compatible:
            flush()
        current.append(char)
        if current_script is None:
            current_script = script
    flush()
    return tokens


class PairRestrictedWordLID:
    """Create explicit weak word LIDs from the row's two known languages."""

    def __init__(self) -> None:
        self._detectors: Dict[Tuple[str, str], Any] = {}

    def _detector(self, codes: Tuple[str, str]) -> Any:
        key = tuple(sorted(codes))
        if key not in self._detectors:
            try:
                from lingua import Language, LanguageDetectorBuilder
            except ImportError as exc:
                raise ImportError("install lingua-language-detector from requirements.txt") from exc
            languages = []
            for code in key:
                language = getattr(Language, _CODE_TO_LINGUA_NAME[code])
                if language not in languages:
                    languages.append(language)
            self._detectors[key] = LanguageDetectorBuilder.from_languages(*languages).build()
        return self._detectors[key]

    @staticmethod
    def _script_label(token: str, candidates: Tuple[str, str]) -> Optional[str]:
        scripts = {_char_script(char) for char in token if char.isalpha()}
        for script, possible in _SCRIPT_CODES.items():
            matches = set(candidates) & possible
            if script in scripts and len(matches) == 1:
                return next(iter(matches))
        if "LATIN" in scripts:
            non_latin = set().union(*_SCRIPT_CODES.values())
            latin_candidates = [code for code in candidates if code not in non_latin]
            if len(latin_candidates) == 1:
                return latin_candidates[0]
        return None

    def annotate(self, text: str, first_language: str, second_language: str) -> Tuple[List[str], List[str]]:
        try:
            candidates = (LANGUAGE_NAME_TO_CODE[first_language], LANGUAGE_NAME_TO_CODE[second_language])
        except KeyError as exc:
            raise ValueError(f"unsupported SwitchLingua language: {exc.args[0]!r}") from exc
        words = tokenize_mixed_text(text)
        labels: List[Optional[str]] = []
        for token in words:
            label = self._script_label(token, candidates)
            if label is None and any(char.isalpha() for char in token):
                detected = self._detector(candidates).detect_language_of(token)
                if detected is not None:
                    label = next(
                        (code for code in candidates if _CODE_TO_LINGUA_NAME[code] == detected.name),
                        None,
                    )
            labels.append(label)
        for index, label in enumerate(labels):
            if label is not None:
                continue
            previous = next((labels[j] for j in range(index - 1, -1, -1) if labels[j]), None)
            following = next((labels[j] for j in range(index + 1, len(labels)) if labels[j]), None)
            labels[index] = previous or following or candidates[0]
        return words, [str(label) for label in labels]


def load_fast_tokenizer(model_name: str = DEFAULT_TOKENIZER) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("word-to-subword alignment requires a fast tokenizer")
    return tokenizer


def records_from_parquet(
    path: str,
    tokenizer: Any,
    *,
    language_pairs: Sequence[str] = SELECTED_LANGUAGE_PAIRS,
    max_rows_per_pair: Optional[int] = None,
    max_utterances_per_row: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Convert a real split into the records consumed by the existing loader."""

    import pandas as pd

    table = pd.read_parquet(path)
    required = {"row_id", "first_language", "second_language", "utterances", "language_pair"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"parquet split is missing columns: {missing}")
    table = table[table["language_pair"].isin(language_pairs)]
    if max_rows_per_pair is not None:
        groups = []
        for offset, pair in enumerate(language_pairs):
            group = table[table["language_pair"] == pair]
            groups.append(group.sample(n=min(max_rows_per_pair, len(group)), random_state=seed + offset))
        table = pd.concat(groups, ignore_index=True)

    lid = PairRestrictedWordLID()
    records: List[Dict[str, Any]] = []
    for row in table.itertuples(index=False):
        utterances = list(row.utterances)
        if max_utterances_per_row is not None:
            utterances = utterances[:max_utterances_per_row]
        for utterance_index, text in enumerate(utterances):
            if not isinstance(text, str) or not text.strip():
                continue
            words, word_lids = lid.annotate(text, row.first_language, row.second_language)
            record = encode_and_label(tokenizer, words, word_lids)
            record.update(
                {
                    "row_id": int(row.row_id),
                    "utterance_index": utterance_index,
                    "text": text,
                    "words": words,
                    "word_language_ids": word_lids,
                    "language_pair": row.language_pair,
                    "lid_source": "automatic-weak-label",
                }
            )
            records.append(record)
    return records
