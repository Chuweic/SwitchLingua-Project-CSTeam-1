"""Runnable Member B example using a tiny FastTokenizer-compatible tokenizer.

Replace DemoTokenizer with a Hugging Face fast tokenizer in the full pipeline.
The word-level language IDs below are supplied preprocessing annotations; they
are not inferred by the tokenizer.
"""

import io
import sys

from src.switchlingua_labels import encode_and_label


if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


class DemoEncoding(dict):
    def __init__(self, input_ids, attention_mask, word_ids):
        super().__init__(input_ids=input_ids, attention_mask=attention_mask)
        self._word_ids = word_ids

    def word_ids(self):
        return self._word_ids


class DemoTokenizer:
    """Small deterministic stand-in for a Hugging Face fast tokenizer."""

    def __call__(self, words, *, is_split_into_words, return_attention_mask, padding=False, max_length=None, truncation=False):
        assert is_split_into_words and return_attention_mask
        input_ids = [101]
        word_ids = [None]
        for index, word in enumerate(words):
            # Deliberately split longer words to demonstrate LID inheritance.
            pieces = [word] if len(word) <= 4 else [word[:4], word[4:]]
            input_ids.extend(200 + index * 10 + piece for piece in range(len(pieces)))
            word_ids.extend([index] * len(pieces))
        input_ids.append(102)
        word_ids.append(None)
        return DemoEncoding(input_ids, [1] * len(input_ids), word_ids)


def main():
    words = ["أنا", "تابعت", "honestly"]
    word_language_ids = ["ar", "ar", "en"]
    result = encode_and_label(DemoTokenizer(), words, word_language_ids)

    print("words:", words)
    print("word_language_ids:", word_language_ids)
    print("word_ids:", result["word_ids"])
    print("subword_language_ids:", result["subword_language_ids"])
    print("switch_labels:", result["switch_labels"])
    print("duration_labels:", result["duration_labels"])


if __name__ == "__main__":
    main()
