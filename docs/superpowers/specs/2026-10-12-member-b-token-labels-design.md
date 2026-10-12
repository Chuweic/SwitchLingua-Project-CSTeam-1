# Member B Tokenization and Predictive Labeling Design

## Goal

Implement the Member B portion of Update 1: convert utterances with word-level language IDs into model sub-word tokens, align each sub-word to its source language, and generate next-token Switch and Duration labels with correct boundary handling.

## Scope

This component accepts one utterance at a time together with an equal-length word/token sequence and its language IDs. It does not implement the Streaming DataLoader, causal attention mask, model, or baseline metrics owned by other team members.

## Data contract

The primary input is:

```python
text: str
words: list[str]
word_language_ids: list[str]
```

`words` and `word_language_ids` must have the same length. The words must concatenate to the utterance under the caller's normalization/tokenization policy; the labeler does not infer missing language IDs silently.

The tokenizer must be a Hugging Face fast tokenizer and must support `is_split_into_words=True` and `word_ids()`.

## Alignment

The tokenizer receives `words` as pre-tokenized input. Every non-special sub-word inherits the language ID of its source word. If a word is split into multiple sub-words, all pieces receive the same language ID. Special tokens, padding, and unmapped positions receive an ignore marker and are excluded from labels and metrics.

The output preserves tokenizer order and includes `input_ids`, `attention_mask`, `subword_language_ids`, and `word_ids`.

## Predictive labels

For each valid sub-word position `t`, the target is the next valid sub-word at `t+1`:

```text
switch_label[t] = 1 if lid[t+1] != lid[t] else 0
```

The final valid sub-word has no next-token target and receives the ignore marker. A switch at `t` starts a burst at `t+1`; the burst length is measured in consecutive sub-word tokens with the new language. Duration bins are:

```text
0: 1–2 sub-word tokens
1: 3–6 sub-word tokens
2: 7+ sub-word tokens
-1: valid position with no switch
ignore: special token, padding, or no next-token target
```

Duration is only supervised at positions where `switch_label[t] == 1`. This matches the project requirement that the duration head use `ignore_index=-1` for non-switch positions while keeping structural invalid positions separately ignorable.

## Validation and errors

- Reject mismatched `words` and `word_language_ids` lengths.
- Reject empty language IDs for valid words.
- Reject tokenizers that do not expose `word_ids()`.
- Verify that every valid sub-word has exactly one source word and language ID.
- Verify that labels have the same sequence length as `input_ids`.
- Include a complete Arabic–English example and boundary tests for one-token, two-token, six-token, and seven-token bursts.

## Testing strategy

Use a small deterministic fake fast-tokenizer interface in unit tests so label logic is testable without downloading a pretrained model. Add an integration-style test with a real tokenizer only when the environment has the required Transformers model cached. Tests must cover sub-word inheritance, no false switch inside a split word, next-position alignment, duration bin boundaries, special-token/padding ignore behavior, and invalid input rejection.
