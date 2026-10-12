# Member B Tokenization and Predictive Labeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and test the Member B preprocessing component that aligns word-level language IDs to model sub-words and emits correctly shifted Switch and Duration labels.

**Architecture:** Add one focused pure-Python module under `src/` with a small tokenizer protocol boundary. The module will consume pre-tokenized words plus equal-length language IDs, call a Hugging Face-style fast tokenizer with `is_split_into_words=True`, propagate each word language ID to all its sub-words, and generate labels only across adjacent valid sub-words. Unit tests will use a deterministic fake tokenizer, so label semantics and edge cases are testable without network access or model downloads.

**Tech Stack:** Python 3, pytest, optional Hugging Face fast tokenizer at integration time.

---

### Task 1: Add failing tests for alignment and label semantics

**Files:**
- Create: `tests/test_token_labels.py`

- [ ] **Step 1: Write tests for sub-word inheritance and no false intra-word switch**

```python
def test_split_word_pieces_inherit_language_and_do_not_switch_inside_word():
    result = encode_and_label(
        tokenizer=FakeTokenizer(),
        words=["hello", "世界"],
        word_language_ids=["en", "zh"],
    )
    assert result["subword_language_ids"] == [None, "en", "en", "zh", None]
    assert result["switch_labels"] == [-100, 0, 1, -100, -100]
```

- [ ] **Step 2: Add tests for duration bins and non-switch masking**

```python
def test_duration_uses_upcoming_burst_length_and_bins_boundaries():
    assert labels_for_language_sequence(["a", "b"])["duration_labels"] == [-100, 0, -100]
    assert labels_for_language_sequence(["a", "b", "b", "b", "b", "b", "b"])["duration_labels"] == [-100, 1, -1, -1, -1, -1, -100]
    assert labels_for_language_sequence(["a"] + ["b"] * 7)["duration_labels"] == [-100, 2, -1, -1, -1, -1, -1, -1, -100]
```

- [ ] **Step 3: Add tests for special tokens, padding, input validation, and length alignment**

```python
def test_special_padding_and_last_position_are_ignored():
    result = encode_and_label(FakeTokenizer(), ["hi"], ["en"], padding="max_length", max_length=5)
    assert result["switch_labels"] == [-100, -100, -100, -100, -100]
    assert result["duration_labels"] == [-100, -100, -100, -100, -100]

def test_mismatched_words_and_language_ids_raise_value_error():
    with pytest.raises(ValueError, match="same length"):
        encode_and_label(FakeTokenizer(), ["hello"], ["en", "zh"])
```

- [ ] **Step 4: Run the focused tests and verify they fail because the module is missing**

Run: `cd /Users/mesutyi/Desktop/SwitchLingua-Project-CSTeam-1 && PYTHONPATH=. pytest -q tests/test_token_labels.py`

Expected: collection or import failure because `src/switchlingua_labels.py` does not yet exist.

### Task 2: Implement the tokenizer alignment and label generator

**Files:**
- Create: `src/__init__.py`
- Create: `src/switchlingua_labels.py`

- [ ] **Step 1: Define constants and the output contract**

Implement `IGNORE_INDEX = -100`, `NON_SWITCH_DURATION = -1`, and a public `encode_and_label(tokenizer, words, word_language_ids, *, padding=False, max_length=None) -> dict` returning tokenizer fields plus `word_ids`, `subword_language_ids`, `switch_labels`, and `duration_labels`.

- [ ] **Step 2: Validate inputs and require fast-tokenizer word alignment**

Reject mismatched list lengths, empty language IDs, empty word lists, missing callable `word_ids()`, and any valid sub-word whose word ID is outside the input range. Call the tokenizer as:

```python
encoding = tokenizer(
    words,
    is_split_into_words=True,
    return_attention_mask=True,
    padding=padding,
    max_length=max_length,
    truncation=max_length is not None,
)
word_ids = list(encoding.word_ids())
```

- [ ] **Step 3: Propagate word LIDs to sub-words and identify valid positions**

Use `None` for special/padding positions and the corresponding language ID for every valid sub-word. Treat only positions with a non-`None` word ID and an attention mask value of 1 as valid.

- [ ] **Step 4: Generate shifted Switch labels and duration labels**

For each valid position `t`, find the next valid position. Set switch to `1` when the next LID differs, else `0`. If there is no next valid position, set both labels to `IGNORE_INDEX`. For a switch, count consecutive next sub-words with the new LID, map 1–2/3–6/7+ to 0/1/2, and set duration to `NON_SWITCH_DURATION` at valid non-switch positions.

- [ ] **Step 5: Run focused tests and make them pass**

Run: `cd /Users/mesutyi/Desktop/SwitchLingua-Project-CSTeam-1 && pytest -q tests/test_token_labels.py`

Expected: all focused tests pass.

### Task 3: Add a complete Arabic–English conversion example and documentation

**Files:**
- Create: `examples/member_b_label_example.py`
- Modify: `README.md`

- [ ] **Step 1: Add a deterministic runnable example**

Show `words = ["أنا", "تابعت", "honestly"]`, `word_language_ids = ["ar", "ar", "en"]`, a tokenizer call, aligned sub-word LIDs, and the resulting shifted labels. The example must state that word-level LIDs are supplied annotations/inferred preprocessing inputs, not generated by the tokenizer.

- [ ] **Step 2: Document the Member B interface and label semantics**

Add a short README section with the input contract, the `word_ids()` mapping rule, duration bins, ignore values, and the focused test command. Explicitly note that the current raw SwitchLingua export does not contain token-level LID annotations and that an LID preparation step is required before calling this module.

- [ ] **Step 3: Run the example and tests**

Run: `cd /Users/mesutyi/Desktop/SwitchLingua-Project-CSTeam-1 && PYTHONPATH=. python examples/member_b_label_example.py && PYTHONPATH=. pytest -q`

Expected: the example prints the complete aligned record and pytest reports zero failures.

### Task 4: Verify scope, formatting, and reproducibility

**Files:**
- Inspect: `src/switchlingua_labels.py`, `tests/test_token_labels.py`, `examples/member_b_label_example.py`, `README.md`

- [ ] **Step 1: Run repository diff checks**

Run: `git diff --check`

Expected: no output and exit code 0.

- [ ] **Step 2: Run the full test suite and inspect the final diff**

Run: `PYTHONPATH=. pytest -q && git status --short && git diff --stat`

Expected: all tests pass; only Member B source, tests, example, and documentation are changed.

- [ ] **Step 3: Confirm the requirements checklist**

Confirm the implementation visibly covers sub-word/LID alignment, Switch Label rule, Duration Label rule, special/padding/end handling, `t → t+1` shifting, one complete example, and tests for duration boundaries.
