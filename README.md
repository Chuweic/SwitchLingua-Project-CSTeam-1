# SwitchLingua Project - CS Team 1

This repository contains our course project on **predictive multitask learning for streaming code-switching** using the SwitchLingua dataset. Using only the prefix through token `t`, the system predicts whether token `t+1` switches language and the duration of the upcoming language segment.

## Project Goals

The project focuses on:

- Predicting the next-token language switch and upcoming segment duration
- Building a Streaming DataLoader with aligned token, LID, and predictive labels
- Enforcing causal, prefix-only context without future-token leakage
- Evaluating Anticipatory F1 and consistency across diverse language pairs
- Comparing multilingual models, beginning with a current-token-only baseline

## Project Update 1

Update 1 covers:

- selection of typologically diverse language pairs;
- real-data ingestion and streaming batch construction;
- word-to-subword Language ID (LID) alignment;
- next-token switch and duration labels;
- causal, prefix-only attention;
- a current-token-only naive baseline; and
- initial Anticipatory F1 and universality results.

The current test suite contains 21 tests covering label alignment, padding, duration masking, shared LID vocabulary, causal masking, and future leakage.

## Setup

Create and activate a virtual environment, install the dependencies, and run the tests from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest -q
```

Expected test result:

```text
21 passed
```

## Dataset

The project uses the Hugging Face dataset `Shelton1013/SwitchLingua_text`. The data was parsed, normalized, deduplicated, and split at the original conversation level using a stratified 80/10/10 train/validation/test split.

The processed parquet files are stored outside GitHub because of their size. Download them from the team’s [Processed SwitchLingua Data folder](https://drive.google.com/drive/folders/1W4y0YEtRhuCIa31Y591JgslPQya9AR5o?usp=sharing) and place them as follows:

```text
data/processed/
├── train.parquet
├── validation.parquet
└── test.parquet
```

These parquet files are excluded by `.gitignore` and should not be committed.

## Language Pairs and Universality Rationale

The initial experiment uses four complementary pairs:

| Language pair | Rationale |
|---|---|
| Arabic–English | Contrasts Arabic and Latin scripts, right-to-left and left-to-right writing, and Semitic and Germanic languages. |
| Hindi–English | Contrasts Devanagari and Latin scripts and Indo-Aryan and Germanic languages. |
| Cantonese–English | Tests Han/Latin script differences and difficult segmentation in text that may not use whitespace consistently. |
| Spanish–English | Creates a same-script condition in which Unicode script rules alone cannot identify the language. |

Together, these pairs test whether performance remains stable across scripts, writing systems, language families, segmentation patterns, and same-script code-switching.

## Tokenization and Word-Level LID

The fixed tokenizer for Update 1 is the fast `xlm-roberta-base` tokenizer. The raw SwitchLingua data provides utterance text and language-pair metadata but does not provide a language label for every word.

`src/switchlingua_labels.py` addresses this gap with explicitly marked **automatic weak LID labels**:

- Unicode-script rules identify words in script-distinct pairs;
- a pair-restricted Lingua detector handles ambiguous and same-script words; and
- punctuation is assigned using the nearest available neighboring language.

The tokenizer is called with `is_split_into_words=True`. Its `word_ids()` alignment copies each word’s LID to all of its subwords, preventing false switches inside a word that is split into multiple pieces.

These weak labels make the real-data pipeline runnable, but they are not human-verified gold labels.

## Predictive Labels

For every valid subword at position `t`, the labels describe the upcoming token at `t+1`:

```text
switch_label[t] = 1 if LID(t + 1) != LID(t), otherwise 0
```

When a switch occurs, the upcoming language burst is assigned a duration class:

| Duration class | Upcoming burst length |
|---:|---:|
| 0 | 1–2 subword tokens |
| 1 | 3–6 subword tokens |
| 2 | 7 or more subword tokens |

Label sentinels are handled as follows:

| Value | Meaning |
|---:|---|
| `-1` | Valid position, but no language switch occurs |
| `-100` | Invalid target position, special token, padding, or no available next-token target |

Switch loss ignores `switch_label == -100`. Duration loss and duration metrics use only `duration_label >= 0`, which excludes both `-1` and `-100`.

## Streaming DataLoader

`src/streaming_dataloader.py` converts labeled utterances into right-padded PyTorch batches. Each batch contains:

- `input_ids`;
- `attention_mask`;
- `current_lid`;
- `switch_labels`;
- `duration_labels`;
- `language_pair`; and
- `causal_mask`.

Padding positions in label and LID tensors use `-100`. `make_split_loaders()` builds `lid_vocab` from the training data and forces validation and test loaders to reuse the same vocabulary.

Run the DataLoader example with:

```bash
PYTHONPATH=. python examples/member_c_dataloader_example.py
```

When `data/processed/train.parquet` is available, the example prints real source text, automatic word LIDs, aligned labels, batch tensors, causal masks, and selected prefix windows. If the parquet file is unavailable, it falls back to a small demo.

## Causal Constraint

The DataLoader creates a lower-triangular attention mask. At position `t`, the model may attend only to positions `0...t`; future positions and padded keys are blocked.

Full sequences can therefore be processed efficiently while preserving the same information constraint as explicit prefix windows. `iter_prefix_windows()` is also provided for streaming demonstrations.

The tests modify every token after a selected position and verify that outputs through that position remain unchanged. A bidirectional negative control confirms that the same test fails when the causal constraint is removed.

## Naive Current-Token Baseline

The naive baseline estimates:

```text
P(switch at t + 1 | current token ID at t)
```

It learns token statistics from the training split only, chooses the decision threshold on validation, and evaluates once on test. The reported run uses seed 42, up to 500/150/150 rows per language pair for train/validation/test, and up to two utterances per row.

| Test set | Anticipatory F1 | Precision | Recall | Duration accuracy |
|---|---:|---:|---:|---:|
| Overall | **0.4419** | 0.4157 | 0.4717 | 0.4953 |
| Arabic–English | 0.5513 | 0.5284 | 0.5762 | 0.5116 |
| Cantonese–English | 0.4017 | 0.3516 | 0.4685 | 0.5038 |
| Hindi–English | 0.3023 | 0.2253 | 0.4594 | 0.5018 |
| Spanish–English | 0.4885 | 0.5667 | 0.4293 | 0.4816 |

The population standard deviation of pair-level Anticipatory F1 is **0.0937**, which is reported as the initial universality measure. The overall switch rate is only 6.94%, so raw accuracy is not used as the headline metric.

The full unrounded output and run configuration are saved in `results/naive_baseline.json`.

Run the baseline with:

```bash
PYTHONPATH=. python -m src.naive_baseline
```

## Repository Structure

```text
├── data/
│   ├── metadata/
│   └── processed/                 # local parquet files; ignored by Git
├── examples/
│   ├── member_b_label_example.py
│   ├── member_c_dataloader_example.py
│   └── plot_causal_mask.py
├── notebooks/
│   └── update_1.ipynb
├── results/
│   └── naive_baseline.json
├── src/
│   ├── naive_baseline.py
│   ├── streaming_dataloader.py
│   └── switchlingua_labels.py
├── tests/
│   ├── test_streaming_dataloader.py
│   └── test_token_labels.py
├── requirements.txt
└── README.md
```

## Limitations

- The raw dataset does not provide gold word-level LID annotations, so the current pipeline relies on automatic weak labels.
- Same-script pairs such as Spanish–English remain difficult, especially for short words, names, and borrowings.
- Punctuation inherits a neighboring language label and may shift an estimated switch boundary.
- Cantonese–English segmentation and mixed Han/Latin text require further validation.
- Duration is currently measured in XLM-R subword tokens rather than human-defined words.
- The reported baseline uses sampled rows and a highly imbalanced switch distribution.
- The current-token baseline is an initial reference system, not the final multitask transformer.

The current metrics are initial engineering results based on automatic weak LIDs. They should not be presented as final gold-label benchmark results.

## Next Steps

1. **Audit word-level LID quality.** Manually review a stratified sample from every language pair, with special attention to short words, names, borrowings, and punctuation.
2. **Improve same-script LID.** Compare the current weak detector with a stronger contextual LID model or human annotation.
3. **Review Cantonese segmentation.** Validate mixed Han/Latin segmentation and decide whether duration should be reported in words or XLM-R subwords.
4. **Evaluate more data.** Run the baseline over larger or complete splits and report confidence intervals in addition to pair-level F1 and standard deviation.
5. **Address class imbalance.** Compare threshold tuning with class-weighted loss, focal loss, and probability calibration.
6. **Train the multitask causal model.** Add a shared multilingual backbone with switch and duration heads, then compare XLM-R, mBERT, and causal alternatives under the same prefix-only evaluation.
7. **Preserve reproducibility.** Keep the split seed, training-only `lid_vocab`, label conventions, and weak-LID status recorded with every experiment.
