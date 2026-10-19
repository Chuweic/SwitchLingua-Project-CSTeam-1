# SwitchLingua Project — CS Team 1

This project studies whether code-switching can be predicted in a real-time, prefix-only setting. At token `t`, the system uses only the observed prefix `x₁, ..., xₜ` to predict whether token `t+1` changes language and how long the upcoming language segment will be.

## Goal

The goal is to evaluate whether multilingual models can anticipate code-switching consistently across diverse language pairs.

The project includes two tasks:

1. **Switch prediction:** predict whether `LID(t+1) != LID(t)`.
2. **Duration prediction:** when a switch occurs, predict whether the upcoming segment is short, medium, or long.

Project Update 1 establishes the data pipeline, predictive labels, causal constraints, and a current-token-only baseline.

## Language Pairs

The current evaluation covers all 15 language pairs available in the processed SwitchLingua data:

| Category | Language pairs | Why they matter |
|---|---|---|
| Cross-script | Arabic–English, Cantonese–English, Chinese–English, Hindi–English, Japanese–English, Korean–English, Russian–English | Tests whether script differences provide reliable switch cues while introducing segmentation and transliteration challenges. |
| Same-script | French–English, German–English, Italian–English, Malay–English, Philippines–English, Spanish–English, Vietnamese–English | Tests harder cases where Unicode script alone cannot distinguish the two languages. |
| Non-English pivot | German–French | Checks whether the pipeline generalizes beyond English-centered code-switching. |

Using all 15 pairs provides a broader initial universality audit than the earlier four-pair subset.

## Dataset and Tokenization

The project uses `Shelton1013/SwitchLingua_text` from Hugging Face and the fast `xlm-roberta-base` tokenizer.

Processed parquet files should be placed under:

```text
data/processed/
├── train.parquet
├── validation.parquet
└── test.parquet
```

Because SwitchLingua does not provide gold word-level Language IDs, `src/switchlingua_labels.py` creates automatic weak LID labels using Unicode-script rules and a pair-restricted Lingua detector. Each word’s LID is copied to all of its XLM-R subwords using `word_ids()`.

## Predictive Labels

For each valid token position `t`:

```text
switch_label[t] = 1 if LID(t+1) != LID(t), otherwise 0
```

Duration classes describe the language segment beginning at `t+1`:

| Class | Upcoming segment length |
|---:|---:|
| 0 | 1–2 subword tokens |
| 1 | 3–6 subword tokens |
| 2 | 7 or more subword tokens |

`duration_label = -1` means no switch occurs. The value `-100` is used for padding, special tokens, and positions without a valid next-token target.

## Streaming DataLoader

`src/streaming_dataloader.py` produces batches containing:

- `input_ids`
- `attention_mask`
- `current_lid`
- `switch_labels`
- `duration_labels`
- `language_pair`
- `causal_mask`

The project simulates streaming prediction with explicit prefix windows and full-sequence batches protected by a causal mask. The tokenizer’s padding ID must be supplied when creating a loader:

```python
loader = create_streaming_dataloader(
    records,
    batch_size=2,
    pad_token_id=tokenizer.pad_token_id,
)
```

Run the demonstration with:

```bash
PYTHONPATH=. python examples/member_c_dataloader_example.py
```

## Causal Constraint

The DataLoader creates a lower-triangular mask so position `t` can attend only to positions `0...t`. Future tokens and padded keys are blocked.

Before the mask is passed to a Hugging Face backbone, `causal_mask_for_backbone()` converts it from `[batch, 1, length, length]` to `[batch, length, length]`.

The tests verify that:

- changing future tokens does not change earlier outputs;
- the same check fails with a bidirectional mask; and
- a small XLM-R model cannot access future tokens when using the causal mask.

## Naive Baseline

The baseline estimates:

```text
P(switch at t+1 | current token ID at t)
```

It learns token statistics from the training split, selects its threshold on validation data, and evaluates once on the test split.

The formal 15-pair run uses at most 500 training rows, 150 validation rows, and 150 test rows per pair, with up to two utterances from each row:

```bash
python -m src.naive_baseline \
  --data-dir data/processed \
  --train-rows-per-pair 500 \
  --validation-rows-per-pair 150 \
  --test-rows-per-pair 150 \
  --utterances-per-row 2 \
  --output results/naive_baseline_15pairs.json
```

| Test set | Anticipatory F1 | Precision | Recall | Duration accuracy |
|---|---:|---:|---:|---:|
| Overall | **0.4643** | 0.4746 | 0.4545 | 0.5059 |
| Arabic–English | 0.5509 | 0.5298 | 0.5736 | 0.4703 |
| Cantonese–English | 0.3956 | 0.3532 | 0.4495 | 0.4794 |
| Chinese–English | 0.5752 | 0.4763 | 0.7261 | 0.5607 |
| French–English | 0.3909 | 0.5064 | 0.3183 | 0.4948 |
| German–English | 0.4724 | 0.5400 | 0.4199 | 0.5046 |
| German–French | 0.4467 | 0.4809 | 0.4170 | 0.5193 |
| Hindi–English | 0.2784 | 0.2080 | 0.4208 | 0.5830 |
| Italian–English | 0.4583 | 0.5633 | 0.3863 | 0.5102 |
| Japanese–English | 0.5966 | 0.4934 | 0.7544 | 0.6658 |
| Korean–English | 0.6363 | 0.6134 | 0.6608 | 0.5636 |
| Malay–English | 0.4101 | 0.4850 | 0.3553 | 0.4408 |
| Philippines–English | 0.4046 | 0.4237 | 0.3873 | 0.4778 |
| Russian–English | 0.5142 | 0.4477 | 0.6037 | 0.4548 |
| Spanish–English | 0.4736 | 0.5274 | 0.4298 | 0.5102 |
| Vietnamese–English | 0.4991 | 0.5043 | 0.4939 | 0.5061 |

The validation-selected threshold is **0.30**. The overall test switch rate is **8.81%**, so raw accuracy is not used as the main metric. The standard deviation of pair-level Anticipatory F1 is **0.0898**, which is the initial universality measure.

Full results are stored in `results/naive_baseline_15pairs.json`.

## Setup and Tests

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest -q
```

Expected result:

```text
23 passed
```

## Limitations

- The current LIDs are automatic weak labels rather than gold annotations.
- Same-script pairs remain difficult to classify, especially for short words, names, and borrowings.
- Punctuation may shift an estimated switch boundary.
- Cantonese–English and Chinese–English segmentation require further validation.
- Duration is currently measured in XLM-R subword tokens.
- Exact utterance overlap across splits should be audited before full-model training.
- The formal baseline uses balanced per-pair row caps rather than every available row.

The current metrics are initial engineering results and should not be presented as final gold-label benchmark results.

## Next Steps

1. Manually audit word-level LID quality across all 15 language pairs.
2. Improve same-script LID handling for short words, names, and borrowings.
3. Validate Cantonese–English and Chinese–English segmentation and the use of subword-based duration labels.
4. Audit and remove exact utterance overlap across data splits.
5. Evaluate the baseline on larger or complete data splits.
6. Train a causal multilingual transformer with switch and duration prediction heads.
7. Compare pair-level Anticipatory F1 and performance variation across language pairs.
