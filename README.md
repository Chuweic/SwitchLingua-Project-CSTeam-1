# SwitchLingua Project — CS Team 1

This project studies whether code-switching can be predicted in a real-time, prefix-only setting. At token `t`, the system uses only the observed prefix `x₁, ..., xₜ` to predict whether token `t+1` changes language and how long the upcoming language segment will be.

## Goal

The goal is to evaluate whether multilingual models can anticipate code-switching consistently across diverse language pairs.

The project includes two tasks:

1. **Switch prediction:** predict whether `LID(t+1) != LID(t)`.
2. **Duration prediction:** when a switch occurs, predict whether the upcoming segment is short, medium, or long.

Project Update 1 establishes the data pipeline, predictive labels, causal constraints, and a current-token-only baseline.

## Language Pairs

We selected four language pairs to test universality:

| Language pair | Rationale |
|---|---|
| Arabic–English | Different scripts, writing directions, and language families. |
| Hindi–English | Devanagari/Latin contrast and Indo-Aryan/Germanic contrast. |
| Cantonese–English | Han/Latin mixing and difficult segmentation. |
| Spanish–English | Same-script pair that cannot be separated using Unicode script alone. |

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

| Test set | Anticipatory F1 | Precision | Recall | Duration accuracy |
|---|---:|---:|---:|---:|
| Overall | **0.4419** | 0.4157 | 0.4717 | 0.4953 |
| Arabic–English | 0.5513 | 0.5284 | 0.5762 | 0.5116 |
| Cantonese–English | 0.4017 | 0.3516 | 0.4685 | 0.5038 |
| Hindi–English | 0.3023 | 0.2253 | 0.4594 | 0.5018 |
| Spanish–English | 0.4885 | 0.5667 | 0.4293 | 0.4816 |

The overall switch rate is **6.94%**, so raw accuracy is not used as the main metric. The standard deviation of pair-level Anticipatory F1 is **0.0937**, which is the initial universality measure.

Full results are stored in `results/naive_baseline.json`.

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
- Spanish–English short words, names, and borrowings remain difficult to classify.
- Punctuation may shift an estimated switch boundary.
- Cantonese–English segmentation requires further validation.
- Duration is currently measured in XLM-R subword tokens.
- Exact utterance overlap across splits should be audited before full-model training.

The current metrics are initial engineering results and should not be presented as final gold-label benchmark results.

## Next Steps

1. Manually audit word-level LID quality for each selected language pair.
2. Improve same-script LID handling for Spanish–English short words, names, and borrowings.
3. Validate Cantonese–English segmentation and the use of subword-based duration labels.
4. Audit and remove exact utterance overlap across data splits.
5. Evaluate the baseline on larger or complete data splits.
6. Train a causal multilingual transformer with switch and duration prediction heads.
7. Compare pair-level Anticipatory F1 and performance variation across language pairs.
