# SwitchLingua Project — CS Team 1

This project studies whether code-switching can be predicted in a real-time, prefix-only setting. At token `t`, the system uses only the observed prefix `x₁, ..., xₜ` to predict whether token `t+1` changes language and how long the upcoming language segment will be.

## Goal

The goal is to evaluate whether multilingual models can anticipate code-switching consistently across diverse language pairs.

The project includes two tasks:

1. **Switch prediction:** predict whether `LID(t+1) != LID(t)`.
2. **Duration prediction:** when a switch occurs, predict whether the upcoming segment is short, medium, or long.

Project Update 2 keeps the causal data pipeline and adds a 14-pair preprocessing revision, a punctuation ablation, and a manual audit of the weak Language ID (LID) labels.

## Language Pairs

The current evaluation covers 14 English-pivot language pairs from the processed SwitchLingua data:

| Category | Language pairs | Why they matter |
|---|---|---|
| Cross-script | Arabic–English, Cantonese–English, Chinese–English, Hindi–English, Japanese–English, Korean–English, Russian–English | Tests whether script differences provide reliable switch cues while introducing segmentation and transliteration challenges. |
| Same-script | French–English, German–English, Italian–English, Malay–English, Philippines–English, Spanish–English, Vietnamese–English | Tests harder cases where Unicode script alone cannot distinguish the two languages. |

German–French is excluded from the Update 2 experiments. Keeping a shared English pivot makes the preprocessing rules and cross-pair comparison consistent while the team tests universality across scripts and language families.

## Dataset and Tokenization

The project uses `Shelton1013/SwitchLingua_text` from Hugging Face and the fast `xlm-roberta-base` tokenizer.

Processed parquet files should be placed under:

```text
data/processed/
├── train.parquet
├── validation.parquet
└── test.parquet
```

Because SwitchLingua does not provide gold word-level Language IDs, `src/switchlingua_labels.py` creates automatic weak LID labels using Unicode-script rules and a pair-restricted Lingua detector. Space-delimited languages use whitespace and punctuation boundaries; Chinese and Cantonese Han spans use `jieba` word segmentation before the shared XLM-R fast tokenizer. Unicode combining marks remain attached to their base tokens, and each pre-token's LID is copied to all of its XLM-R subwords using `word_ids()`.

The preprocessing pipeline supports two punctuation policies:

- `keep`: preserve punctuation and symbol tokens before subword tokenization.
- `drop`: remove punctuation and symbol tokens before tokenization and predictive-label generation.

Applying the policy before label generation ensures that both switch and duration targets correspond to the same token sequence seen by the model.

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
PYTHONPATH=. python examples/streaming_dataloader_example.py
```

## Causal Constraint

The DataLoader creates a lower-triangular mask so position `t` can attend only to positions `0...t`. Future tokens and padded keys are blocked.

Before the mask is passed to a Hugging Face backbone, `causal_mask_for_backbone()` converts it from `[batch, 1, length, length]` to `[batch, length, length]`.

The tests verify that:

- changing future tokens does not change earlier outputs;
- the same check fails with a bidirectional mask; and
- a small XLM-R model cannot access future tokens when using the causal mask.

## Punctuation Ablation: Naive Baseline

The baseline estimates:

```text
P(switch at t+1 | current token ID at t)
```

It learns token statistics from the training split, selects its threshold on validation data, and evaluates once on the test split.

Both 14-pair runs use at most 500 training rows, 150 validation rows, and 150 test rows per pair, with up to two utterances from each row. The random seed and all other settings are held constant.

Keep punctuation:

```bash
python -m src.naive_baseline \
  --data-dir data/processed \
  --train-rows-per-pair 500 \
  --validation-rows-per-pair 150 \
  --test-rows-per-pair 150 \
  --utterances-per-row 2 \
  --punctuation-mode keep \
  --output results/naive_baseline_14pairs_keep.json
```

Drop punctuation by changing the last two options to:

```bash
  --punctuation-mode drop \
  --output results/naive_baseline_14pairs_drop.json
```

| Metric | Keep | Drop | Drop − Keep |
|---|---:|---:|---:|
| Anticipatory F1 | **0.4763** | 0.3403 | -0.1360 |
| Duration accuracy | **0.5003** | 0.4952 | -0.0050 |
| Switch rate | 0.0752 | 0.0884 | +0.0133 |
| Universality σ | 0.0894 | **0.0822** | -0.0072 |

Removing punctuation lowers Anticipatory F1 by 0.1360 while leaving the number of positive switch examples unchanged at 8,937. The test sequence becomes shorter, so the measured switch rate rises. The lower pair-level standard deviation after punctuation removal does not compensate for the much lower overall F1.

The machine-readable summary is stored in `results/punctuation_ablation_summary.csv`, and the presentation figure is stored in `docs/figures/punctuation_ablation.png`. Recreate the figure with:

```bash
PYTHONPATH=. python examples/plot_punctuation_ablation.py
```

## LID Validation Audit

The audit samples three validation utterances from each of the 14 language pairs. Manual gold labels were added for 849 pre-token rows and compared with the automatic weak labels.

| Audit metric | Result |
|---|---:|
| Token LID accuracy | **0.9423** (800/849) |
| Switch-boundary F1 | **0.6102** |
| Switch-boundary precision | 0.5047 |
| Switch-boundary recall | 0.7714 |

The token-level accuracy is high, but the lower boundary F1 shows that individual LID errors can create or remove switch boundaries. Same-script pairs and romanized Hindi remain the main failure modes. The completed annotations and detailed per-pair metrics are stored in `results/lid_audit_14pairs.csv` and `results/lid_audit_14pairs_metrics.json`.

## Setup and Tests

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest -q
```

Expected result:

```text
42 passed
```

## Limitations

- The current LIDs are automatic weak labels rather than gold annotations.
- Same-script pairs remain difficult to classify, especially for short words, names, and borrowings.
- Chinese and Cantonese segmentation is automatic word segmentation rather than gold word-boundary annotation, and segmentation choices can change subword-based duration counts.
- Punctuation materially affects the naive current-token baseline, so later model comparisons must use one declared policy.
- Cantonese–English and Chinese–English segmentation require further validation.
- Duration is currently measured in XLM-R subword tokens.
- Exact utterance overlap across splits should be audited before full-model training.
- The baseline uses balanced per-pair row caps rather than every available row.
- The 849-token LID audit is a targeted quality check, not a complete gold-label benchmark.

The current metrics are initial engineering results and should not be presented as final gold-label benchmark results.

## Next Steps

1. Improve same-script LID handling for short words, names, borrowings, and romanized text.
2. Expand the manual LID audit beyond three utterances per pair.
3. Validate Cantonese–English and Chinese–English segmentation and the use of subword-based duration labels.
4. Audit and remove exact utterance overlap across data splits.
5. Use one fixed punctuation policy for the mBERT and XLM-R comparison.
6. Train the causal dual-head model and compare pair-level Anticipatory F1 and duration convergence.
