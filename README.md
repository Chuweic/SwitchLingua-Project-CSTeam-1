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

The preprocessing pipeline supports three punctuation policies:

- `keep` (Approach A, default): preserve punctuation and let it inherit a nearby lexical LID.
- `drop` (Approach B): remove punctuation and symbols before tokenization and target generation.
- `neutral` (Approach C): preserve punctuation as model input, assign it the language-independent LID `neutral`, and exclude it from linguistic switch and duration calculations.

Applying the policy before label generation ensures that both switch and duration targets correspond to the same token sequence seen by the model.

For Approach C, a target immediately before a neutral punctuation span is ignored. The final neutral token before the next lexical token anchors the boundary by comparing the nearest non-neutral LIDs on both sides. Thus `en , ar` produces one switch target at the comma rather than artificial `en -> neutral` and `neutral -> ar` switches. Neutral punctuation does not contribute to the upcoming duration length.

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

Keep punctuation with a neutral LID by changing them to:

```bash
  --punctuation-mode neutral \
  --output results/naive_baseline_14pairs_neutral.json
```

| Metric | A: Keep/inherit | B: Drop | C: Keep/neutral | C − A |
|---|---:|---:|---:|---:|
| Anticipatory F1 | 0.4763 | 0.3403 | **0.4883** | +0.0121 |
| Duration accuracy | 0.5003 | 0.4952 | **0.5748** | +0.0745 |
| Switch rate | 0.0752 | 0.0884 | 0.0884 | +0.0133 |
| Universality σ | **0.0894** | 0.0822 | 0.0901 | +0.0007 |

Removing punctuation lowers Anticipatory F1 by 0.1360 while leaving the number of positive switch examples unchanged at 8,937. The test sequence becomes shorter, so the measured switch rate rises. The lower pair-level standard deviation after punctuation removal does not compensate for the much lower overall F1.

Approach C improves the naive Anticipatory F1 by 0.0121 and duration accuracy by 0.0745 relative to A. It retains punctuation as an input cue while removing targets that predict a linguistic switch into punctuation. Because C changes the valid target set and the duration definition, these results are engineering evidence for an A/C model comparison rather than a final claim that C is universally better.

The completed 849-token manual audit gives the same overall boundary F1 for A and C (0.6102) and the same near-punctuation boundary F1 (0.8788). C removes 107 targets that point into punctuation and changes 15 shared switch-duration classes without reducing audited boundary quality. Recreate the audit comparison with:

```bash
PYTHONPATH=. python examples/compare_punctuation_policies.py
```

The detailed outputs are stored in `results/punctuation_ac_label_comparison.json` and `results/punctuation_ac_label_comparison.csv`.

The machine-readable summary is stored in `results/punctuation_ablation_summary.csv`, and the presentation figure is stored in `docs/figures/punctuation_ablation.png`. Recreate the figure with:

```bash
PYTHONPATH=. python examples/plot_punctuation_ablation.py
```

### Training handoff and policy recommendation

Approach C (`neutral`) is the provisional recommendation for the final model because it keeps punctuation as input while excluding punctuation-induced switch targets. Approach A remains available for controlled comparison. Both policies can be selected when constructing records:

```python
records_from_parquet(..., punctuation_mode="keep")     # Approach A
records_from_parquet(..., punctuation_mode="neutral")  # Approach C
```

The real-data integration example exposes the same choice:

```bash
PYTHONPATH=. python examples/test_multitask_integration.py --punctuation-mode keep
PYTHONPATH=. python examples/test_multitask_integration.py --punctuation-mode neutral
```

Since C is at least as good as A in the current cross-policy model comparison and improves the naive F1 and duration accuracy without lowering manual boundary F1, the final comparison should keep all seeds, row caps, splits, and hyperparameters fixed and use C as the primary policy.

## Model Training and Comparison

`src/train.py` fine-tunes the causal dual-head model and evaluates it with the same row caps, seed, validation-selected threshold, and result layout as the naive baseline, so the outputs are directly comparable.

```bash
# Backbone comparison under the current default policy (Approach A)
PYTHONPATH=. python -m src.train --backbone xlmr  --punctuation-mode keep --cross-eval-mode neutral
PYTHONPATH=. python -m src.train --backbone mbert --punctuation-mode keep

# A/C model comparison: same seed, rows, and hyperparameters, only the policy changes
PYTHONPATH=. python -m src.train --backbone xlmr --punctuation-mode neutral --cross-eval-mode keep
```

Each run writes `results/train_<backbone>_<mode>.json` with overall and per-pair test metrics, the universality σ, and the per-step and per-epoch history used for convergence plots. The best epoch and the switch threshold are selected on validation data; the test split is scored once. `--cross-eval-mode` also scores the trained model on test labels built with the other punctuation policy, because A and C use different valid-target sets and their own-label F1 values are not measured on the same positions.

Hyperparameters are set with `--learning-rate`, `--epochs`, `--batch-size`, `--switch-weight`, `--duration-weight`, and `--max-length`; use `--output` to keep tuning runs apart.

Build the comparison table and the convergence figure from all finished runs with:

```bash
PYTHONPATH=. python examples/compare_training_runs.py
```

This writes `results/training_comparison.csv` and `docs/figures/training_convergence.png`. mBERT records are built with the mBERT tokenizer, so its subword-based duration bins differ slightly from XLM-R's.

### Training results

All runs use the 14 pairs, the baseline row caps (500 / 150 / 150 rows per pair), seed 42, batch size 32, and a maximum length of 128. The learning rate 5e-5 was selected from {2e-5, 3e-5, 5e-5} by validation Anticipatory F1 on XLM-R with Approach A (`results/learning_rate_search.csv`).

| Backbone | Approach | Epochs (best) | Anticipatory F1 | Duration accuracy | Universality σ |
|---|---|---:|---:|---:|---:|
| Naive baseline | A: keep | – | 0.4763 | 0.5003 | 0.0894 |
| Naive baseline | C: neutral | – | 0.4883 | 0.5748 | 0.0901 |
| XLM-R | A: keep | 3 (3) | 0.5385 | 0.5691 | 0.0853 |
| mBERT | A: keep | 3 (3) | 0.5813 | 0.5874 | 0.0762 |
| XLM-R | C: neutral | 3 (3) | 0.5715 | 0.6180 | 0.0814 |
| mBERT | C: neutral | 3 (3) | **0.6079** | 0.6111 | 0.0683 |
| XLM-R | C: neutral | 8 (7) | 0.6018 | 0.6102 | 0.0747 |
| mBERT | C: neutral | 8 (5) | 0.6008 | 0.5806 | **0.0677** |

**Punctuation policy.** Models trained with Approach C match or beat models trained with Approach A on switch F1 when both are scored on the same labels (XLM-R: 0.5466 vs 0.5385 on A's labels, 0.5715 vs 0.5531 on C's labels; mBERT: 0.5823 vs 0.5813 and 0.6079 vs 0.5989). Duration accuracy is higher for each model on its own labels because the two policies define segment length differently. Approach C is the declared policy for the remaining experiments (`results/punctuation_ac_model_comparison.csv`).

**Backbone and convergence.** After 3 epochs mBERT leads XLM-R by about 0.04 F1, but the 8-epoch runs show this is a difference in convergence speed: mBERT's validation F1 plateaus at epoch 3, while XLM-R keeps improving until epoch 7 and reaches the same test F1 (0.6018 vs 0.6008). Both backbones level off near 0.60. Validation loss rises after epoch 2–3 for both, driven by the duration head (mBERT's validation duration loss grows from 0.83 to 1.69 by epoch 8), so the best epoch for switch F1 is later than the best epoch for duration. mBERT trains faster (about 3 minutes per epoch on a T4 versus 4) and has the lower pair-level spread.

Each configuration was run with one seed, so differences below roughly 0.01 F1 should not be read as rankings. mBERT and XLM-R are scored on their own subword positions (97,282 and 100,626 test targets under Approach C), and the naive baseline rows use the XLM-R tokenizer. Per-epoch curves are in `results/convergence_curves.csv`, with figures in `docs/figures/training_convergence.png` (3 epochs) and `docs/figures/training_convergence_8ep.png` (8 epochs).
### Person 4 final analysis

Generate the Person 4 model comparison, pair-level universality analysis, CSV summaries, and report/slide figures with:

```bash
PYTHONPATH=. python examples/run_person4_analysis.py
```

The script generates two fair views: the four three-epoch files for A/C policy comparison, and the two eight-epoch neutral files for XLM-R/mBERT backbone comparison. It writes `results/person4_ac_comparison_3ep.csv`, `results/person4_backbone_comparison_8ep.csv`, pair-level tables, and separate figures for each view. The selected inputs are recorded in `results/person4_analysis_metadata.json`. The report-ready text is in `docs/person4_report_section.md`, and the slide-ready content is in `docs/person4_slides.md`.

## LID Validation Audit

The audit samples three validation utterances from each of the 14 language pairs. Manual gold labels were added for 849 pre-token rows and compared with the automatic weak labels.

| Audit metric | Result |
|---|---:|
| Token LID accuracy | **0.9423** (800/849) |
| Switch-boundary F1 | **0.6102** |
| Switch-boundary precision | 0.5047 |
| Switch-boundary recall | 0.7714 |

The token-level accuracy is high, but the lower boundary F1 shows that individual LID errors can create or remove switch boundaries. Same-script pairs and romanized Hindi remain the main failure modes. The completed annotations and detailed per-pair metrics are stored in `results/lid_audit_14pairs.csv` and `results/lid_audit_14pairs_metrics.json`.

### Expanded French–English audit

Because the initial French–English result was based on only three utterances, a targeted follow-up audit uses 20 validation utterances sampled with seed 42. All 364 lexical tokens were assigned manual `fr` or `en` labels; punctuation remains excluded from the audit.

| French–English audit metric | 3-utterance audit | Expanded 20-utterance audit |
|---|---:|---:|
| Audited lexical tokens | 64 | 364 |
| Token LID accuracy | 0.9219 | **0.9176** |
| Switch-boundary F1 | 0.2857 | **0.4898** |
| Switch-boundary precision | 0.1818 | 0.3333 |
| Switch-boundary recall | 0.6667 | **0.9231** |

The larger sample confirms that automatic token LIDs are usually correct, while same-script homographs, borrowings, and names create extra predicted boundaries. French–English is retained in the 14-pair evaluation because token accuracy remains above 0.90 and boundary recall is high; its lower boundary precision is reported as a limitation rather than hidden by removing the pair.

Recreate the blank 20-utterance template and score the completed annotations with:

```bash
PYTHONPATH=. python -m src.lid_audit prepare \
  --data data/processed/validation.parquet \
  --pairs French-English \
  --utterances-per-pair 20 \
  --seed 42 \
  --output results/lid_audit_french_english_20.csv

PYTHONPATH=. python -m src.lid_audit score \
  --audit results/lid_audit_french_english_20.csv \
  --output results/lid_audit_french_english_20_metrics.json
```

The completed annotations and metrics are stored in `results/lid_audit_french_english_20.csv` and `results/lid_audit_french_english_20_metrics.json`.

## Setup and Tests

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest -q
```

Expected result:

```text
71 passed
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
2. Expand the manual LID audit beyond three utterances for the remaining language pairs; French–English now has a completed 20-utterance follow-up audit.
3. Validate Cantonese–English and Chinese–English segmentation and the use of subword-based duration labels.
4. Audit and remove exact utterance overlap across data splits.
5. Regenerate the Person 4 outputs from the eight-epoch neutral runs now stored in `results/`.
6. Repeat the backbone comparison with several seeds, and select the duration head's epoch separately or lower its loss weight to limit duration overfitting.
