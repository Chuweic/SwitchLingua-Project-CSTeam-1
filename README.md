# SwitchLingua Project - CS Team 1

This repository contains our course project on **predictive multitask learning for streaming code-switching** using the SwitchLingua dataset.

## Project Goals

The project focuses on:

- Predicting whether the next token will switch languages
- Predicting the duration of the upcoming switched-language segment
- Evaluating performance across multiple language pairs
- Comparing multilingual modeling approaches under causal, prefix-only constraints

## Update 1 
### Step 1: Data Preparation

The initial data preparation includes:

- Exploration of the SwitchLingua dataset
- Parsing and normalization of conversation records
- Removal of duplicate conversations
- Conversation-level train / validation / test splitting
- Dataset statistics and language-pair analysis
- The processed train, validation, and test datasets are stored separately
because of their file sizes.

Files:
- `train.parquet`
- `validation.parquet`
- `test.parquet`

Dataset folder: [Processed SwitchLingua Data](https://drive.google.com/drive/folders/1W4y0YEtRhuCIa31Y591JgslPQya9AR5o?usp=sharing)

## Repository Structure

```text
SwitchLingua-Project-CSTeam-1/
├── notebooks/
│   └── update_1.ipynb
├── data/
│   └── processed/
│       ├── cleaning_summary.csv
│       ├── split_summary.csv
│       └── language_pair_distribution.csv
├── README.md
└── .gitignore
```

## Dataset

The project uses the **SwitchLingua** dataset from Hugging Face:

`Shelton1013/SwitchLingua_text`

The dataset contains multilingual code-switching conversations across 15 language pairs.

## Data Splitting

The cleaned dataset is split into:

- 80% training
- 10% validation
- 10% test

Splitting is performed at the original conversation level and stratified by language pair so that utterances from the same conversation do not appear in different data splits.

## Team

**CS Team 1**

## Member B: Tokenization and Predictive Labels

The Member B preprocessing module is available at
`src/switchlingua_labels.py`. It accepts pre-tokenized words and one language
ID per word, then calls a Hugging Face-style fast tokenizer with
`is_split_into_words=True`. Each sub-word inherits the language ID of its
source word through `word_ids()`, so splitting a word does not create a false
language switch.

```python
from src.switchlingua_labels import encode_and_label

result = encode_and_label(tokenizer, words, word_language_ids)
```

For every valid sub-word at position `t`, `switch_labels[t]` predicts whether
the next valid sub-word changes language. `duration_labels[t]` is supervised
only at an anticipated switch: class `0` means 1–2 upcoming sub-word tokens,
class `1` means 3–6, and class `2` means 7 or more. Valid non-switch positions
use `-1`; special tokens, padding, and the final position without a next-token
target use `-100` (`IGNORE_INDEX`).

The raw SwitchLingua export contains utterance text and conversation-level
language-pair metadata, but not token-level language ID annotations. A
language-identification/alignment preprocessing step must therefore provide
`word_language_ids` before this module is called. The complete deterministic
Arabic–English conversion example is in
`examples/member_b_label_example.py`.

Run the Member B tests from the repository root with:

```bash
PYTHONPATH=. pytest -q
```
