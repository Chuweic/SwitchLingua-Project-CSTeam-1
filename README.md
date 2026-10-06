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
