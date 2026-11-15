# Results Analysis, Universality Evaluation, and Discussion

The analysis separates the fair three-epoch A/C policy comparison from the eight-epoch C-only backbone comparison. This avoids comparing models trained for different numbers of epochs.

## Experimental comparison

We evaluated the causal dual-head model on 14 language pairs using the same capped train/validation/test construction as the naive baseline. We compared XLM-R and mBERT under two punctuation policies: Approach A (`keep`), which preserves punctuation with an inherited LID, and Approach C (`neutral`), which preserves punctuation as input but excludes it from switch and duration targets. The threshold was selected on the validation split and the test split was evaluated once.

The three-epoch A/C results are:

| Backbone | Policy | Anticipatory F1 | Duration accuracy | Universality σ | F1 gain over same-policy naive |
|---|---|---:|---:|---:|---:|
| XLM-R | A: keep | 0.539 | 0.569 | 0.085 | +0.062 |
| mBERT | A: keep | 0.581 | 0.587 | 0.076 | +0.105 |
| XLM-R | C: neutral | 0.572 | 0.618 | 0.081 | +0.083 |
| mBERT | C: neutral | **0.608** | 0.611 | **0.068** | **+0.120** |

Both pretrained backbones improve over the corresponding current-token naive baseline. The best three-epoch switch result is mBERT with Approach C, reaching an Anticipatory F1 of 0.6079 versus 0.4883 for the C-policy naive baseline.

When the number of epochs is held constant, Approach C is at least as good as Approach A on switch prediction under the cross-policy comparison. It also avoids treating punctuation as a language switch target, so C is used for the final backbone comparison.

### Eight-epoch backbone comparison under Approach C

| Backbone | Best epoch | Anticipatory F1 | Duration accuracy | Universality σ |
|---|---:|---:|---:|---:|
| XLM-R | 7 | **0.6018** | **0.6102** | 0.0747 |
| mBERT | 5 | 0.6008 | 0.5806 | **0.0677** |

After longer training, XLM-R and mBERT are effectively tied on F1. The earlier mBERT advantage was mainly a convergence-speed advantage. mBERT remains more uniform across language pairs, while XLM-R has better duration accuracy in the eight-epoch run.

The mBERT + C three-epoch checkpoint remains the strongest single switch-prediction result, but continuing to eight epochs lowers its duration accuracy from 0.6111 to 0.5806. This indicates duration-head overfitting, so the final model choice should use validation-selected early stopping rather than assuming that more epochs improve every task.

## Universality across language pairs

We measure universality using the standard deviation σ of pair-level Anticipatory F1; lower σ indicates more uniform performance across languages. Under the final eight-epoch C comparison, mBERT has the lower σ (0.0677 versus 0.0747 for XLM-R), although the two backbones have nearly identical overall F1.

The mBERT + C pair-level results range from 0.529 to 0.752. Japanese–English and Korean–English are the strongest pairs, with F1 values of 0.752 and 0.751. Cantonese–English and Malay–English are the weakest at 0.534 and 0.529. Chinese–English reaches 0.693, while the remaining same-script or Latin-script pairs mostly fall in the mid-0.50 to mid-0.60 range.

The pattern is consistent with the preprocessing audit: languages with a distinct writing system are easier for weak LID assignment and token-level prediction, while same-script pairs are harder because Unicode script alone cannot distinguish them. Cantonese–English is an exception to the distinct-script trend because Cantonese shares Han characters with Chinese and requires additional word-segmentation and language-identification decisions.

## Error analysis and limitations

The results should not be interpreted as a complete gold-label benchmark. The word-level LIDs are automatically generated weak labels, and the targeted manual audit achieved high token-level accuracy but substantially lower switch-boundary F1. Short words, names, borrowings, romanized text, and same-script language pairs can therefore create false or missed switch boundaries.

The two punctuation policies define different valid target sets. Consequently, A-policy F1 and C-policy F1 are not perfectly interchangeable unless the model is cross-evaluated on both label definitions. The three-epoch four-run comparison is used for the policy decision, while the eight-epoch C-only comparison is used for backbone convergence. The corresponding outputs are `results/person4_ac_comparison_3ep.csv` and `results/person4_backbone_comparison_8ep.csv`.

Each backbone configuration uses one random seed. Differences near 0.01 F1 should not be treated as definitive rankings. In addition, duration accuracy is affected by tokenizer-specific subword positions.

## Connection to prior work

Feng, Li, and Koehn (2022) studied the limitations of code-switching augmentation for cross-lingual transfer. They showed that replacing tokens without considering contextual similarity can create grammatically incoherent code-switched sentences, and proposed context-aware substitutions evaluated on cross-lingual POS and NER across more than 30 languages. Their results improved over both mBERT and an original code-switching method.

This work is relevant to our universality analysis because it warns that multilingual performance may depend on whether the language mixture is linguistically coherent. Our pair-level results similarly show that a single multilingual model does not perform uniformly across language pairs. Future experiments should therefore separate tokenizer/script effects, weak-LID quality, and genuine cross-lingual generalization rather than treating the overall F1 as a universal score.

**Citation:** Yukun Feng, Feng Li, and Philipp Koehn. 2022. “Toward the Limitation of Code-Switching in Cross-Lingual Transfer.” In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 5966–5971. Abu Dhabi, United Arab Emirates: Association for Computational Linguistics. https://aclanthology.org/2022.emnlp-main.400/

## Final-stage plan

1. Repeat the C-only backbone comparison with multiple random seeds and report variability.
2. Use validation-selected early stopping or duration regularization to reduce duration-head overfitting.
3. Expand the LID audit for same-script pairs, especially Cantonese–English, and audit exact utterance overlap across splits.
4. Report each language pair's F1, duration accuracy, and contribution to universality σ together.
