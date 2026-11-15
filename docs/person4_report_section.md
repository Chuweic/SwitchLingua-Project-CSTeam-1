# Results Analysis, Universality Evaluation, and Discussion

> **Result status.** The numerical snapshot below is generated from the available three-epoch runs and is **provisional**. The final report should rerun `examples/run_person4_analysis.py` after the eight-epoch files are added, then replace the table and figures with the regenerated outputs.

## Experimental comparison

We evaluated the causal dual-head model on 14 language pairs using the same capped train/validation/test construction as the naive baseline. We compared XLM-R and mBERT under two punctuation policies: Approach A (`keep`), which preserves punctuation with an inherited LID, and Approach C (`neutral`), which preserves punctuation as input but excludes it from switch and duration targets. The threshold was selected on the validation split and the test split was evaluated once.

The provisional three-epoch results are:

| Backbone | Policy | Anticipatory F1 | Duration accuracy | Universality σ | F1 gain over same-policy naive |
|---|---|---:|---:|---:|---:|
| XLM-R | A: keep | 0.539 | 0.569 | 0.085 | +0.062 |
| mBERT | A: keep | 0.581 | 0.587 | 0.076 | +0.105 |
| XLM-R | C: neutral | 0.572 | 0.618 | 0.081 | +0.083 |
| mBERT | C: neutral | **0.608** | 0.611 | **0.068** | **+0.120** |

Both pretrained backbones improve over the corresponding current-token naive baseline. mBERT is stronger than XLM-R under both policies. The best provisional configuration is mBERT with Approach C, reaching an Anticipatory F1 of 0.608 versus 0.488 for the C-policy naive baseline. Approach C also improves the duration accuracy of XLM-R relative to Approach A and gives the strongest overall F1 for both backbones in this snapshot.

The comparison should be interpreted cautiously because the three-epoch runs have not fully converged. In particular, Approach C is selected as the recommended policy because it is at least as good as Approach A on the cross-policy comparison and avoids treating punctuation as a language switch target; the final policy claim must be checked again with the eight-epoch runs.

## Universality across language pairs

We measure universality using the standard deviation σ of pair-level Anticipatory F1; lower σ indicates more uniform performance across languages. Under Approach C, the provisional σ is 0.081 for XLM-R and 0.068 for mBERT. This suggests that mBERT is not only stronger on average but also more consistent across the 14 pairs.

The mBERT + C pair-level results range from 0.529 to 0.752. Japanese–English and Korean–English are the strongest pairs, with F1 values of 0.752 and 0.751. Cantonese–English and Malay–English are the weakest at 0.534 and 0.529. Chinese–English reaches 0.693, while the remaining same-script or Latin-script pairs mostly fall in the mid-0.50 to mid-0.60 range.

The pattern is consistent with the preprocessing audit: languages with a distinct writing system are easier for weak LID assignment and token-level prediction, while same-script pairs are harder because Unicode script alone cannot distinguish them. Cantonese–English is an exception to the distinct-script trend because Cantonese shares Han characters with Chinese and requires additional word-segmentation and language-identification decisions.

## Error analysis and limitations

The results should not be interpreted as a complete gold-label benchmark. The word-level LIDs are automatically generated weak labels, and the targeted manual audit achieved high token-level accuracy but substantially lower switch-boundary F1. Short words, names, borrowings, romanized text, and same-script language pairs can therefore create false or missed switch boundaries.

The two punctuation policies also define different valid target sets. Consequently, an A-policy F1 and a C-policy F1 are not perfectly interchangeable unless the model is cross-evaluated on both label definitions. We report both own-policy and cross-policy comparisons in `results/punctuation_ac_model_comparison.csv` and use C as the recommended engineering policy because it keeps punctuation as useful input while preventing artificial punctuation-induced switches.

## Connection to prior work

Feng, Li, and Koehn (2022) studied the limitations of code-switching augmentation for cross-lingual transfer. They showed that replacing tokens without considering contextual similarity can create grammatically incoherent code-switched sentences, and proposed context-aware substitutions evaluated on cross-lingual POS and NER across more than 30 languages. Their results improved over both mBERT and an original code-switching method.

This work is relevant to our universality analysis because it warns that multilingual performance may depend on whether the language mixture is linguistically coherent. Our pair-level results similarly show that a single multilingual model does not perform uniformly across language pairs. Future experiments should therefore separate tokenizer/script effects, weak-LID quality, and genuine cross-lingual generalization rather than treating the overall F1 as a universal score.

**Citation:** Yukun Feng, Feng Li, and Philipp Koehn. 2022. “Toward the Limitation of Code-Switching in Cross-Lingual Transfer.” In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pages 5966–5971. Abu Dhabi, United Arab Emirates: Association for Computational Linguistics. https://aclanthology.org/2022.emnlp-main.400/

## Final-stage plan

1. Replace the provisional three-epoch inputs with `converge_xlmr_neutral_8ep.json` and `converge_mbert_neutral_8ep.json` when they are pushed.
2. Regenerate `results/person4_model_summary.csv`, `results/person4_pair_metrics.csv`, the three Person 4 figures, and this section's table.
3. Confirm that the final selected epoch and threshold are determined only from validation data.
4. Report the final mBERT/XLM-R comparison and the A/C policy decision using the same test split and the same language-pair list.
5. Expand the LID audit for same-script pairs, especially Cantonese–English, and audit exact utterance overlap across splits.
