# Person 4 Slides: Results, Universality, and Next Steps

> These slide contents use the current three-epoch snapshot and are provisional until the eight-epoch neutral runs are available. Replace the numbers and figures by rerunning `PYTHONPATH=. python examples/run_person4_analysis.py`.

## Slide 1 — Model comparison

**Title:** Model Comparison: mBERT and XLM-R

- Same causal dual-head architecture, same 14 language pairs, same evaluation protocol.
- Approach C (`neutral`) keeps punctuation as input but removes punctuation-induced switch targets.
- Provisional best result: **mBERT + C, Anticipatory F1 = 0.608**.
- Naive C baseline: **0.488 F1**; model improvement: **+0.120**.
- mBERT outperforms XLM-R under both punctuation policies.

**Visual:** `docs/figures/person4_model_comparison.png`

**Speaker note:** These are three-epoch results, so we treat them as an interim comparison rather than the final benchmark.

## Slide 2 — Universality across 14 language pairs

**Title:** Universality Is Strong but Not Uniform

- Universality is summarized by pair-level F1 standard deviation σ; lower is more consistent.
- mBERT + C has the lowest provisional σ: **0.068**.
- Strongest pairs: **Japanese–English 0.752** and **Korean–English 0.751**.
- Weakest pairs: **Malay–English 0.529** and **Cantonese–English 0.534**.
- Distinct-script pairs are generally easier; same-script pairs require stronger LID and contextual cues.

**Visual:** `docs/figures/person4_pair_f1.png` and `docs/figures/person4_universality.png`

**Speaker note:** The language-pair spread shows why the overall average alone is not enough to support a universality claim.

## Slide 3 — Conclusions and next steps

**Title:** Conclusions and Final-Stage Plan

- Recommend Approach C provisionally: it is at least as good as A and avoids artificial punctuation switches.
- mBERT is the strongest current backbone, but the final claim depends on the eight-epoch runs.
- Weak LID quality and same-script ambiguity remain important error sources.
- Next: replace the provisional inputs with the eight-epoch files, regenerate all tables/figures, and audit split overlap.
- Next: expand manual LID validation, especially for Cantonese–English and other same-script pairs.

**Speaker note:** We will not present the three-epoch numbers as final; the final report will use the regenerated outputs from the same analysis command.
