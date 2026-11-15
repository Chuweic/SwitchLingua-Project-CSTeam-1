# Person 4 Slides: Results, Universality, and Next Steps

> The slides separate the fair three-epoch A/C comparison from the eight-epoch C-only backbone comparison. Regenerate all outputs with `PYTHONPATH=. python examples/run_person4_analysis.py` after any new runs.

## Slide 1 — Model comparison

**Title:** Model Comparison: mBERT and XLM-R

- Same causal dual-head architecture, same 14 language pairs, same evaluation protocol.
- Approach C (`neutral`) keeps punctuation as input but removes punctuation-induced switch targets.
- Best three-epoch switch result: **mBERT + C, Anticipatory F1 = 0.6079**.
- Naive C baseline: **0.488 F1**; model improvement: **+0.120**.
- mBERT outperforms XLM-R under both punctuation policies.

**Visual:** `docs/figures/person4_model_comparison.png`

**Speaker note:** We use the four three-epoch runs for the A/C decision, so the policy comparison does not mix different training lengths.

## Slide 2 — Backbone comparison after convergence

**Title:** Longer Training Closes the Backbone Gap

- Both eight-epoch runs use Approach C.
- XLM-R: **F1 = 0.6018**, duration accuracy **0.6102**.
- mBERT: **F1 = 0.6008**, duration accuracy **0.5806**.
- The F1 difference is only **0.0011**: mBERT's earlier lead was mainly faster convergence.
- mBERT has lower universality σ (**0.0677 vs. 0.0747**), but its duration head shows overfitting.

**Visual:** `docs/figures/person4_backbone_8ep.png` and `docs/figures/person4_universality_8ep.png`

**Speaker note:** More epochs do not automatically improve both tasks. The switch head and duration head peak at different times, so validation-based early stopping matters.

## Slide 3 — Universality across 14 language pairs

**Title:** Universality Is Strong but Not Uniform

- Universality is summarized by pair-level F1 standard deviation σ; lower is more consistent.
- mBERT + C has the lowest provisional σ: **0.068**.
- Strongest pairs: **Japanese–English 0.752** and **Korean–English 0.751**.
- Weakest pairs: **Malay–English 0.529** and **Cantonese–English 0.534**.
- Distinct-script pairs are generally easier; same-script pairs require stronger LID and contextual cues.

**Visual:** `docs/figures/person4_pair_f1.png`

**Speaker note:** The language-pair spread shows why the overall average alone is not enough to support a universality claim.

## Slide 4 — Conclusions and next steps

**Title:** Conclusions and Final-Stage Plan

- Use Approach C as the declared punctuation policy.
- Use validation-selected early stopping rather than assuming eight epochs is optimal.
- Repeat the C-only XLM-R/mBERT comparison with multiple random seeds.
- Investigate duration-head overfitting and expand the same-script LID audit.
- Report pair-level F1, duration accuracy, and universality σ together.

**Speaker note:** The final analysis separates the A/C policy decision from the converged backbone comparison.
