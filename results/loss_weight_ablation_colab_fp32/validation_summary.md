# XLM-R loss-weight ablation (validation selection)

Colab Tesla T4, fp32, batch 32, max length 128, 8 epochs, seed 42. The table reports **validation** metrics only. The test set was not used to select weights.

| Switch weight | Duration weight | Best epoch | Val switch F1 | Val duration acc. | Duration guard |
|---:|---:|---:|---:|---:|---|
| 1.0 | 1 | 8 | 0.5967 | 59.97% | pass |
| 1.0 | 0.5 | 6 | 0.6191 | 60.69% | pass |
| 1.0 | 0.3 | 6 | 0.6290 | 60.26% | pass |
| 1.0 | 0.2 | 5 | 0.6299 | 61.73% | pass |
| 1.0 | 0.125 | 5 | 0.6298 | 61.71% | pass |

Duration guard: validation duration accuracy >= 58.97% (1.0 / 1.0 control minus 1 pp).

Selected: **switch=1.0, duration=0.2**.
This is single-seed model selection. Repeat with additional seeds before strong generalization claims.
