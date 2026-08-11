# Table 2a — Stage 2 Cumulative Ablation (binary A-C | D-G, pooled 5-fold OOF, n=8,068)

| Run | Macro-F1 | 95% CI | Quadratic κ | Accuracy | Balanced acc | MCC | ROC-AUC | Δ macro-F1 from S_min |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| M0 | 0.4129 | [0.409, 0.416] | 0.0000 | 0.7034 | 0.5000 | 0.0000 | 0.4971 | — |
| S_min | 0.4596 | [0.450, 0.469] | 0.0369 | 0.7002 | 0.5138 | 0.0674 | 0.6190 | (baseline) |
| S_lookup | 0.4594 | [0.450, 0.470] | 0.0371 | 0.7004 | 0.5138 | 0.0681 | 0.6194 | -0.0002 |
| S_full | 0.4776 | [0.468, 0.488] | 0.0460 | 0.6936 | 0.5177 | 0.0679 | 0.6408 | +0.0180 |

Accuracy must be read against M0: the pool is ~70/30, so a constant prediction already scores acc ≈ 0.70 at macro-F1 ≈ 0.41.
