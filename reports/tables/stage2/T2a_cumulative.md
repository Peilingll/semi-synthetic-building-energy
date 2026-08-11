# Table 2a — Stage 2 Cumulative Ablation (pooled 5-fold OOF, n=8,068)

| Run | Macro-F1 | 95% CI | Quadratic κ | Accuracy | Δ macro-F1 from S_min |
|---|---:|---|---:|---:|---:|
| M0 | 0.0681 | [0.066, 0.070] | 0.0000 | 0.3127 | — |
| S_min | 0.1635 | [0.157, 0.170] | 0.1838 | 0.3519 | (baseline) |
| S_lookup | 0.1637 | [0.157, 0.171] | 0.1778 | 0.3513 | +0.0002 |
| S_full | 0.1808 | [0.173, 0.189] | 0.1915 | 0.3487 | +0.0173 |
