# Table 3-reg — regression-to-kWh vs direct classification (hold-out)

| Route | kWh MAE | kWh R² | macro-F1 (reg→bin) | κ (reg→bin) | macro-F1 (cls) | κ (cls) |
|---|---:|---:|---:|---:|---:|---:|
| M1-reg | 53.7 | 0.121 | 0.4793 | 0.0654 | 0.4919 | 0.0754 |
| M3-DINOv2-reg | 58.2 | 0.028 | 0.4650 | 0.0356 | 0.4904 | 0.0566 |
| M3-ResNet50-reg | 59.1 | 0.010 | 0.4636 | 0.0370 | 0.4879 | 0.0467 |
| M3-VLMv3-reg | 63.2 | -0.100 | 0.4442 | 0.0103 | 0.5276 | 0.0558 |

Boundaries: one threshold at the C boundary, 250 kWh/m²·yr.
Regression objective = L1 (MAE) on the registered primary fossil energy (`PrimaireFossieleEnergieEMGForfaitair` with fallback — audit A04).
