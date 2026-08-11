# Table 3-reg — regression-to-kWh vs direct classification (loco_amsterdam)

| Route | kWh MAE | kWh R² | macro-F1 (reg→bin) | κ (reg→bin) | macro-F1 (cls) | κ (cls) |
|---|---:|---:|---:|---:|---:|---:|
| M1-reg | 63.5 | -0.158 | 0.4573 | -0.0142 | 0.4963 | 0.0014 |
| M3-DINOv2-reg | 62.7 | -0.104 | 0.4780 | 0.0027 | 0.4904 | 0.0005 |
| M3-ResNet50-reg | 64.0 | -0.137 | 0.4893 | 0.0228 | 0.4907 | 0.0133 |

Boundaries: one threshold at the C boundary, 250 kWh/m²·yr.
Regression objective = L1 (MAE) on the registered primary fossil energy (`PrimaireFossieleEnergieEMGForfaitair` with fallback — audit A04).
