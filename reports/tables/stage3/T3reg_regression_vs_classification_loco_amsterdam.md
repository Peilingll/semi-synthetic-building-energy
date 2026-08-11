# Table 3-reg — regression-to-kWh vs direct classification (loco_amsterdam)

| Route | kWh MAE | kWh R² | macro-F1 (reg→bin) | κ (reg→bin) | macro-F1 (cls) | κ (cls) |
|---|---:|---:|---:|---:|---:|---:|
| M1-reg | 63.5 | -0.158 | 0.1390 | 0.0543 | 0.1429 | 0.0263 |
| M3-DINOv2-reg | 62.7 | -0.104 | 0.1198 | 0.0256 | 0.1412 | 0.0295 |
| M3-ResNet50-reg | 64.0 | -0.137 | 0.1186 | 0.0263 | 0.1462 | 0.0363 |

Boundaries: official NTA8800 residential (A≤160 B≤190 C≤250 D≤290 E≤335 F≤380 G>380 kWh/m²·yr).
Regression objective = L1 (MAE) on the registered primary fossil energy (`PrimaireFossieleEnergieEMGForfaitair` with fallback — audit A04).
