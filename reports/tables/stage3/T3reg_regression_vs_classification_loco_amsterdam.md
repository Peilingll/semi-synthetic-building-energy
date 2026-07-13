# Table 3-reg — regression-to-kWh vs direct classification (loco_amsterdam)

| Route | kWh MAE | kWh R² | macro-F1 (reg→bin) | κ (reg→bin) | macro-F1 (cls) | κ (cls) |
|---|---:|---:|---:|---:|---:|---:|
| M1-reg | 65.3 | -0.233 | 0.1322 | 0.0523 | 0.1429 | 0.0263 |
| M3-DINOv2-reg | 62.5 | -0.125 | 0.1232 | 0.0338 | 0.1412 | 0.0295 |
| M3-ResNet50-reg | 63.9 | -0.159 | 0.1195 | 0.0316 | 0.1462 | 0.0363 |

Boundaries: official NTA8800 residential (A≤160 B≤190 C≤250 D≤290 E≤335 F≤380 G>380 kWh/m²·yr).
regression objective = L1 (MAE) on PrimaireFossieleEnergie.
