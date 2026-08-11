# Table 3-reg — regression-to-kWh vs direct classification (hold-out)

| Route | kWh MAE | kWh R² | macro-F1 (reg→bin) | κ (reg→bin) | macro-F1 (cls) | κ (cls) |
|---|---:|---:|---:|---:|---:|---:|
| M1-reg | 53.7 | 0.121 | 0.1646 | 0.2216 | 0.1720 | 0.2330 |
| M3-DINOv2-reg | 58.2 | 0.028 | 0.1249 | 0.1046 | 0.1500 | 0.0825 |
| M3-ResNet50-reg | 59.1 | 0.010 | 0.1262 | 0.1090 | 0.1490 | 0.0913 |
| M3-VLMv3-reg | 63.2 | -0.100 | 0.1151 | 0.0408 | 0.1370 | 0.0408 |

Boundaries: official NTA8800 residential (A≤160 B≤190 C≤250 D≤290 E≤335 F≤380 G>380 kWh/m²·yr).
Regression objective = L1 (MAE) on the registered primary fossil energy (`PrimaireFossieleEnergieEMGForfaitair` with fallback — audit A04).
