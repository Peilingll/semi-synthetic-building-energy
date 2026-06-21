# Table 3-reg — regression-to-kWh vs direct classification (hold-out)

| Route | kWh MAE | kWh R² | macro-F1 (reg→bin) | κ (reg→bin) | macro-F1 (cls) | κ (cls) |
|---|---:|---:|---:|---:|---:|---:|
| M1-reg | 54.8 | 0.140 | 0.1652 | 0.2165 | 0.1720 | 0.2330 |
| M3-DINOv2-reg | 59.2 | 0.047 | 0.1274 | 0.1101 | 0.1500 | 0.0825 |
| M3-ResNet50-reg | 60.2 | 0.030 | 0.1298 | 0.1191 | 0.1490 | 0.0913 |
| M3-VLMv3-reg | 64.7 | -0.106 | 0.1184 | 0.0429 | 0.1370 | 0.0408 |

Boundaries: official NTA8800 residential (A≤160 B≤190 C≤250 D≤290 E≤335 F≤380 G>380 kWh/m²·yr).
regression objective = L1 (MAE) on PrimaireFossieleEnergie.
