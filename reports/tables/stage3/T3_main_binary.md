# Table 3 — Stage 3 pipeline comparison (binary A-C | D-G, hold-out, n=2016)

| Route | macro-F1 | 95% CI | κ | acc | bal.acc | MCC | ROC-AUC | M3−M1 mF1 | M3−M1 κ |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 0.4124 | [0.406, 0.419] | 0.0000 | 0.7019 | 0.5000 | 0.0000 | 0.5000 |  |  |
| M1 | 0.4919 | [0.472, 0.513] | 0.0754 | 0.7029 | 0.5289 | 0.1137 | 0.6466 |  |  |
| M3-DINOv2 | 0.4904 | [0.470, 0.514] | 0.0566 | 0.6895 | 0.5223 | 0.0765 | 0.5819 | -0.0015 | -0.0188 |
| M3-ResNet50 | 0.4879 | [0.468, 0.508] | 0.0467 | 0.6835 | 0.5185 | 0.0611 | 0.5699 | -0.0040 | -0.0287 |
| M3-VLMv3 | 0.5276 | [0.506, 0.550] | 0.0558 | 0.5967 | 0.5285 | 0.0558 | 0.5346 | +0.0357 | -0.0196 |

The pool is 29.8% D-G, so M0 (constant A-C) already scores acc 0.7019. Read accuracy only against M0; macro-F1, balanced accuracy and ROC-AUC are the informative columns.
