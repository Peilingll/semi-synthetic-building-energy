# Table 3 — Stage 3 pipeline comparison (binary A-C | D-G, loco_amsterdam, n=8011)

| Route | macro-F1 | 95% CI | κ | acc | bal.acc | MCC | ROC-AUC | M3−M1 mF1 | M3−M1 κ |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| M0 | 0.4100 | [0.407, 0.413] | 0.0000 | 0.6949 | 0.5000 | 0.0000 | 0.5000 |  |  |
| M1 | 0.4963 | [0.485, 0.507] | 0.0014 | 0.6085 | 0.5006 | 0.0014 | 0.5282 |  |  |
| M3-DINOv2 | 0.4904 | [0.479, 0.501] | 0.0005 | 0.6233 | 0.5002 | 0.0005 | 0.5050 | -0.0059 | -0.0009 |
| M3-ResNet50 | 0.4907 | [0.480, 0.501] | 0.0133 | 0.6399 | 0.5057 | 0.0146 | 0.5153 | -0.0056 | +0.0119 |

The pool is 30.5% D-G, so M0 (constant A-C) already scores acc 0.6949. Read accuracy only against M0; macro-F1, balanced accuracy and ROC-AUC are the informative columns.
