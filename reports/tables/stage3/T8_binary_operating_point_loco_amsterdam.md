# T8 — Binary task at two operating points (loco_amsterdam, n=8,011)

Positive class = `D-G`. Dev base rate P(D-G) = **0.265**; the rate-matched point labels that same fraction of the hold-out (highest scores first). ROC-AUC is threshold-free and shared by both columns.

| route | mF1 @0.5 | bal.acc @0.5 | D-G recall @0.5 | mF1 @rate | bal.acc @rate | D-G recall @rate | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 0.4100 | 0.5000 | 0.0000 | 0.4100 | 0.5000 | 0.0000 | 0.5000 | 0.3051 |
| M1 | 0.4963 | 0.5006 | 0.2238 | 0.4902 | 0.4916 | 0.2533 | 0.5282 | 0.3090 |
| M3-DINOv2 | 0.4904 | 0.5002 | 0.1845 | 0.4993 | 0.5012 | 0.2504 | 0.5050 | 0.3060 |
| M3-ResNet50 | 0.4907 | 0.5057 | 0.1616 | 0.5025 | 0.5033 | 0.2696 | 0.5153 | 0.3148 |

Rate-matched thresholds on P(D-G): M0 0.305, M1 0.445, M3-DINOv2 0.440, M3-ResNet50 0.427.

