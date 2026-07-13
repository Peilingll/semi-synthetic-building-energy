# Table — Ordinal metrics + literature-aligned collapses (loco_amsterdam, n=8011)

Primary task stays 7-class A–G; collapses re-bin the SAME predictions to each
reference paper's label granularity. External anchors: Mayer et al. 2023 binary
macro-F1 0.646 (UK, multi-modal); Sun et al. 2026 binary acc 0.64/0.69 (Glasgow/Edinburgh).

| route | 7c mF1 | 7c acc | ±1 acc | grade MAE | Mayer mF1 | Mayer acc | Sun mF1 | Sun acc | 3c mF1 | 3c acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 0.0503 | 0.2138 | 0.3771 | 2.006 | 0.4556 | 0.8367 | 0.4100 | 0.6949 | 0.1826 | 0.3771 |
| M1 | 0.1429 | 0.2383 | 0.4917 | 1.680 | 0.4958 | 0.7778 | 0.4986 | 0.6135 | 0.3365 | 0.4044 |
| M3-DINOv2 | 0.1412 | 0.2351 | 0.5254 | 1.569 | 0.5007 | 0.7390 | 0.4982 | 0.5867 | 0.3426 | 0.4049 |
| M3-ResNet50 | 0.1462 | 0.2385 | 0.5127 | 1.601 | 0.5092 | 0.7623 | 0.5061 | 0.6060 | 0.3492 | 0.4106 |
