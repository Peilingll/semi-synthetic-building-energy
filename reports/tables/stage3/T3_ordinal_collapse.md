# Table — Ordinal metrics + literature-aligned collapses (hold-out, n=2016)

Primary task stays 7-class A–G; collapses re-bin the SAME predictions to each
reference paper's label granularity. External anchors: Mayer et al. 2023 binary
macro-F1 0.646 (UK, multi-modal); Sun et al. 2026 binary acc 0.64/0.69 (Glasgow/Edinburgh).

| route | 7c mF1 | 7c acc | ±1 acc | grade MAE | Mayer mF1 | Mayer acc | Sun mF1 | Sun acc | 3c mF1 | 3c acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | 0.0676 | 0.3100 | 0.6116 | 1.198 | 0.4569 | 0.8413 | 0.4124 | 0.7019 | 0.2067 | 0.4494 |
| M1 | 0.1720 | 0.3557 | 0.6548 | 1.181 | 0.5019 | 0.8323 | 0.5139 | 0.6969 | 0.4078 | 0.5357 |
| M3-DINOv2 | 0.1500 | 0.2917 | 0.5888 | 1.373 | 0.4876 | 0.8140 | 0.4975 | 0.6627 | 0.3477 | 0.4633 |
| M3-ResNet50 | 0.1490 | 0.2817 | 0.5868 | 1.381 | 0.5024 | 0.8209 | 0.4918 | 0.6523 | 0.3525 | 0.4618 |
| M3-VLMv3 | 0.1370 | 0.2783 | 0.5908 | 1.369 | 0.4817 | 0.8160 | 0.5027 | 0.6438 | 0.3250 | 0.4563 |
| M2-DINOv2 | 0.2131 | 0.2664 | 0.5437 | 1.625 | 0.5620 | 0.6989 | 0.5827 | 0.6076 | 0.4430 | 0.4638 |
| M2-ResNet50 | 0.2014 | 0.2634 | 0.5704 | 1.537 | 0.5408 | 0.7341 | 0.5755 | 0.6166 | 0.4211 | 0.4628 |
| M2-VLM | 0.0154 | 0.0456 | 0.1672 | 3.091 | 0.1501 | 0.1687 | 0.2426 | 0.3051 | 0.1042 | 0.1657 |
