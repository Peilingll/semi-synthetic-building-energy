# Table 3 — Stage 3 pipeline comparison incl. M2 (hold-out, n=2,016)

| Route | macro-F1 | 95% CI | κ | acc | vs M1 (mF1) | vs M1 (κ) |
|---|---:|---|---:|---:|---:|---:|
| M0 | 0.0676 | [0.064, 0.071] | 0.0000 | 0.3100 |  |  |
| M1 | 0.1720 | [0.160, 0.185] | 0.2330 | 0.3557 |  |  |
| M3-DINOv2 | 0.1500 | [0.135, 0.164] | 0.0825 | 0.2917 | -0.0220 | -0.1505 |
| M3-ResNet50 | 0.1490 | [0.134, 0.165] | 0.0913 | 0.2817 | -0.0230 | -0.1417 |
| M3-VLMv3 | 0.1370 | [0.125, 0.150] | 0.0408 | 0.2783 | -0.0350 | -0.1922 |
| M2-DINOv2 (aligned) | 0.2131 | [0.194, 0.230] | 0.2355 | 0.2664 | +0.0411 | +0.0025 |
| M2-DINOv2 (frozen-probe SMOKE) | 0.1868 | [0.170, 0.204] | 0.1759 | 0.2857 | +0.0148 | -0.0571 |

> **M2-DINOv2 (aligned)** is the paper M2: frozen DINOv2 + Stage-1-style neural head (trunk+GELU+dropout+7-class), class-weighted CE, AdamW, cosine, early-stop, 5-fold, best-fold holdout. Only deviation from Stage 1: no train-time augmentation (frozen backbone). The frozen-probe SMOKE row (LightGBM/LR head) is a superseded preview.
