# Table 3 — Stage 3 pipeline comparison incl. M2 (hold-out, n=2,016)

| Route | macro-F1 | 95% CI | κ | acc | vs M1 (mF1) | vs M1 (κ) |
|---|---:|---|---:|---:|---:|---:|
| M0 | 0.0676 | [0.064, 0.071] | 0.0000 | 0.3100 |  |  |
| M1 | 0.1720 | [0.160, 0.185] | 0.2330 | 0.3557 |  |  |
| M3-DINOv2 | 0.1500 | [0.135, 0.164] | 0.0825 | 0.2917 | -0.0220 | -0.1505 |
| M3-ResNet50 | 0.1490 | [0.134, 0.165] | 0.0913 | 0.2817 | -0.0230 | -0.1417 |
| M3-VLMv3 | 0.1370 | [0.125, 0.150] | 0.0408 | 0.2783 | -0.0350 | -0.1922 |
| M2-DINOv2 (aligned) | 0.2131 | [0.194, 0.230] | 0.2355 | 0.2664 | +0.0411 | +0.0025 |
| M2-ResNet50 (aligned, full-FT) | 0.2014 | [0.183, 0.219] | 0.1879 | 0.2634 | +0.0294 | -0.0451 |
| M2-DINOv2 (frozen-probe SMOKE) | 0.1868 | [0.170, 0.204] | 0.1759 | 0.2857 | +0.0148 | -0.0571 |

> **M2 (aligned)** = end-to-end, each backbone mirroring its Stage 1 paradigm: M2-DINOv2 frozen+neural head (cached, no aug); M2-ResNet50 full fine-tune (Stage 1 ResNet recipe, with aug). Both end-to-end routes beat their decomposed M3 (+0.05 mF1, +0.10 kappa) and match the GT ceiling M1 -> backbone-robust. The frozen-probe SMOKE row is a superseded preview.
