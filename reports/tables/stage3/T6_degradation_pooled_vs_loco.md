# Table 6 — Pooled (RQ3a) vs LOCO-Amsterdam (RQ3b) degradation

Same models, same pipeline; the only change is that Amsterdam is entirely
held out of training (train = R+U+D 2,075 buildings; test = Amsterdam 8,011).
Pooled hold-out n=2,018 (upstream) / 2,016 (downstream); VLM LOCO on the
1,595-building comparable subset.

## Upstream attribute extraction

| model | metric | pooled | LOCO | Δ |
|---|---|---:|---:|---:|
| DINOv2 frozen | type_acc | 0.9014 | 0.8969 | **−0.005** |
| | year MAE (yr) | 9.45 | 19.50 | **+10.1 (×2.1)** |
| | period_acc | 0.9009 | 0.8760 | −0.025 |
| | joint_acc | 0.8251 | 0.7988 | −0.026 |
| ResNet-50 ft | type_acc | 0.9148 | 0.8325 | **−0.082** |
| | year MAE (yr) | 11.82 | 19.17 | +7.4 (×1.6) |
| | period_acc | 0.8915 | 0.8835 | −0.008 |
| | joint_acc | 0.8256 | 0.7435 | **−0.082** |
| InternVL3 (ZS) | type_acc | 0.4945 | 0.4815 | −0.013 |
| | joint_acc | 0.3413 | 0.3624 | +0.021 |

**majority-cell baseline**: pooled 0.7904 → LOCO **0.8678** (Amsterdam is even
more AB|NL.01-concentrated). Both trained paradigms' joint_acc fall BELOW the
LOCO baseline (DINOv2 0.799 < 0.868; ResNet 0.744 < 0.868): under cross-city
shift the model no longer beats "always guess the dominant cell".

## Downstream energy (A–G classification)

| route | macro-F1 pooled → LOCO | κ pooled → LOCO |
|---|---|---|
| M1 (GT-attribute) | 0.1720 → 0.1429 | 0.2330 → **0.0263** |
| M3-DINOv2 | 0.1500 → 0.1412 | 0.0825 → 0.0295 |
| M3-ResNet50 | 0.1490 → 0.1462 | 0.0913 → 0.0363 |
| M3−M1 gap (mF1) | −0.022 → **−0.002** | — |

## Downstream energy (kWh regression, MAE)

| route | pooled | LOCO |
|---|---:|---:|
| M1-reg | 54.8 | 65.3 |
| M3-DINOv2-reg | 59.2 | 62.5 |
| M3-ResNet50-reg | 60.2 | 63.9 |

## Readings

1. **Fine-tuned degrades most (hypothesis confirmed, upstream).** ResNet-50 ft
   type_acc −0.082 and joint −0.082 vs DINOv2 frozen −0.005 / −0.026. The
   frozen self-supervised backbone is markedly more shift-robust; ResNet
   overfits the 2,075-building source pool (train F1 0.98 vs val 0.52).
2. **The cell-assignment claim does NOT survive LOCO.** Both trained paradigms
   fall below the (higher) majority-cell baseline. Root cause is isolated:
   type transfers (DINOv2 0.897) but **year regression collapses cross-city**
   (MAE ×2.1), dragging period → cell wrong.
3. **Substitution cost vanishes because M1 itself collapses.** M1 κ 0.233 →
   0.026: GT attributes give almost no downstream signal on unseen Amsterdam.
   The M3−M1 gap shrinks to −0.002 — the bottleneck under shift is the
   task/pool, not vision. Consistent with the pooled 82%-downstream error split.
4. **Downstream washout.** Despite far worse upstream, M3-ResNet50 downstream is
   marginally ABOVE M3-DINOv2 (κ 0.036 vs 0.030): once everything is pinned near
   the M0 floor, upstream differences no longer propagate. Report both, do not
   over-read the flip.
