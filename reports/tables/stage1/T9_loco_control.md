# T9 — LOCO control: city shift or sample size? (test n=1,595)

Both arms train **2,075** buildings on the same cached DINOv2 embeddings with the same head and protocol, and predict the same Amsterdam test set. The only difference is whether Amsterdam is in the training pool, so sample size is no longer confounded with city composition.

- test set: 1,595 buildings — AB 94.1% / NL.01 91.8% / Amsterdam 100.0%
- arm A (loco, no Amsterdam): AB 65.4% / NL.01 76.6% / Amsterdam 0.0%
- arm B (matched, all four cities): AB 88.0% / NL.01 88.7% / Amsterdam 74.9%

Amsterdam is ~80% of the imaged stock, so arm B's pool is the *more* homogeneous one. Arm A trains on a more diverse but mismatched pool; arm B on a narrower pool that resembles the deployment site.

| arm | type acc | type macro-F1 | year MAE | period acc | joint cell | macro-cell recall |
|---|---:|---:|---:|---:|---:|---:|
| A — loco (no Amsterdam) | 0.9436 | 0.3972 | 18.13 | 0.8765 | 0.8408 | 0.1476 |
| B — matched (all cities) | 0.9674 | 0.5251 | 8.17 | 0.9335 | 0.9053 | 0.1953 |
| **B − A** | +0.0238 | +0.1279 | -9.96 | +0.0571 | +0.0646 | +0.0477 |

Majority-cell baseline on this test set: **0.8658** (13 occupied cells).

Reading: a large B − A means the cross-city gap is real at fixed n. A gap near zero means the headline LOCO degradation is a sample-size effect and the cross-city claim does not survive.

Both arms use the cached-embedding head, not the full augmented Stage 1 recipe, so these absolute values are not interchangeable with `T4_joint_cell_loco_amsterdam.md`; the A-vs-B gap is the result.
