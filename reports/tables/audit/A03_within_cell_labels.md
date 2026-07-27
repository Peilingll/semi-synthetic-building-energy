# A03 — EPC label distribution inside a TABULA cell

Cell = `building_type` x `tabula_period` (the archetype M1 consumes). Label = pand-level `Energieklasse` from `stage1_gt.parquet` (latest certificate per pand, A+..A++++ merged to A).

## Pool `dev` (n=8,068, 21 non-empty cells)

| cell | n | share of pool | modal | modal share | entropy (bits) | norm. entropy | classes present |
|---|---:|---:|---|---:|---:|---:|---:|
| AB|NL.01 | 6,416 | 79.5% | C | 0.324 | 2.524 | 0.899 | 7 |
| TH|NL.01 | 651 | 8.1% | C | 0.409 | 2.346 | 0.836 | 7 |
| AB|NL.03 | 387 | 4.8% | A | 0.398 | 1.827 | 0.651 | 7 |
| AB|NL.04 | 212 | 2.6% | A | 0.745 | 1.141 | 0.406 | 6 |
| SFH|NL.01 | 81 | 1.0% | C | 0.358 | 2.529 | 0.901 | 7 |
| AB|NL.05 | 67 | 0.8% | A | 0.985 | 0.112 | 0.040 | 2 |
| TH|NL.05 | 43 | 0.5% | A | 1.000 | 0.000 | 0.000 | 1 |
| TH|NL.03 | 42 | 0.5% | A | 0.381 | 1.665 | 0.593 | 4 |
| AB|NL.02 | 35 | 0.4% | A | 0.400 | 1.976 | 0.704 | 6 |
| TH|NL.04 | 33 | 0.4% | A | 0.939 | 0.330 | 0.117 | 2 |
| AB|NL.06 | 23 | 0.3% | A | 1.000 | 0.000 | 0.000 | 1 |
| MFH|NL.01 | 13 | 0.2% | C | 0.308 | 2.565 | 0.914 | 7 |

Pool-weighted mean within-cell entropy: **2.373 bits** (max 2.807); marginal label entropy of the pool: 2.491 bits -> a cell removes 0.118 bits of label uncertainty.

## Pool `manifest` (n=10,086, 22 non-empty cells)

| cell | n | share of pool | modal | modal share | entropy (bits) | norm. entropy | classes present |
|---|---:|---:|---|---:|---:|---:|---:|
| AB|NL.01 | 8,011 | 79.4% | C | 0.324 | 2.525 | 0.899 | 7 |
| TH|NL.01 | 817 | 8.1% | C | 0.406 | 2.348 | 0.836 | 7 |
| AB|NL.03 | 484 | 4.8% | A | 0.399 | 1.827 | 0.651 | 7 |
| AB|NL.04 | 266 | 2.6% | A | 0.748 | 1.118 | 0.398 | 6 |
| SFH|NL.01 | 102 | 1.0% | C | 0.363 | 2.527 | 0.900 | 7 |
| AB|NL.05 | 84 | 0.8% | A | 0.976 | 0.186 | 0.066 | 3 |
| TH|NL.03 | 53 | 0.5% | B | 0.415 | 1.598 | 0.569 | 4 |
| TH|NL.05 | 53 | 0.5% | A | 1.000 | 0.000 | 0.000 | 1 |
| TH|NL.04 | 45 | 0.4% | A | 0.889 | 0.583 | 0.208 | 3 |
| AB|NL.02 | 45 | 0.4% | A | 0.356 | 2.118 | 0.754 | 6 |
| AB|NL.06 | 26 | 0.3% | A | 1.000 | 0.000 | 0.000 | 1 |
| MFH|NL.01 | 18 | 0.2% | C | 0.278 | 2.636 | 0.939 | 7 |

Pool-weighted mean within-cell entropy: **2.375 bits** (max 2.807); marginal label entropy of the pool: 2.492 bits -> a cell removes 0.117 bits of label uncertainty.

## Pool `fullstock` (n=124,784, 24 non-empty cells)

| cell | n | share of pool | modal | modal share | entropy (bits) | norm. entropy | classes present |
|---|---:|---:|---|---:|---:|---:|---:|
| AB|NL.01 | 53,875 | 43.2% | C | 0.305 | 2.529 | 0.901 | 7 |
| TH|NL.01 | 28,677 | 23.0% | C | 0.344 | 2.485 | 0.885 | 7 |
| TH|NL.03 | 10,061 | 8.1% | A | 0.492 | 1.558 | 0.555 | 7 |
| AB|NL.03 | 7,611 | 6.1% | A | 0.375 | 1.863 | 0.664 | 7 |
| TH|NL.04 | 5,713 | 4.6% | A | 0.820 | 0.799 | 0.285 | 5 |
| TH|NL.02 | 3,416 | 2.7% | C | 0.370 | 2.052 | 0.731 | 7 |
| TH|NL.05 | 2,961 | 2.4% | A | 0.993 | 0.069 | 0.025 | 4 |
| AB|NL.04 | 2,850 | 2.3% | A | 0.756 | 1.095 | 0.390 | 7 |
| SFH|NL.01 | 2,064 | 1.7% | C | 0.239 | 2.729 | 0.972 | 7 |
| AB|NL.02 | 1,741 | 1.4% | C | 0.311 | 2.518 | 0.897 | 7 |
| TH|NL.06 | 1,678 | 1.3% | A | 0.998 | 0.021 | 0.007 | 3 |
| AB|NL.05 | 1,134 | 0.9% | A | 0.925 | 0.506 | 0.180 | 7 |

Pool-weighted mean within-cell entropy: **2.135 bits** (max 2.807); marginal label entropy of the pool: 2.387 bits -> a cell removes 0.252 bits of label uncertainty.

## Cell-only oracle: the explicit ceiling for M1

Predict each group's modal label. No model whose input is only that grouping can beat these numbers.

| pool | grouping | groups | acc | macro-F1 | quad. kappa |
|---|---|---:|---:|---:|---:|
| dev | TABULA cell | 21 | 0.363 | 0.130 | 0.192 |
| dev | cell x city | 63 | 0.365 | 0.137 | 0.195 |
| dev | type x exact bouwjaar | 389 | 0.398 | 0.228 | 0.241 |
| manifest | TABULA cell | 22 | 0.362 | 0.130 | 0.192 |
| manifest | cell x city | 66 | 0.365 | 0.138 | 0.194 |
| manifest | type x exact bouwjaar | 412 | 0.393 | 0.217 | 0.243 |
| fullstock | TABULA cell | 24 | 0.408 | 0.151 | 0.364 |
| fullstock | cell x city | 86 | 0.413 | 0.159 | 0.369 |
| fullstock | type x exact bouwjaar | 708 | 0.423 | 0.183 | 0.344 |

Reading: the TABULA cell alone caps macro-F1 at **0.130** (dev). M1 reaches 0.180 dev-OOF / 0.172 hold-out only because it also gets exact `bouwjaar`, `num_floors` and `city` on top of the cell — and the type x exact-year oracle caps that at 0.228. M1 is therefore within ~0.05 macro-F1 of the hard ceiling of its own feature set: the gap to a useful classifier is missing information, not a weak model.

## Figure

- `reports/figures/audit/A03_cell_label_mix.png`
