# Table 5 — LOCO-amsterdam pool composition

Train = all imaged R+U+D buildings (n=2075, pooled dev+holdout merged); test = all imaged amsterdam buildings (n=8011).
Strictly comparable zero-shot subset (test INTERSECT pooled hold-out): n=1595.

## Size class x pool

| building_type | test (amsterdam) | train (R+U+D) | test (amsterdam) % | train (R+U+D) % |
|---|---:|---:|---:|---:|
| AB | 7559 | 1357 | 94.4 | 65.4 |
| MFH | 41 | 5 | 0.5 | 0.2 |
| SFH | 17 | 115 | 0.2 | 5.5 |
| TH | 394 | 598 | 4.9 | 28.8 |

## TABULA period x pool

| tabula_period | test (amsterdam) | train (R+U+D) | test (amsterdam) % | train (R+U+D) % |
|---|---:|---:|---:|---:|
| NL.01 | 7359 | 1589 | 91.9 | 76.6 |
| NL.02 | 21 | 43 | 0.3 | 2.1 |
| NL.03 | 351 | 206 | 4.4 | 9.9 |
| NL.04 | 214 | 118 | 2.7 | 5.7 |
| NL.05 | 51 | 93 | 0.6 | 4.5 |
| NL.06 | 15 | 26 | 0.2 | 1.3 |

## Joint-cell coverage (sorted by test support)

| cell | test (amsterdam) | train (R+U+D) |
|---|---:|---:|
| AB|NL.01 | 6952 | 1059 |
| TH|NL.01 | 375 | 442 |
| AB|NL.03 | 331 | 153 |
| AB|NL.04 | 194 | 72 |
| AB|NL.05 | 48 | 36 |
| AB|NL.02 | 19 | 26 |
| SFH|NL.01 | 17 | 85 |
| AB|NL.06 | 15 | 11 |
| MFH|NL.01 | 15 | 3 |
| MFH|NL.03 | 13 | 1 |
| MFH|NL.04 | 11 | 1 |
| TH|NL.04 | 9 | 36 |
| TH|NL.03 | 7 | 46 |
| TH|NL.05 | 3 | 50 |
| MFH|NL.02 | 2 | 0 |
| SFH|NL.04 | 0 | 9 |
| SFH|NL.02 | 0 | 3 |
| SFH|NL.03 | 0 | 6 |
| TH|NL.02 | 0 | 14 |
| SFH|NL.06 | 0 | 5 |
| SFH|NL.05 | 0 | 7 |
| TH|NL.06 | 0 | 10 |

Cells present in test but absent from train: MFH|NL.02 (2 test buildings).
