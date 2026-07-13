# Table 4 — Joint TABULA-cell assignment (loco_amsterdam)

Cell = (size class x TABULA period). Joint = type AND period both correct.
majority_cell = always predicting the most frequent TRUE cell (baseline).
macro_cell_recall = unweighted mean recall over occupied cells.

| model | n | type_acc | period_acc | joint_acc | majority-cell baseline | macro-cell recall | cells |
|---|---:|---:|---:|---:|---:|---:|---:|
| DINOv2 frozen | 8011 | 0.8969 | 0.8760 | 0.7988 | 0.8678 | 0.1532 | 15 |
| ResNet-50 ft | 8011 | 0.8325 | 0.8835 | 0.7435 | 0.8678 | 0.1135 | 15 |
| InternVL3 (ZS) | 1595 | 0.4815 | 0.8276 | 0.3624 | 0.8658 | 0.0572 | 13 |

## DINOv2 frozen — per-cell recall (true cells, by support)

| true cell | support | recall |
|---|---:|---:|
| AB|NL.01 | 6952 | 0.884 |
| TH|NL.01 | 375 | 0.400 |
| AB|NL.03 | 331 | 0.236 |
| AB|NL.04 | 194 | 0.077 |
| AB|NL.05 | 48 | 0.000 |
| AB|NL.02 | 19 | 0.105 |
| SFH|NL.01 | 17 | 0.529 |
| MFH|NL.01 | 15 | 0.000 |
| AB|NL.06 | 15 | 0.067 |
| MFH|NL.03 | 13 | 0.000 |
| MFH|NL.04 | 11 | 0.000 |
| TH|NL.04 | 9 | 0.000 |
| TH|NL.03 | 7 | 0.000 |
| TH|NL.05 | 3 | 0.000 |
| MFH|NL.02 | 2 | 0.000 |

### DINOv2 frozen — top confusion pairs (true -> pred, joint errors)

| true cell | pred cell | count |
|---|---|---:|
| AB|NL.01 | TH|NL.01 | 459 |
| AB|NL.01 | AB|NL.02 | 155 |
| AB|NL.03 | AB|NL.01 | 118 |
| AB|NL.04 | AB|NL.01 | 83 |
| AB|NL.01 | AB|NL.03 | 76 |
| AB|NL.03 | AB|NL.02 | 67 |
| TH|NL.01 | AB|NL.01 | 64 |
| AB|NL.01 | TH|NL.02 | 52 |

## ResNet-50 ft — per-cell recall (true cells, by support)

| true cell | support | recall |
|---|---:|---:|
| AB|NL.01 | 6952 | 0.825 |
| TH|NL.01 | 375 | 0.469 |
| AB|NL.03 | 331 | 0.118 |
| AB|NL.04 | 194 | 0.031 |
| AB|NL.05 | 48 | 0.000 |
| AB|NL.02 | 19 | 0.000 |
| SFH|NL.01 | 17 | 0.118 |
| MFH|NL.01 | 15 | 0.000 |
| AB|NL.06 | 15 | 0.000 |
| MFH|NL.03 | 13 | 0.000 |
| MFH|NL.04 | 11 | 0.000 |
| TH|NL.04 | 9 | 0.000 |
| TH|NL.03 | 7 | 0.143 |
| TH|NL.05 | 3 | 0.000 |
| MFH|NL.02 | 2 | 0.000 |

### ResNet-50 ft — top confusion pairs (true -> pred, joint errors)

| true cell | pred cell | count |
|---|---|---:|
| AB|NL.01 | TH|NL.01 | 967 |
| AB|NL.03 | AB|NL.01 | 162 |
| AB|NL.04 | AB|NL.01 | 104 |
| AB|NL.01 | AB|NL.02 | 97 |
| TH|NL.01 | AB|NL.01 | 61 |
| AB|NL.03 | TH|NL.01 | 56 |
| TH|NL.01 | TH|NL.03 | 52 |
| AB|NL.03 | AB|NL.02 | 52 |

## InternVL3 (ZS) — per-cell recall (true cells, by support)

| true cell | support | recall |
|---|---:|---:|
| AB|NL.01 | 1381 | 0.401 |
| TH|NL.01 | 75 | 0.173 |
| AB|NL.03 | 65 | 0.169 |
| AB|NL.04 | 39 | 0.000 |
| AB|NL.05 | 10 | 0.000 |
| AB|NL.02 | 4 | 0.000 |
| MFH|NL.01 | 4 | 0.000 |
| SFH|NL.01 | 4 | 0.000 |
| TH|NL.04 | 4 | 0.000 |
| MFH|NL.03 | 3 | 0.000 |
| AB|NL.06 | 2 | 0.000 |
| MFH|NL.02 | 2 | 0.000 |
| MFH|NL.04 | 2 | 0.000 |

### InternVL3 (ZS) — top confusion pairs (true -> pred, joint errors)

| true cell | pred cell | count |
|---|---|---:|
| AB|NL.01 | TH|NL.01 | 682 |
| AB|NL.01 | AB|NL.03 | 51 |
| AB|NL.01 | AB|NL.02 | 43 |
| AB|NL.01 | SFH|NL.01 | 31 |
| AB|NL.03 | AB|NL.01 | 27 |
| AB|NL.03 | AB|NL.02 | 19 |
| AB|NL.04 | AB|NL.01 | 15 |
| TH|NL.01 | SFH|NL.01 | 14 |
