# A06 — Does the archetype chain track measured demand? (design B)

`cell -> TABULA U -> transmission loss over 3DBAG areas -> demand`, compared against the register's own **EP1 `Energiebehoefte`** (net demand). Not EP2, not the label — EP2 adds the installation side that no envelope model reaches (A02).

Four-city pands with GT cell + valid 3DBAG geometry + measured EP1: n=113,708.

## B-1 — Does `H_tr` from the registry cell correlate with measured EP1 at all?

| subset | n | Pearson r | R2 | Spearman rho |
|---|---:|---:|---:|---:|
| all four cities | 113,708 | 0.464 | 0.215 | 0.513 |
| type AB | 65,893 | 0.397 | 0.157 | 0.398 |
| type MFH | 418 | 0.488 | 0.238 | 0.477 |
| type SFH | 2,755 | 0.651 | 0.424 | 0.729 |
| type TH | 44,642 | 0.612 | 0.375 | 0.654 |
| city amsterdam | 52,671 | 0.453 | 0.205 | 0.500 |
| city delft | 3,310 | 0.520 | 0.271 | 0.529 |
| city rotterdam | 37,738 | 0.481 | 0.231 | 0.527 |
| city utrecht | 19,989 | 0.472 | 0.223 | 0.486 |

## B-2 — Implied vs measured demand per cell (uncalibrated: HDD 2900, ACH 0.5, gains 35, WWR 0.25)

| cell | n | median H_tr | implied kWh/m2.yr | measured EP1 median | ratio |
|---|---:|---:|---:|---:|---:|
| AB|NL.01 | 52,966 | 2.33 | 156 | 143 | 1.09 |
| TH|NL.01 | 25,633 | 4.16 | 280 | 153 | 1.83 |
| TH|NL.03 | 7,713 | 1.74 | 109 | 108 | 1.01 |
| AB|NL.03 | 7,189 | 1.50 | 96 | 105 | 0.92 |
| TH|NL.04 | 4,928 | 0.91 | 55 | 93 | 0.59 |
| AB|NL.04 | 2,706 | 0.72 | 42 | 87 | 0.48 |
| TH|NL.05 | 2,576 | 0.57 | 31 | 78 | 0.40 |
| TH|NL.02 | 2,271 | 2.87 | 189 | 134 | 1.41 |
| TH|NL.06 | 1,521 | 0.51 | 28 | 70 | 0.40 |
| AB|NL.02 | 1,471 | 2.01 | 133 | 128 | 1.03 |

A ratio far from 1 with everything else near 1 indicates a bad TABULA row rather than a modelling error.

## B-3 — Calibrated comparison on the hold-out

To avoid the assumed HDD/ACH/gains driving the result, EP1 is regressed on `H_tr` **once** on the dev pool with the registry cell (EP1 = 105.9 + 16.50·H_tr, n=8,050); the same two coefficients are then applied to every branch. Nothing else is fitted.

| model | joint cell acc | A: MAE(H_tr) | B_gt MAE | B_pred MAE | B_pred − B_gt |
|---|---:|---:|---:|---:|---:|
| DINOv2 frozen | 0.830 | 0.148 | 34.8 | 36.0 | +1.3 |
| ResNet-50 ft | 0.830 | 0.154 | 34.8 | 36.2 | +1.4 |
| InternVL3 ZS | 0.341 | 0.474 | 34.7 | 36.3 | +1.6 |

Units: A in W/(K·m²), B in kWh/(m²·yr).

## B-4 — Is B_gt large?

| reference quantity | value |
|---|---:|
| measured EP1 std (four cities) | 48.3 kWh/m2.yr |
| measured EP1 mean | 137.3 |
| MAE of predicting the global mean EP1 | 38.5 |
| MAE of the calibrated registry-cell model (dev) | 35.0 |
| median within-pand EP1 std (multi-cert pands) | 21.0 |

If the calibrated registry-cell model does not beat 'predict the mean', the archetype chain carries no usable information about measured demand, and any Stage-1 improvement is invisible in B by construction.
