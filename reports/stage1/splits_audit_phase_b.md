# Stage 1 Phase B Splits Audit

Two-stage StratifiedGroupKFold: 20% hold-out test + 80% dev 5-fold CV.

## Provenance

```
sha256_prefix: 7b877c7d4c066832
n_buildings:   2018
random_state:  42
n_splits:      5
strata_keys:   city,Gebouwtype,Energieklasse,tabula_period
created_with:  src/stage1/splits.py
```

## Sizes

- Training universe (manifest ∩ GT): **10,086** buildings
- Hold-out test: **2,018** buildings (20.0%)
- Dev set: **8,068** buildings (80.0%)
- Dev folds: **5** (val ~1,613 per fold)

## Hold-out vs Dev distribution

### By city

| city | Holdout | Dev | Holdout % |
|---|---:|---:|---:|
| amsterdam | 1595 | 6416 | 19.9% |
| rotterdam | 286 | 1129 | 20.2% |
| utrecht | 103 | 384 | 21.1% |
| delft | 34 | 139 | 19.7% |

### By Gebouwtype (Stage 1 target)

| Gebouwtype | Holdout | Dev | Holdout % |
|---|---:|---:|---:|
| Appartement | 1776 | 7140 | 19.9% |
| Rijwoning tussen | 138 | 535 | 20.5% |
| Rijwoning hoek | 66 | 253 | 20.7% |
| Twee-onder-één-kap | 13 | 56 | 18.8% |
| Vrijstaande woning | 13 | 50 | 20.6% |
| Woongebouw met niet-zelfstandige woonruimte | 12 | 34 | 26.1% |

### By building_type (4-class, derived)

| building_type | Holdout | Dev | Holdout % |
|---|---:|---:|---:|
| AB | 1776 | 7140 | 19.9% |
| TH | 204 | 788 | 20.6% |
| SFH | 26 | 106 | 19.7% |
| MFH | 12 | 34 | 26.1% |

### By Energieklasse (Stage 2/3 target)

| Energieklasse | Holdout | Dev | Holdout % |
|---|---:|---:|---:|
| C | 625 | 2523 | 19.9% |
| A | 375 | 1500 | 20.0% |
| B | 329 | 1286 | 20.4% |
| D | 281 | 1121 | 20.0% |
| E | 155 | 608 | 20.3% |
| F | 88 | 351 | 20.0% |
| G | 77 | 313 | 19.7% |
| A+ | 62 | 260 | 19.3% |
| A++ | 18 | 74 | 19.6% |
| A+++ | 7 | 28 | 20.0% |
| A++++ | 1 | 4 | 20.0% |

### By tabula_period

| tabula_period | Holdout | Dev | Holdout % |
|---|---:|---:|---:|
| NL.01 | 1787 | 7161 | 20.0% |
| NL.03 | 112 | 445 | 20.1% |
| NL.04 | 69 | 263 | 20.8% |
| NL.05 | 29 | 115 | 20.1% |
| NL.02 | 16 | 48 | 25.0% |
| NL.06 | 5 | 36 | 12.2% |

## Dev 5-fold composition (val side)

| Fold | Train | Val | val_amsterdam | val_rotterdam | val_utrecht | val_delft | val_SFH | val_TH | val_MFH | val_AB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6454 | 1614 | 1279 | 222 | 78 | 35 | 21 | 156 | 7 | 1430 |
| 1 | 6455 | 1613 | 1284 | 226 | 79 | 24 | 21 | 160 | 5 | 1427 |
| 2 | 6454 | 1614 | 1285 | 228 | 76 | 25 | 22 | 154 | 9 | 1429 |
| 3 | 6454 | 1614 | 1283 | 230 | 75 | 26 | 23 | 155 | 8 | 1428 |
| 4 | 6455 | 1613 | 1285 | 223 | 76 | 29 | 19 | 163 | 5 | 1426 |

## Reproducibility note

All splits are reproducible by running `python -m src.stage1.splits` with
the same `random_state` (42). The SHA256 prefix above is a stable digest of the
sorted hold-out `pand_id` list and should match across re-runs.