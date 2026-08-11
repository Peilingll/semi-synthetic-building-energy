# A07 — is the low score caused by imbalance? (binary A-C | D-G)

Feature set fixed at S_full (type, bouwjaar, 4 U-values, num_floors, city), hyper-parameters fixed, 5-fold OOF. Only the pool and the class weighting change, so the differences are attributable.

The first row uses the frozen Stage-1 dev folds and is therefore the row comparable to T2a; the rest use a stratified shuffle split, the only option for the down-sampled and full-stock pools, and are compared among themselves.

| run | n | composition | macro-F1 | quad. kappa | acc |
|---|---:|---|---:|---:|---:|
| dev, frozen Stage-1 folds | 8,068 | top cell 79.5%, largest class 70.3%, smallest class 29.7% | **0.4776** | 0.0460 | 0.6936 |
| dev (reference) | 8,068 | top cell 79.5%, largest class 70.3%, smallest class 29.7% | **0.4773** | 0.0498 | 0.6967 |
| dev + class_weight=balanced | 8,068 | top cell 79.5%, largest class 70.3%, smallest class 29.7% | **0.5479** | 0.1581 | 0.5564 |
| dev, cell-capped at 651 | 2,303 | top cell 28.3%, largest class 78.5%, smallest class 21.5% | **0.5635** | 0.1425 | 0.7594 |
| dev, every class capped at 313 | 2,191 | top cell 83.1%, largest class 57.1%, smallest class 42.9% | **0.5872** | 0.2000 | 0.6276 |
| full stock, same n as dev | 8,068 | top cell 43.3%, largest class 76.2%, smallest class 23.8% | **0.5645** | 0.1601 | 0.7583 |
| full stock, all | 124,784 | top cell 43.2%, largest class 76.5%, smallest class 23.5% | **0.5414** | 0.1428 | 0.7758 |
| full stock + class_weight=balanced | 124,784 | top cell 43.2%, largest class 76.5%, smallest class 23.5% | **0.5981** | 0.2701 | 0.6174 |

## What each comparison isolates

| comparison | effect | d macro-F1 |
|---|---|---:|
| dev (reference) -> dev + class_weight=balanced | class weighting only | +0.0706 |
| dev (reference) -> dev, cell-capped at 651 | breaking the AB|NL.01 monopoly | +0.0862 |
| dev (reference) -> dev, every class capped at 313 | perfect label balance | +0.1099 |
| dev (reference) -> full stock, same n as dev | less skewed pool, same n | +0.0872 |
| full stock, same n as dev -> full stock, all | 15x more data | -0.0231 |
| dev (reference) -> full stock, all | pool + size combined | +0.0641 |

Reference macro-F1 on the dev pool: 0.4773 (frozen folds: 0.4776).
