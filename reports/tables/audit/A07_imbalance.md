# A07 — is the low score caused by imbalance?

Feature set fixed at S_full (type, bouwjaar, 4 U-values, num_floors, city), hyper-parameters fixed, 5-fold stratified OOF. Only the pool and the class weighting change, so the differences are attributable.

| run | n | composition | macro-F1 | quad. kappa | acc |
|---|---:|---|---:|---:|---:|
| dev (reference) | 8,068 | top cell 79.5%, largest class 31.3%, smallest class 3.9% | **0.1702** | 0.1839 | 0.3447 |
| dev + class_weight=balanced | 8,068 | top cell 79.5%, largest class 31.3%, smallest class 3.9% | **0.1882** | 0.1495 | 0.2053 |
| dev, cell-capped at 651 | 2,303 | top cell 28.3%, largest class 31.6%, smallest class 3.0% | **0.2343** | 0.3636 | 0.4151 |
| dev, every class capped at 313 | 2,191 | top cell 83.1%, largest class 14.3%, smallest class 14.3% | **0.2087** | 0.1835 | 0.2068 |
| full stock, same n as dev | 8,068 | top cell 43.3%, largest class 33.4%, smallest class 3.2% | **0.2094** | 0.3220 | 0.3985 |
| full stock, all | 124,784 | top cell 43.2%, largest class 32.8%, smallest class 3.0% | **0.2082** | 0.3757 | 0.4402 |
| full stock + class_weight=balanced | 124,784 | top cell 43.2%, largest class 32.8%, smallest class 3.0% | **0.2471** | 0.3466 | 0.3114 |

## What each comparison isolates

| comparison | effect | d macro-F1 |
|---|---|---:|
| dev (reference) -> dev + class_weight=balanced | class weighting only | +0.0180 |
| dev (reference) -> dev, cell-capped at 651 | breaking the AB|NL.01 monopoly | +0.0641 |
| dev (reference) -> dev, every class capped at 313 | perfect label balance | +0.0385 |
| dev (reference) -> full stock, same n as dev | less skewed pool, same n | +0.0392 |
| full stock, same n as dev -> full stock, all | 15x more data | -0.0012 |
| dev (reference) -> full stock, all | pool + size combined | +0.0380 |

Reference macro-F1 on the dev pool: 0.1702.
