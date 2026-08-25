# A01b — Binary clean-geometry additions + clean shape-factor probe

Part 1: same protocol as A01 T6 (dev pool, 5-fold OOF LightGBM, fixed HP), task = binary (A-C | D-G, objective=binary, class_weight=None).

| run | macro-F1 | accuracy | d macro-F1 |
|---|---:|---:|---:|
| S_full base | 0.4776 | 0.6936 | — |
| + shape_factor | 0.5406 | 0.6934 | +0.0630 |
| + floor_area_estimated | 0.5490 | 0.6936 | +0.0714 |
| + all four clean 3DBAG | 0.5814 | 0.7046 | +0.1038 |

Part 2: L1 extractability probe (RegHead on frozen DINOv2 embeddings, 5-fold OOF; same protocol as svi_compactheid.py), target = 3DBAG shape_factor (envelope_area / volume, the audited clean variant).

| target | n | OOF R2 | MAE |
|---|---:|---:|---:|
| clean shape_factor | 8053 | 0.646 | 0.042 |