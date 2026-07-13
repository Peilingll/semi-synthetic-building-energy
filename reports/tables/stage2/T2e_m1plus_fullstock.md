# Table 2e — M1+ upper-envelope arm on the full-stock pool

n = 124,784 buildings (4 cities, registry pool, no image requirement).
Protocol identical to E6 (fixed-HP LightGBM, 5-fold OOF, class_weight=None).
Tier C = CBS buurt 2023: avg WOZ value + owner-occupied share (PDOK wijkenbuurten).
E6 reference (log 2026.06.27): macro-F1 0.349 / kappa 0.560 / acc 0.510.

| run | features | macro-F1 | quadratic kappa | accuracy |
|---|---:|---:|---:|---:|
| E6_base | 14 | 0.3513 | 0.5636 | 0.5101 |
| +woz | 15 | 0.3566 | 0.5744 | 0.5169 |
| +koop | 15 | 0.3609 | 0.5787 | 0.5181 |
| M1_plus | 16 | 0.3632 | 0.5801 | 0.5206 |
