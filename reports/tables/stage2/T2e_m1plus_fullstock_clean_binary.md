# Table 2e — M1+ upper-envelope arm on the full-stock pool (clean 3DBAG geometry, binary A-C | D-G)

n = 124,784 buildings (4 cities, registry pool, no image requirement).
Protocol identical to E6 (fixed-HP LightGBM, 5-fold OOF, class_weight=None).
Tier C = CBS buurt 2023: avg WOZ value + owner-occupied share (PDOK wijkenbuurten).

**Clean variant.** The two EP-Online certificate columns `compactheid` and `floor_area` (audit A01: per dwelling unit, NTA 8800 calculation inputs, absent for any building without a certificate) are replaced by their 3DBAG counterparts `shape_factor` and `floor_area_estimated`. **This is the variant to quote as a registry-only reference**; the original run below is an 'if you already hold the certificate' figure and is not reachable for the population the method targets.

Original (leaked) run for comparison: macro-F1 0.3513 / kappa 0.5636 / acc 0.5101 at 14 features (7-class).

| run | features | macro-F1 | quadratic kappa | accuracy |
|---|---:|---:|---:|---:|
| E6_base | 14 | 0.6172 | 0.2642 | 0.7930 |
| +woz | 15 | 0.6334 | 0.2919 | 0.7978 |
| +koop | 15 | 0.6343 | 0.2933 | 0.7978 |
| M1_plus | 16 | 0.6374 | 0.2981 | 0.7981 |
