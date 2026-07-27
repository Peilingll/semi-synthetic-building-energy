# A01 — `compactheid` provenance and leakage audit

## T1 — Provenance in code

| consumer | column read | source file |
|---|---|---|
| `src/stage2/svi_compactheid.py:46` | `Compactheid` | raw EP-Online CSV |
| `src/stage2/m1_plus_fullstock.py:178` | `Compactheid` | raw EP-Online CSV |
| `src/stage2/m1_plus_fullstock.py:179` | `GebruiksoppervlakteThermischeZone` | raw EP-Online CSV |

No code path derives `compactheid` from 3DBAG. The 3DBAG geometric ratio exists separately as `shape_factor` (`src/lod2_features.py:49`, `envelope_area / volume_lod22`) and is **not** what the ablation used.

## T2 — Is EP `Compactheid` a building-level quantity?

Four-city NTA 8800 residential certificates: n=413,761 over 134,007 pands; 45,566 pands carry >= 2 certificates.

| quantity | value |
|---|---:|
| pands with >= 2 certs where all `Compactheid` identical | 1.7% |
| median within-pand std (multi-cert pands) | 0.411 |
| median within-pand range (max-min) | 0.790 |
| 90th pct within-pand range | 1.480 |
| overall value range (p5-p95) | 0.44-2.14 |

## T3 — Agreement with the 3DBAG geometric surface/volume ratio

Side finding: `residential_with_3d_features.parquet` itself carries 1 pands with implausible `volume` (max 4.3e21 m3) and 2 with `envelope_area` <= 0 across the four cities; they are excluded here.

| comparison | n | Pearson r | Spearman | R2 (identity) | median EP | median BAG | median ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| EP latest vs 3DBAG shape_factor | 113,661 | 0.626 | 0.605 | -4.034 | 1.46 | 0.45 | 2.94 |
| EP median vs 3DBAG shape_factor | 113,688 | 0.720 | 0.706 | -4.469 | 1.42 | 0.45 | 2.96 |
| EP latest vs shape_factor incl. party walls | 113,753 | 0.397 | 0.410 | -2.215 | 1.46 | 0.73 | 1.86 |

### By Gebouwtype (EP latest vs shape_factor)

| Gebouwtype | n | Pearson r | Spearman | median EP | median BAG |
|---|---:|---:|---:|---:|---:|
| Appartement | 65,854 | 0.391 | 0.378 | 1.26 | 0.37 |
| Rijwoning hoek | 10,646 | 0.639 | 0.669 | 1.99 | 0.70 |
| Rijwoning tussen | 33,986 | 0.676 | 0.724 | 1.50 | 0.53 |
| Twee-onder-één-kap | 1,399 | 0.617 | 0.667 | 1.95 | 0.69 |
| Vrijstaande woning | 1,356 | 0.650 | 0.645 | 2.20 | 0.75 |
| Woongebouw met niet-zelfstandige woonruimte | 417 | 0.325 | 0.218 | 1.35 | 0.38 |

### By BAG unit count per pand

| units in pand | n | Pearson r | median EP | median BAG |
|---|---:|---:|---:|---:|
| 1 | 45,895 | 0.794 | 1.61 | 0.57 |
| 2 | 12,546 | 0.709 | 1.40 | 0.47 |
| 3-4 | 27,709 | 0.385 | 1.21 | 0.37 |
| 5-10 | 19,953 | 0.249 | 1.26 | 0.37 |
| >10 | 7,558 | 0.284 | 1.13 | 0.35 |

## T4 — `floor_area` provenance

`m1_plus_fullstock.load_ep_geometry` reads `GebruiksoppervlakteThermischeZone`: the **thermal-zone floor area of the certified unit**, not the pand.

| comparison | n | Pearson r | Spearman | median EP | median BAG | median ratio |
|---|---:|---:|---:|---:|---:|---:|
| EP thermal zone vs BAG floor_area_estimated (grond x floors) | 113,755 | -0.009 | -0.122 | 82.9 | 266.5 | 0.34 |
| EP thermal zone vs BAG oppervlakte_max (largest VBO) | 113,795 | 0.118 | 0.627 | 82.9 | 95.0 | 0.97 |
| EP thermal zone vs BAG b3_opp_grond (footprint) | 113,777 | 0.003 | -0.029 | 82.9 | 68.7 | 1.23 |

Within-pand spread of `GebruiksoppervlakteThermischeZone` among multi-cert pands: median std 8.8 m2 (median pand area 69.5 m2) -> per-unit, as expected.

## T5 — How much of the label does `compactheid` explain?

Spearman against the two NTA 8800 calculation outputs that define the label, on the dev pool (n=8,068), overall and inside the dominant archetype cell.

| subset | n | rho(compactheid, EP1 energiebehoefte) | rho(compactheid, EP2 PF) | rho(shape_factor, EP1) |
|---|---:|---:|---:|---:|
| dev, all cells | 8,049 | 0.440 | 0.351 | 0.177 |
| dev, cell AB|NL.01 | 6,407 | 0.470 | 0.391 | 0.161 |
| dev, cell AB|NL.03 | 384 | 0.818 | 0.638 | 0.180 |
| dev, cell AB|NL.04 | 211 | 0.810 | 0.538 | 0.383 |
| dev, cell TH|NL.01 | 651 | 0.108 | 0.151 | 0.057 |

## T6 — Re-running the headline ablation with real geometry

Same protocol as the 2026.06.26 ablation (dev pool, 5-fold OOF LightGBM, fixed HP): each feature added singly to S_full.

| added feature | provenance | macro-F1 | quad. kappa | acc | d macro-F1 |
|---|---|---:|---:|---:|---:|
| — (S_full base) | — | 0.1804 | 0.1865 | 0.3488 | — |
| `ep_compactheid` | **EP-Online certificate** (leak) | 0.2629 | 0.3571 | 0.3677 | +0.0825 |
| `shape_factor` | 3DBAG envelope/volume (clean) | 0.2078 | 0.2151 | 0.3325 | +0.0274 |
| `ep_floor_area` | **EP-Online certificate** (leak) | 0.2210 | 0.2420 | 0.3444 | +0.0406 |
| `floor_area_estimated` | 3DBAG grond x floors (clean) | 0.2084 | 0.2259 | 0.3359 | +0.0280 |
| `volume` | 3DBAG lod22 (clean) | 0.2037 | 0.2136 | 0.3286 | +0.0233 |
| `shared_ratio` | 3DBAG party-wall frac (clean) | 0.2095 | 0.1954 | 0.3339 | +0.0291 |
| all four clean 3DBAG features | 3DBAG only | 0.2169 | 0.2400 | 0.3339 | +0.0365 |
