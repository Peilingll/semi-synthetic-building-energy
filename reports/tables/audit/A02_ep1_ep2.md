# A02 — EP1 x EP2 structure and the envelope-only ceiling

Scope: all NTA 8800 residential certificates in Amsterdam / Rotterdam / Utrecht / Delft, no dedup, n=404,025 (133,947 pands). EP1 = `Energiebehoefte`, EP2 = `PrimaireFossieleEnergie` (kWh/m2.yr).

Plausibility filter applied: EP1 in [0, 1000], EP2 in [-300, 1500] kWh/m2.yr — drops 150 of 404,175 certificates (0.0371%). Those rows are data-entry errors in the national register (max observed EP1 1.08e7, EP2 3.70e8, `Compactheid` 1.4e5); they are large enough to flip any Pearson r on their own, which is why earlier inline correlations looked erratic.

## EP1 -> EP2 association

| subset | n | Pearson r | R2 (linear) | Spearman rho | rho^2 |
|---|---:|---:|---:|---:|---:|
| all | 404,025 | 0.822 | 0.675 | 0.829 | 0.687 |
| Appartement | 333,523 | 0.812 | 0.659 | 0.817 | 0.667 |
| Rijwoning hoek | 15,151 | 0.902 | 0.814 | 0.916 | 0.838 |
| Rijwoning tussen | 46,451 | 0.884 | 0.782 | 0.907 | 0.822 |
| Twee-onder-één-kap | 2,372 | 0.933 | 0.870 | 0.937 | 0.879 |
| Vrijstaande woning | 2,080 | 0.911 | 0.830 | 0.881 | 0.776 |
| Woongebouw met niet-zelfstandige woonruimte | 4,332 | 0.738 | 0.545 | 0.737 | 0.543 |

Adding `AandeelHernieuwbareEnergie` to a linear model of EP2: R2 0.675 -> 0.821 (n=404,025). The gap is installation-side information with no facade correlate.

## Label spread inside a 10 kWh/m2.yr slice of EP1 — 11-class ladder (A++++..G)

| quantity | median over slices |
|---|---:|
| certificates per slice | 5226 |
| distinct classes present | 9.0 |
| class-index span p5-p95 (steps) | 3.0 |
| std of class index | 0.94 |
| modal-class share (oracle acc at fixed EP1) | 0.589 |

## Label spread inside a 10 kWh/m2.yr slice of EP1 — 7-class merged (A..G, model convention)

| quantity | median over slices |
|---|---:|
| certificates per slice | 5226 |
| distinct classes present | 7.0 |
| class-index span p5-p95 (steps) | 2.0 |
| std of class index | 0.77 |
| modal-class share (oracle acc at fixed EP1) | 0.714 |

## Oracle ceiling: label from one scalar alone

200 equal-count bins, predict each bin's modal class. Upper bound for any model whose only information is that scalar.

| predictor | target ladder | acc | macro-F1 | quad. kappa |
|---|---|---:|---:|---:|
| EP1 energiebehoefte | 11-class | 0.505 | 0.398 | 0.801 |
| EP1 energiebehoefte | 7-class | 0.709 | 0.553 | 0.834 |
| EP2 primaire fossiele energie | 11-class | 0.814 | 0.790 | 0.951 |
| EP2 primaire fossiele energie | 7-class | 0.955 | 0.954 | 0.976 |

## The same ceiling on the experiment pool (comparable to M1/M2/M3)

dev pool, pand level, latest certificate per pand, n=8,052; modal label cross-fitted over the frozen 5 folds (no in-sample optimism).

| predictor | acc | macro-F1 | quad. kappa |
|---|---:|---:|---:|
| majority class | 0.3131 | 0.0681 | 0.0000 |
| TABULA cell | 0.3621 | 0.1283 | 0.1932 |
| EP1, 20 equal-count bins | 0.6325 | 0.5057 | 0.8280 |
| EP1, 50 equal-count bins | 0.6405 | 0.5770 | 0.8356 |
| EP1, 200 equal-count bins | 0.6376 | 0.5660 | 0.8280 |

Reference points from `reports/tables/stage3/T3_main.md` (hold-out): M0 acc 0.310 / mF1 0.068, M1 0.356 / 0.172, M2-DINOv2 0.266 / 0.213, M3-DINOv2 0.292 / 0.150.

**Use the 50-bin row (acc ~0.64 / macro-F1 ~0.58) as the envelope-only ceiling, not the 0.709 certificate-level figure above** — that one is in-sample and computed on a different, four-city certificate population.

## Figures

- `reports/figures/audit/A02_ep1_ep2_class.png`
- `reports/figures/audit/A02_ep1_ep2_renewable.png`
