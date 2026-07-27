# A04 — `bin(PrimaireFossieleEnergie) == Energieklasse`?

Scope: four-city NTA 8800 residential certificates with a non-null EP2, n=404,025 (dropped 150 implausible EP2 values).

Boundaries tested (official NTA 8800 woningbouw, upper bound in kWh/m2.yr): A++++ <= 0, A+++ <= 50, A++ <= 75, A+ <= 105, A <= 160, B <= 190, C <= 250, D <= 290, E <= 335, F <= 380, G > 380.

## Overall agreement

| test | n | agreement |
|---|---:|---:|
| 11-class ladder on `PrimaireFossieleEnergie` | 404,025 | 78.4345% |
| 7-class ladder (repo `regression_kwh.PF_BINS`) | 404,025 | 96.1143% |
| 11-class, using `...EMGForfaitair` where present | 404,025 | 99.9849% |
| — subset that has an EMG forfaitair value | 248,458 | plain 64.9522% -> forfaitair 99.9960% |

## Agreement by registration year

| year | n | 11-class | 7-class |
|---|---:|---:|---:|
| 2021 | 49,232 | 78.605% | 96.336% |
| 2022 | 64,439 | 73.898% | 95.593% |
| 2023 | 62,198 | 79.583% | 96.426% |
| 2024 | 95,283 | 78.887% | 96.091% |
| 2025 | 107,569 | 80.297% | 96.595% |
| 2026 | 25,304 | 77.209% | 94.293% |

## Agreement by Berekeningstype

| calc type | n | 11-class |
|---|---:|---:|
| NTA 8800:2020 (basisopname woningbouw) | 63,778 | 84.098% |
| NTA 8800:2020 (detailopname woningbouw) | 12,746 | 41.276% |
| NTA 8800:2022 (basisopname woningbouw) | 54,979 | 80.891% |
| NTA 8800:2022 (detailopname woningbouw) | 11,670 | 53.865% |
| NTA 8800:2023 (basisopname woningbouw) | 64,937 | 83.821% |
| NTA 8800:2023 (detailopname woningbouw) | 9,802 | 58.233% |
| NTA 8800:2024 (basisopname woningbouw) | 163,320 | 82.594% |
| NTA 8800:2024 (detailopname woningbouw) | 22,793 | 53.556% |

## The 87,130 mismatches (21.565%)

| registered - binned (class steps) | n |
|---:|---:|
| -8 | 1 |
| -6 | 1 |
| -5 | 4 |
| -4 | 6 |
| -3 | 18 |
| -2 | 10 |
| -1 | 23 |
| +1 | 44,908 |
| +2 | 36,043 |
| +3 | 5,611 |
| +4 | 427 |
| +5 | 68 |
| +6 | 8 |
| +7 | 2 |

| registered | binned | n | median EP2 | median EP1 | median renewables % |
|---|---|---:|---:|---:|---:|
| A++ | A+++ | 16,497 | 31.2 | 60.8 | 62.0 |
| A+ | A+++ | 14,505 | 41.6 | 70.6 | 58.8 |
| A | A++ | 12,515 | 65.6 | 84.3 | 51.8 |
| A+ | A++ | 12,437 | 56.5 | 68.9 | 46.8 |
| A | A+ | 9,703 | 84.8 | 90.3 | 46.1 |
| C | A | 4,443 | 127.3 | 131.4 | 47.7 |
| A | A+++ | 3,736 | 36.6 | 87.4 | 73.4 |
| B | A+ | 3,548 | 94.7 | 100.9 | 50.8 |
| B | A | 3,479 | 116.3 | 107.3 | 40.4 |
| A+++ | A++++ | 1,780 | -5.7 | 57.0 | 108.7 |

Share of mismatches that carry an EMG forfaitair value: 99.9%; share within 2 kWh/m2.yr of a class boundary (rounding): 12.1%.

## Consequence for the kWh regression route

`src/stage2/extract_kwh.py` takes `PrimaireFossieleEnergie` as the regression target `pf_kwh`, but the registered label is the binning of `PrimaireFossieleEnergieEMGForfaitair` wherever that column is filled. For those certificates the regression target and the classification label are two different quantities.

| quantity | value |
|---|---:|
| certificates with an EMG forfaitair value | 248,458 (61.5%) |
| of those, share where the two columns actually differ (>1 kWh/m2.yr) | 38.7% |
| median absolute difference where they differ | 45.5 kWh/m2.yr |
| p90 absolute difference where they differ | 81.8 kWh/m2.yr |
| mean absolute difference over all EMG rows | 19.3 kWh/m2.yr |
| 7-class label mismatch using plain PF | 3.89% |
| 7-class label mismatch using EMG forfaitair | 0.01% |

So the two definitions coincide for most certificates and diverge sharply for a minority (39% of the EMG rows, median gap 46 kWh/m2.yr). On the 7-class ladder the resulting label error is 3.89% — small, but it is concentrated in exactly the district-heating stock that dominates Amsterdam and Rotterdam, and it is removable at zero cost by reading the column the register itself bins.

## Empirical boundaries implied by the data

For each class, the observed EP2 range. Under an exact binning the max of class k equals the min of class k+1 up to rounding.

| class | n | min EP2 | max EP2 | official upper bound |
|---|---:|---:|---:|---:|
| A++++ | 2,342 | -199.23 | 0.00 | 0 |
| A+++ | 19,944 | -93.03 | 50.00 | 50 |
| A++ | 27,495 | -31.51 | 1138.39 | 75 |
| A+ | 44,640 | -9.09 | 248.77 | 105 |
| A | 104,425 | -9.71 | 472.63 | 160 |
| B | 57,171 | 24.13 | 691.70 | 190 |
| C | 79,785 | 35.72 | 581.23 | 250 |
| D | 31,467 | 44.36 | 503.09 | 290 |
| E | 17,166 | 66.38 | 335.00 | 335 |
| F | 9,358 | 73.11 | 380.00 | 380 |
| G | 10,232 | 137.78 | 1486.90 | — |
