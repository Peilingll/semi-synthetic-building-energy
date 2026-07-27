# T7 — H_tr instrument: scoring cell predictions by physical consequence

The per-building EPC label cannot resolve Stage-1 quality (audit A06: its full range is macro-F1 0.068–0.172, while perfecting Stage 1 is worth +0.022). Neither can per-building measured demand (range 3.7 kWh/m²·yr, model separation 0.3). This table scores the **same** hold-out cell predictions by what the error costs physically instead of by 0/1 loss.

`H_tr' = [U_wall·A_wall·(1−WWR) + U_win·A_wall·WWR + U_roof·A_roof + U_floor·A_ground] / A_floor`, W/(K·m²). U from the corrected TABULA lookup, areas from 3DBAG, party walls adiabatic. No model is retrained — this reads the existing hold-out prediction files.

WWR = 0.25 in the main table; swept over (0.15, 0.25, 0.35) below. 95% CI from 1000 bootstrap resamples of the hold-out buildings.

## Building level

| model | n | joint cell acc | H_tr MAE | 95% CI | MAPE | bias |
|---|---:|---:|---:|---|---:|---:|
| DINOv2 frozen | 2,016 | 0.825 | **0.150** | [0.133, 0.169] | 12.9% | +0.108 |
| ResNet-50 ft | 2,016 | 0.826 | **0.157** | [0.139, 0.177] | 13.7% | +0.095 |
| InternVL3 (ZS) | 2,014 | 0.342 | **0.474** | [0.449, 0.500] | 26.3% | +0.169 |

Separation: **3.2×** between DINOv2 frozen (joint 0.825) and InternVL3 (ZS) (joint 0.342), non-overlapping CIs. On the EPC-label instrument the same two models differ by 1.09×.

## Stock level

Floor-area-weighted total heat-loss coefficient, predicted vs registry cells. Per-building random error averages out at 1/√n here; a systematic cell bias does not — which is why this readout is the one that matters for UBEM use.

| model | stock deviation |
|---|---:|
| DINOv2 frozen | **+10.0%** |
| ResNet-50 ft | **+10.5%** |
| InternVL3 (ZS) | **+16.6%** |

All three over-estimate: the regression-to-NL.01 error found in T4 makes the stock look less insulated than the registry says.

## WWR sensitivity

The one free parameter. Ordering and separation are stable across it.

| model | WWR 0.15 | WWR 0.25 | WWR 0.35 |
|---|---:|---:|---:|
| DINOv2 frozen | 0.150 | 0.150 | 0.150 |
| ResNet-50 ft | 0.157 | 0.157 | 0.157 |
| InternVL3 (ZS) | 0.495 | 0.474 | 0.453 |

## What this table is and is not

**Is**: a physically weighted rescoring of the Stage-1 confusion matrix. The weights come from TABULA and 3DBAG, not from the modeller, and they convert "which cell was wrong" into "how much that costs". Confusing NL.01 with NL.02 is cheap; confusing NL.01 with NL.05 is not; 0/1 accuracy scores them the same.

**Is not**: validation against reality. It compares SVI-assigned cells with registry-assigned cells, so it inherits whatever is wrong with the registry archetype — see A06, which shows the archetype chain explains only R²=0.215 of measured demand and that no downstream metric can close that gap.
