# Chapter 4 experiment inventory and correspondence audit

## Scope and decision rule

This inventory was reconstructed from the current Method chapter, executable source code, saved prediction files, generated result tables, data-integrity audits, and the AD/JQ reviewer comments. Only analyses with a defined data population, reproducible comparison, and saved evidence are promoted to the main Chapter 4 narrative. Leave-one-city-out evaluation with Amsterdam is excluded at the author's request.

## One-to-one chapter structure

| ID | Method stage | Experimental Design | Results | Main evidence | Discussion question |
|---|---|---|---|---|---|
| Evaluation setup | Stage 1, Dataset Construction | Evaluation sample, building-level split, comparison-specific denominators, and reference-target checks | Dataset composition and certificate consistency | `A05_count_reconcile.md`, `T1_per_city_train_holdout.md`, `T2d_label_entropy.md` | What sample and target uncertainty bound all three experiments? |
| E1 | Stage 2, Building Attribute Prediction | Building type, construction year and TABULA period, floor count | Attribute performance and error patterns for ResNet-50, DINOv2, and InternVL3-2B | Stage 1 holdout JSON files, `T3_model_comparison.md` | Which model configuration provides the most reliable explicit attributes? |
| E2 | Stage 3, Archetype Assignment and Thermal Parameters | Joint TABULA-NL cell assignment and thermal consequence scoring | Joint-cell accuracy, macro-cell recall, htr difference, stock-level bias, WWR sensitivity | `T4_joint_cell.md`, `T7_htr_instrument.md` | Do categorical assignment errors also change the assigned thermal properties materially? |
| E3 | Stage 4, Energy Class Prediction | Reference attribute, predicted attribute, and direct image conditions for binary and seven class targets | Binary and seven class comparisons and feature contribution analysis | Stage 3 metrics and Stage 2 ablations | How does the source of building information affect registered energy class prediction? |

## Verified metrics and comparisons

### Evaluation setup

- Published SVI delivery: 10,104 buildings and 47,238 images.
- Experimental population: 10,086 buildings and 47,150 images.
- Development set: 8,068 buildings. Fixed holdout: 2,018 buildings.
- Comparison subset for all energy-class conditions: 2,016 buildings because InternVL3-2B did not return a valid attribute record for two holdout buildings.
- Reference-target audit: prevalence of multiple certificates, within-building energy-class disagreement, modal share, latest-to-modal agreement, and within-building variation in the continuous energy indicator.

### E1: Building attribute prediction

- Building type: accuracy, macro F1, per-class precision/recall/F1, and confusion matrix.
- Construction year: MAE, R2, bootstrap interval, and TABULA period accuracy.
- Floor count: MAE, R2, bootstrap interval, exact rounded agreement, and within-one-floor agreement.
- Headline holdout results: DINOv2 is best in six of seven headline columns; ResNet-50 has the highest type accuracy; InternVL3-2B without parameter updating is lowest in every headline column.

### E2: Archetype assignment and thermal consequences

- Joint cell: joint accuracy and macro-cell recall.
- Baselines: uniform assignment over 24 cells and majority-cell assignment.
- Thermal consequences: building-level htr MAE and R2, plus floor-area-weighted stock-level bias.
- Sensitivity: WWR values 0.15, 0.25, and 0.35.
- Boundary: htr compares predicted-cell and reference-cell assignments. It is not validation against measured heat loss or energy use.

### E3: Energy class prediction

- Information-source conditions: reference attributes, predicted attributes, and direct image prediction, plus an analytic uniform random baseline.
- Primary binary target: A to C versus D to G. Reported metrics are macro precision, macro recall, macro F1, and class level recall at the predefined threshold of 0.5.
- Secondary seven-class target: macro F1, exact accuracy, and plus-or-minus-one-class accuracy.
- Feature ablation: leave one input or input group out from the structured model; clean 3DBAG geometry additions are reported separately.

## Analyses not promoted to the main comparison

| Analysis | Decision | Reason |
|---|---|---|
| LOCO Amsterdam and cross-city transfer | Exclude | Explicit author decision. Geographic transfer remains a limitation, not a reported Chapter 4 experiment. |
| EP `Compactheid` and EP thermal-zone floor area | Exclude | These are certificate fields and therefore leak target-side information. Only clean 3DBAG geometry results may be retained. |
| Regression from inputs to primary fossil energy and subsequent binning | Retain in repository, not headline | It answers a different target formulation and is not required by the three objectives. Audit A04 also required correction of the fossil-energy field used as the target. |
| Rate matched operating point | Retain in repository, not headline | It is a threshold sensitivity analysis rather than evidence required for the objective aligned comparison. |
| Type and period error decomposition | Retain in repository, not headline | It omits floor count and does not provide complete attribution of energy prediction errors. |
| Sample composition interventions | Retain in repository, not headline | The main chapter reports the fixed evaluation sample and class specific performance without introducing additional sampling experiments. |
| EP1 to EP2 association and envelope-only ceiling | Data-integrity support | Useful for interpreting the limits of envelope information, but not one of the three objective-aligned experiments. |
| Measured-demand comparison in A06 | Data-integrity support | It establishes the boundary of htr as an internal consequence metric, but does not validate the registered energy-class task. |
| Training curves and smoke tests | Reproducibility support | They document model development and numerical health rather than answer a research objective. |
| Geometry extraction values without a saved generated report | Do not use as headline evidence | Code exists, but the current repository does not contain a complete generated output table for every reported geometry target. |

## Corrections required during rewriting

1. Use 2,016, not 2,018, as the denominator for the strict energy-class condition comparison.
2. Distinguish the analytic uniform random baseline from the older repository files in which `M0` denotes majority-class prediction.
3. Do not promote the type and period error decomposition, operating point sensitivity, or sample composition interventions to the main Chapter 4 comparison.
4. Keep dataset description, Experimental Design, Results, and Discussion functionally separate.
5. Introduce every figure and table before it appears, then report the result pattern, and reserve explanation and implications for Discussion.
