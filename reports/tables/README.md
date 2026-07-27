# Results index

Every table in this tree, the script that regenerates it, and its one-line result.
Nothing here should be hand-edited — if a number is wrong, fix the script and re-run.

Environment: `.venv/Scripts/python.exe -m <module>` for ETL / CPU work;
conda `stage1-gpu` for anything that trains a vision model.

## Stage 1 — can street view read building attributes?

| table | script | result |
|---|---|---|
| `stage1/T1_per_city_train_holdout` | `notebooks/figs_stage1_dataset.py` | 10,086 buildings / 47,150 images, dev 8,068 + hold-out 2,018 |
| `stage1/T1_*_cv_per_fold` | `notebooks/figs_stage1_resnet.py`, `figs_stage1_dataset.py` | 5-fold CV curves per backbone |
| `stage1/T2_*_holdout_headline` | `src/stage1/eval_holdout.py` | DINOv2 type 0.901 / period 0.901 / year MAE 9.45 |
| `stage1/T3_model_comparison` | `notebooks/figs_stage1_comparison.py` | DINOv2 ≈ ResNet-50 >> InternVL3 zero-shot |
| `stage1/T4_joint_cell` | `src/stage1/joint_cell_eval.py` | joint cell 0.825 vs majority-cell 0.790; macro-cell recall 0.18 |
| `stage1/T5_loco_pool_composition` | `src/stage1/loco_pool_table.py` | LOCO-Amsterdam pool composition |
| `stage1/*_loco_amsterdam` | `src/stage1/loco_compare.py` | LOCO variants of T3/T4 |

## Stage 2 — are those attributes worth anything? (ground-truth inputs, no images)

| table | script | result |
|---|---|---|
| `stage2/T2a_cumulative` | `src/stage2/run_ablation.py` | S_full macro-F1 0.1804, κ 0.1865, acc 0.3488 |
| `stage2/T2b_leave_one_out` | `src/stage2/run_ablation.py` | per-feature leave-one-out from S_full |
| `stage2/T2_per_class_s_full` | `src/stage2/run_ablation.py` | per-class breakdown |
| `stage2/T2d_label_entropy` | `src/stage2/label_entropy.py` | 58% of pands multi-cert, 77% of those disagree; oracle 0.79 |
| `stage2/T2e_m1plus_fullstock` | `src/stage2/m1_plus_fullstock.py` | Tier-C full-stock upper envelope |

## Stage 3 — join the two halves and compare routes

| table | script | result |
|---|---|---|
| `stage3/T3_main` | `src/stage3/run_stage3.py` | M0 0.068 / M1 0.172 / M3-DINOv2 0.150 / M2-DINOv2 0.213 |
| `stage3/T3_error_propagation` | `src/stage3/run_stage3.py` | where M3 loses relative to M1 |
| `stage3/T3_ordinal_collapse` | `src/stage3/ordinal_collapse.py` | ordinal metrics + literature-aligned collapses |
| `stage3/T3reg_regression_vs_classification` | `src/stage3/regression_kwh.py` | kWh regression vs direct classification |
| `stage3/T3_full_comparison` | `notebooks/figs_stage3_routes.py` | all routes, one table |
| `stage3/T6_degradation_pooled_vs_loco` | **no script yet** | pooled vs LOCO-Amsterdam degradation |
| `stage3/T7_htr_instrument` | `src/stage3/htr_instrument.py` | the only downstream readout that resolves Stage-1 quality: 3.2x separation vs 1.09x on the EPC label |

## Audit (0.x) — does the data mean what the experiments assume?

See `audit/README.md`. Headline: `compactheid`/`floor_area` were certificate
columns (leakage); the label bins `PrimaireFossieleEnergieEMGForfaitair`, not the
plain column; the TABULA lookup carried a uniform −0.26 m²K/W error; and no
downstream energy metric can resolve Stage-1 quality.

## Known gaps

- `stage3/T6_degradation_pooled_vs_loco.md` has no producing script (numbers were
  computed interactively). Needs `src/stage1/loco_degradation.py`.
- The 2026.06.26 geometry ablation was also interactive; it is superseded by
  `src/audit/a01_compactheid_source.py` (T6 section), which uses clean features.
