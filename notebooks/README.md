# Notebooks — presentation only

**Nothing here trains a model or computes a result.** These scripts read finished
artifacts out of `reports/` and turn them into the figures and tables that go in
the thesis. If a number is wrong, the fix belongs in `src/`, not here.

| script | reads | writes |
|---|---|---|
| `figs_stage1_dataset.py` | `data/processed/` splits + manifest | `tables/stage1/T1_*`, `figures/stage1/data/F1_label_distributions` |
| `figs_stage1_dinov2.py` | `reports/stage1/dinov2_frozen/` | `figures/stage1/dinov2/F1–F4` |
| `figs_stage1_resnet.py` | `reports/stage1/resnet50_ft/` | `figures/stage1/resnet/F1–F4`, `tables/stage1/T1_resnet50_cv_per_fold` |
| `figs_stage1_vlm.py` | `reports/stage1/vlm_internvl3/` | `figures/stage1/vlm/F1–F2` |
| `figs_stage1_comparison.py` | all three model dirs | `figures/stage1/F_model_comparison`, `tables/stage1/T3_model_comparison` |
| `figs_stage3_routes.py` | `reports/stage3/` | `figures/stage3/F1–F3`, `tables/stage3/T3_full_comparison` |
| `_stage1_plot.py`, `_stage3_plot.py` | — | shared matplotlib style + `save_fig` helpers |

The `.ipynb` files are the interactive twins of the same content; the `.py` files
are what to run for reproducibility.

## Where results actually come from

| | produced by | landing in |
|---|---|---|
| Stage 1 training (DINOv2 / ResNet-50) | `src/stage1/train.py` (GPU, conda `stage1-gpu`) | `reports/stage1/<model>/*_history.json`, `*_val_preds.parquet`; weights in `models/` (gitignored) |
| Stage 1 hold-out evaluation | `src/stage1/eval_holdout.py` | `reports/stage1/<model>/holdout_preds.parquet` |
| Stage 1 VLM zero-shot | `src/stage1/vlm/internvl3_runner.py` + `aggregate.py` | `reports/stage1/vlm_internvl3/` |
| Stage 2 ablations | `src/stage2/run_ablation.py` | `reports/tables/stage2/` |
| Stage 3 routes | `src/stage3/run_stage3.py` and siblings | `reports/stage3/`, `reports/tables/stage3/` |
| Data audits | `src/audit/a0*.py` | `reports/tables/audit/`, `reports/figures/audit/` |

`archived/01_data_exploration.backup.ipynb` is a 66 MB snapshot of the April/May
exploration, kept deliberately and gitignored.
