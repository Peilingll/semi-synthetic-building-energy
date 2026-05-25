"""Build notebooks/04_stage1_data.ipynb from declarative cell list."""

import json
from pathlib import Path


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _src(lines),
    }


def _src(lines):
    """Each element is a single 'line' (may contain newlines); join with '\\n' but preserve."""
    body = "\n".join(lines)
    # nbformat expects list of strings each ending in '\n' except possibly the last
    parts = body.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


cells = [
    # 1
    md(
        "# 04 — Stage 1 Data: Training Universe and Label Distributions",
        "",
        "Four Dutch cities (Amsterdam, Rotterdam, Utrecht, Delft). Training universe = SVI manifest ∩ Stage-1 GT.",
        "This notebook documents per-city train/holdout sizes and the two label distributions Stage 1 actually predicts: **building_type** and **tabula_period**.",
    ),
    # 2
    code(
        "import sys",
        "from pathlib import Path",
        "",
        "import pandas as pd",
        "",
        "sys.path.insert(0, str(Path.cwd()))",
        "from _stage1_plot import (",
        "    REPO, FIG_DIR, TABLE_DIR,",
        "    setup_mpl, save_fig, save_table,",
        "    TYPE_LABELS, CITY_LABELS, PERIOD_LABELS,",
        "    TYPE_PALETTE, CITY_PALETTE,",
        ")",
        "",
        "setup_mpl()",
        "",
        "manifest = pd.read_parquet(REPO / 'data/processed/svi_manifest.parquet')",
        "gt       = pd.read_parquet(REPO / 'data/processed/stage1_gt.parquet')",
        "holdout  = pd.read_parquet(REPO / 'data/processed/holdout_test_pand_ids.parquet')",
        "for df in (manifest, gt, holdout):",
        "    df['pand_id'] = df['pand_id'].astype(str)",
        "",
        "universe = sorted(set(manifest['pand_id']) & set(gt['pand_id']))",
        "print(f'manifest images       = {len(manifest):>7,}')",
        "print(f'manifest buildings    = {manifest[\"pand_id\"].nunique():>7,}')",
        "print(f'GT buildings (4 city) = {gt[\"pand_id\"].nunique():>7,}')",
        "print(f'training universe     = {len(universe):>7,}  (manifest ∩ GT)')",
        "print(f'holdout buildings     = {len(holdout):>7,}')",
    ),
    # 3
    md("## 1. Training universe"),
    # 4
    code(
        "uni_set = set(universe)",
        "ho_set  = set(holdout['pand_id'])",
        "manifest_uni = manifest[manifest['pand_id'].isin(uni_set)].copy()",
        "manifest_uni['split'] = manifest_uni['pand_id'].map(",
        "    lambda pid: 'holdout' if pid in ho_set else 'train'",
        ")",
        "",
        "agg = (",
        "    manifest_uni.groupby(['city', 'split'])",
        "    .agg(buildings=('pand_id', 'nunique'),",
        "         images=('pand_id', 'size'))",
        "    .reset_index()",
        "    .pivot(index='city', columns='split', values=['buildings', 'images'])",
        ")",
        "agg.columns = [f'{a}_{b}' for a, b in agg.columns]",
        "agg = agg.reindex(CITY_LABELS).reset_index()",
        "agg = agg[['city', 'buildings_train', 'images_train', 'buildings_holdout', 'images_holdout']]",
        "agg = agg.rename(columns={",
        "    'buildings_train': 'train_buildings',",
        "    'images_train':    'train_images',",
        "    'buildings_holdout': 'holdout_buildings',",
        "    'images_holdout':    'holdout_images',",
        "})",
        "total = agg.drop(columns=['city']).sum().to_dict()",
        "total['city'] = 'TOTAL'",
        "agg_with_total = pd.concat([agg, pd.DataFrame([total])], ignore_index=True)",
        "for c in ['train_buildings', 'train_images', 'holdout_buildings', 'holdout_images']:",
        "    agg_with_total[c] = agg_with_total[c].astype(int)",
        "save_table(agg_with_total, 'T1_per_city_train_holdout')",
        "agg_with_total",
    ),
    # 5
    md("## 2. Label distributions (training universe)"),
    # 6
    code(
        "import matplotlib.pyplot as plt",
        "",
        "uni_gt = gt[gt['pand_id'].isin(uni_set)].copy()",
        "",
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))",
        "",
        "# Left: building_type x city stacked",
        "type_city = (",
        "    uni_gt.groupby(['city', 'building_type']).size()",
        "    .unstack(fill_value=0)",
        "    .reindex(index=CITY_LABELS)",
        "    .reindex(columns=TYPE_LABELS, fill_value=0)",
        ")",
        "type_city.plot(kind='bar', stacked=True, ax=axes[0],",
        "               color=[TYPE_PALETTE[t] for t in TYPE_LABELS], width=0.7)",
        "axes[0].set_title('building_type × city')",
        "axes[0].set_xlabel('')",
        "axes[0].set_ylabel('buildings')",
        "axes[0].tick_params(axis='x', rotation=0)",
        "axes[0].legend(title='type', loc='upper right', frameon=False)",
        "",
        "# Right: tabula_period x city stacked",
        "period_city = (",
        "    uni_gt.groupby(['city', 'tabula_period']).size()",
        "    .unstack(fill_value=0)",
        "    .reindex(index=CITY_LABELS)",
        "    .reindex(columns=PERIOD_LABELS, fill_value=0)",
        ")",
        "period_city.plot(kind='bar', stacked=True, ax=axes[1],",
        "                 colormap='viridis', width=0.7)",
        "axes[1].set_title('tabula_period × city')",
        "axes[1].set_xlabel('')",
        "axes[1].set_ylabel('buildings')",
        "axes[1].tick_params(axis='x', rotation=0)",
        "axes[1].legend(title='period', loc='upper right', frameon=False, ncol=2)",
        "",
        "fig.tight_layout()",
        "save_fig(fig, 'F1_label_distributions', 'data')",
        "plt.show()",
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("notebooks/04_stage1_data.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out, "with", len(cells), "cells")
