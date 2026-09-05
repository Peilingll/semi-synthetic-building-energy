"""Fig. 4.3 -- per-cell recall heatmap of the TABULA-NL matrix (hold-out).

For each of the 24 archetype cells (4 size classes x 6 periods), recall =
share of hold-out buildings whose REFERENCE cell is that cell and whose
predicted cell (predicted type + period of predicted year) matches it exactly.
One panel per vision configuration, fixed order DINOv2 -> ResNet-50 ->
InternVL3-2B. Cells with no hold-out buildings are shown hatched grey.

Run:  .venv/Scripts/python.exe scripts/fig_ch4_3_cell_recall_heatmap.py
      add --vertical for F4_3_cell_recall_heatmap_vertical.{png,pdf}
      (y = construction period, x = size class)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "notebooks"))
sys.path.insert(0, str(REPO))
from _stage1_plot import setup_mpl  # noqa: E402
from src.tabula_matcher import classify_period  # noqa: E402

OUT_DIR = REPO / "reports" / "figures" / "ch4"
TYPES = ["SFH", "TH", "MFH", "AB"]
PERIODS = ["NL.01", "NL.02", "NL.03", "NL.04", "NL.05", "NL.06"]
PERIOD_RANGES = ["≤1964", "65–74", "75–91", "92–05", "06–14", "≥2015"]
PERIOD_RANGES_FULL = ["≤1964", "1965–1974", "1975–1991", "1992–2005", "2006–2014", "≥2015"]

MODELS = {
    "DINOv2 (frozen)": "reports/stage1/dinov2_frozen/holdout_preds.parquet",
    "ResNet-50 (fine-tuned)": "reports/stage1/resnet50_ft/holdout_preds.parquet",
    "InternVL3-2B (zero-shot)": "reports/stage1/vlm_internvl3/v3_holdout_per_pand_id.parquet",
}


def main(vertical: bool = False) -> None:
    setup_mpl()
    if vertical:
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 5.6))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))

    for ax, (name, path) in zip(axes, MODELS.items()):
        df = pd.read_parquet(REPO / path)
        df = df.dropna(subset=["true_type", "pred_type", "true_bouwjaar", "pred_year"])
        true_p = df["true_bouwjaar"].map(lambda y: classify_period(int(y)))
        pred_p = df["pred_year"].map(lambda y: classify_period(int(round(y))))
        hit = (df["true_type"] == df["pred_type"]) & (true_p == pred_p)

        recall = np.full((4, 6), np.nan)
        count = np.zeros((4, 6), dtype=int)
        for i, t in enumerate(TYPES):
            for j, p in enumerate(PERIODS):
                mask = (df["true_type"] == t) & (true_p == p)
                count[i, j] = int(mask.sum())
                if count[i, j]:
                    recall[i, j] = float(hit[mask].mean())

        if vertical:  # rows = periods, cols = size classes
            recall, count = recall.T, count.T

        masked = np.ma.masked_invalid(recall)
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad("#EBEBEB")
        ax.imshow(masked, cmap=cmap, vmin=0, vmax=1)
        n_row, n_col = recall.shape
        for i in range(n_row):
            for j in range(n_col):
                if count[i, j]:
                    color = "white" if recall[i, j] > 0.6 else "#1F3B57"
                    ax.text(j, i, f"{recall[i, j]:.2f}\n(n={count[i, j]})",
                            ha="center", va="center", fontsize=7, color=color)
                else:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=8,
                            color="#999999")
        if vertical:
            ax.set_xticks(range(4), TYPES)
            ax.set_yticks(range(6), PERIOD_RANGES_FULL, fontsize=8)
            ax.set_xlabel("Size class")
            if ax is axes[0]:
                ax.set_ylabel("Construction period")
        else:
            ax.set_xticks(range(6), PERIOD_RANGES, fontsize=8)
            ax.set_yticks(range(4), TYPES)
            ax.set_xlabel("Construction period")
            if ax is axes[0]:
                ax.set_ylabel("Size class")
        ax.set_title(name)
        ax.spines[:].set_visible(False)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "F4_3_cell_recall_heatmap" + ("_vertical" if vertical else "")
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(out)
        print(f"[fig] {out.relative_to(REPO)}")


if __name__ == "__main__":
    main(vertical="--vertical" in sys.argv)
