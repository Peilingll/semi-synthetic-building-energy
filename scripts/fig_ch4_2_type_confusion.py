"""Fig. 4.2 -- building-type confusion matrices (hold-out), three vision configurations.

One row-normalized 4x4 confusion matrix per configuration, fixed order
DINOv2 -> ResNet-50 -> InternVL3-2B (same as Tables 4.2/4.3/4.6). Each cell is
annotated with the row share and the building count. Supports the accuracy vs
macro-F1 discussion in section 4.2.1 (e.g. the MFH row is entirely off-diagonal).

Run:  .venv/Scripts/python.exe scripts/fig_ch4_2_type_confusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "notebooks"))
from _stage1_plot import setup_mpl  # noqa: E402

OUT_DIR = REPO / "reports" / "figures" / "ch4"
TYPES = ["SFH", "TH", "MFH", "AB"]

MODELS = {
    "DINOv2 (frozen)": "reports/stage1/dinov2_frozen/holdout_preds.parquet",
    "ResNet-50 (fine-tuned)": "reports/stage1/resnet50_ft/holdout_preds.parquet",
    "InternVL3-2B (zero-shot)": "reports/stage1/vlm_internvl3/v3_holdout_per_pand_id.parquet",
}


def main() -> None:
    setup_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))

    for ax, (name, path) in zip(axes, MODELS.items()):
        df = pd.read_parquet(REPO / path)[["true_type", "pred_type"]].dropna()
        cm = np.zeros((4, 4), dtype=int)
        for i, t in enumerate(TYPES):
            for j, p in enumerate(TYPES):
                cm[i, j] = int(((df["true_type"] == t) & (df["pred_type"] == p)).sum())
        row = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

        ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                color = "white" if row[i, j] > 0.6 else "#1F3B57"
                ax.text(j, i, f"{row[i, j]:.2f}\n(n={cm[i, j]})", ha="center",
                        va="center", fontsize=8, color=color)
        ax.set_xticks(range(4), TYPES)
        ax.set_yticks(range(4), TYPES)
        ax.set_xlabel("Predicted type")
        if ax is axes[0]:
            ax.set_ylabel("Reference type (EP-Online)")
        ax.set_title(name)
        ax.spines[:].set_visible(False)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"F4_2_type_confusion.{ext}"
        fig.savefig(out)
        print(f"[fig] {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
