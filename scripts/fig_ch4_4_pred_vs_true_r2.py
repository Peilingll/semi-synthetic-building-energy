"""Fig. 4.3 -- predicted vs true scatter for year and floor regression (hold-out).

Two rows x three columns:
  row 1: construction year, with identity diagonal + TABULA-NL period boundaries
  row 2: floor count, with identity diagonal
  cols:  DINOv2 (frozen), ResNet-50 (fine-tuned), InternVL3-2B (zero-shot)
Each panel is annotated with the hold-out R^2 and MAE (same evaluated subset as
Table 4.2; values match reports/stage1/r2_holdout.json).

Run:  .venv/Scripts/python.exe scripts/fig_ch4_3_pred_vs_true_r2.py
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

MODELS = {
    "DINOv2 (frozen)": "reports/stage1/dinov2_frozen/holdout_preds.parquet",
    "ResNet-50 (fine-tuned)": "reports/stage1/resnet50_ft/holdout_preds.parquet",
    "InternVL3-2B (zero-shot)": "reports/stage1/vlm_internvl3/v3_holdout_per_pand_id.parquet",
}

PERIOD_BOUNDS = [1964.5, 1974.5, 1991.5, 2005.5, 2014.5]
POINT_COLOR = "#4C78A8"


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def _panel(ax, y_true, y_pred, lims, title=None, boundaries=None, jitter=0.0):
    rng = np.random.default_rng(42)
    xt, yp = y_true.astype(float), y_pred.astype(float)
    if jitter:
        xt = xt + rng.uniform(-jitter, jitter, len(xt))
    if boundaries:
        for b in boundaries:
            ax.axvline(b, color="#DDDDDD", lw=0.6, zorder=0)
            ax.axhline(b, color="#DDDDDD", lw=0.6, zorder=0)
    ax.plot(lims, lims, ls="--", color="#888888", lw=1.0, zorder=1)
    ax.scatter(xt, yp, s=6, alpha=0.25, color=POINT_COLOR, edgecolors="none", zorder=2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    val_r2 = r2(y_true.astype(float), y_pred.astype(float))
    mae = float(np.abs(y_true - y_pred).mean())
    fmt = f"R² = {val_r2:.3f}\nMAE = {mae:.2f}" if mae >= 1 else f"R² = {val_r2:.3f}\nMAE = {mae:.3f}"
    ax.text(0.03, 0.97, fmt, transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.9))
    if title:
        ax.set_title(title)


def main() -> None:
    setup_mpl()
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.4))

    for col, (name, path) in enumerate(MODELS.items()):
        df = pd.read_parquet(REPO / path)

        year = df[["pred_year", "true_bouwjaar"]].dropna()
        _panel(axes[0, col], year["true_bouwjaar"].to_numpy(), year["pred_year"].to_numpy(),
               lims=(1795, 2035), title=name, boundaries=PERIOD_BOUNDS)
        axes[0, col].set_xlabel("Reference construction year (BAG)")
        if col == 0:
            axes[0, col].set_ylabel("Predicted construction year")

        floors = df[["pred_floors", "true_num_floors"]].dropna()
        _panel(axes[1, col], floors["true_num_floors"].to_numpy(), floors["pred_floors"].to_numpy(),
               lims=(0, 15), jitter=0.15)
        axes[1, col].set_xlabel("Reference floor count (3DBAG)")
        if col == 0:
            axes[1, col].set_ylabel("Predicted floor count")

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"F4_3_pred_vs_true_r2.{ext}"
        fig.savefig(out)
        print(f"[fig] {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
