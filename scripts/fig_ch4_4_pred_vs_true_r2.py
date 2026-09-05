"""Fig. 4.3 -- predicted vs true scatter for year and floor regression (hold-out).

Two rows x three columns:
  row 1: construction year, with identity diagonal + TABULA-NL period boundaries
  row 2: floor count, with identity diagonal
  cols:  DINOv2 (frozen), ResNet-50 (fine-tuned), InternVL3-2B (zero-shot)
Each panel is annotated with the hold-out R^2 and MAE (same evaluated subset as
Table 4.2; values match reports/stage1/r2_holdout.json).

Run:  .venv/Scripts/python.exe scripts/fig_ch4_4_pred_vs_true_r2.py
      add --by-city for F4_4_pred_vs_true_r2_by_city.{png,pdf}:
      portrait layout, rows = models, cols = (year, floor count),
      points coloured by city, one shared legend.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "notebooks"))
from _stage1_plot import setup_mpl, CITY_LABELS, CITY_PALETTE  # noqa: E402

OUT_DIR = REPO / "reports" / "figures" / "ch4"

MODELS = {
    "DINOv2 (frozen)": "reports/stage1/dinov2_frozen/holdout_preds.parquet",
    "ResNet-50 (fine-tuned)": "reports/stage1/resnet50_ft/holdout_preds.parquet",
    "InternVL3-2B (zero-shot)": "reports/stage1/vlm_internvl3/v3_holdout_per_pand_id.parquet",
}

PERIOD_BOUNDS = [1964.5, 1974.5, 1991.5, 2005.5, 2014.5]
POINT_COLOR = "#4C78A8"
CITY_DISPLAY = {"amsterdam": "Amsterdam", "rotterdam": "Rotterdam",
                "utrecht": "Utrecht", "delft": "Delft"}


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def _panel(ax, y_true, y_pred, lims, title=None, boundaries=None, jitter=0.0,
           city=None, font=9, ms=6):
    rng = np.random.default_rng(42)
    xt, yp = y_true.astype(float), y_pred.astype(float)
    if jitter:
        xt = xt + rng.uniform(-jitter, jitter, len(xt))
    k = font / 9  # line-width scale relative to the default 9 pt layout
    if boundaries:
        for b in boundaries:
            ax.axvline(b, color="#DDDDDD", lw=0.6 * k, zorder=0)
            ax.axhline(b, color="#DDDDDD", lw=0.6 * k, zorder=0)
    ax.plot(lims, lims, ls="--", color="#888888", lw=1.0 * k, zorder=1)
    if city is None:
        ax.scatter(xt, yp, s=ms, alpha=0.25, color=POINT_COLOR, edgecolors="none", zorder=2)
    else:
        # largest city first so the smaller ones stay visible on top
        for c in CITY_LABELS:
            m = (city == c)
            ax.scatter(xt[m], yp[m], s=ms, alpha=0.45, color=CITY_PALETTE[c],
                       edgecolors="none", zorder=2, label=CITY_DISPLAY[c])
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    val_r2 = r2(y_true.astype(float), y_pred.astype(float))
    mae = float(np.abs(y_true - y_pred).mean())
    fmt = f"R² = {val_r2:.3f}\nMAE = {mae:.2f}" if mae >= 1 else f"R² = {val_r2:.3f}\nMAE = {mae:.3f}"
    ax.text(0.03, 0.97, fmt, transform=ax.transAxes, va="top", ha="left", fontsize=font,
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


FONT = 30  # uniform font size for the by-city variant; canvas enlarged to match


def main_by_city() -> None:
    """Portrait variant: rows = models, cols = (year, floor count), colour = city."""
    setup_mpl()
    plt.rcParams.update({k: FONT for k in (
        "font.size", "axes.titlesize", "axes.labelsize",
        "xtick.labelsize", "ytick.labelsize", "legend.fontsize")})
    fig, axes = plt.subplots(3, 2, figsize=(20, 29))

    for row, (name, path) in enumerate(MODELS.items()):
        df = pd.read_parquet(REPO / path)
        short = name.split(" (")[0]  # drop the parenthetical suffix

        year = df[["pred_year", "true_bouwjaar", "city"]].dropna()
        ax = axes[row, 0]
        _panel(ax, year["true_bouwjaar"].to_numpy(), year["pred_year"].to_numpy(),
               lims=(1795, 2035), boundaries=PERIOD_BOUNDS,
               city=year["city"].to_numpy(), font=FONT, ms=45)
        ax.set_ylabel("Predicted construction year")
        ax.set_xlabel("Reference construction year")
        ax.text(-0.30, 0.5, short, transform=ax.transAxes, rotation=90,
                ha="center", va="center", fontsize=FONT, fontweight="bold")

        floors = df[["pred_floors", "true_num_floors", "city"]].dropna()
        ax = axes[row, 1]
        _panel(ax, floors["true_num_floors"].to_numpy(), floors["pred_floors"].to_numpy(),
               lims=(0, 15), jitter=0.15, city=floors["city"].to_numpy(),
               font=FONT, ms=45)
        ax.set_ylabel("Predicted floor count")
        ax.set_xlabel("Reference floor count")

    axes[0, 0].set_title("Construction year")
    axes[0, 1].set_title("Floor count")

    # one shared legend, city names only
    handles = [plt.Line2D([], [], marker="o", ls="", color=CITY_PALETTE[c],
                          markersize=18, label=CITY_DISPLAY[c])
               for c in CITY_LABELS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.005))

    fig.tight_layout(rect=(0.04, 0.03, 1, 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"F4_4_pred_vs_true_r2_by_city.{ext}"
        fig.savefig(out)
        print(f"[fig] {out.relative_to(REPO)}")


if __name__ == "__main__":
    if "--by-city" in sys.argv:
        main_by_city()
    else:
        main()
