"""Fig. 4.5 -- pooled vs LOCO slope chart (OpenFACADES Fig. 9 template).

Small-multiple slope panels showing the cost of moving from the pooled
four-city setting to the Amsterdam-held-out (LOCO) setting:
  top row    -- upstream DINOv2: year MAE doubles while type accuracy is
                essentially unchanged;
  bottom row -- downstream binary macro-F1 of the three routes converges
                to the random-guessing expectation (dashed line).

All values are taken verbatim from the finalised tables in
doc_processed/Thesis/05-2026.08.21_obj_conclusion_argument.md:
  Table 4.2 (pooled upstream), Table 4.5 (pooled downstream),
  Table 4.7 (LOCO upstream), Table 4.8 (LOCO downstream),
  type accuracy 0.901 -> 0.897 from the Objective 3 discussion (§4.2.5).

Run:  .venv/Scripts/python.exe scripts/fig_ch4_5_pooled_vs_loco.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "notebooks"))
from _stage1_plot import setup_mpl  # noqa: E402

OUT_DIR = REPO / "reports" / "figures" / "ch4"

BLUE = "#4C78A8"    # DINOv2 / M2, matches Tables 4.2-4.4 figure palette
ORANGE = "#F58518"  # M3
GREY = "#555555"    # M1 (reference attributes, no vision component)
BASE = "#999999"    # M0 random-guessing baseline
INK = "#333333"

# (label, pooled, LOCO, note, color, ylim, note_dy) -- sources in docstring
UPSTREAM = [
    ("Construction-year MAE (yr)", 9.45, 19.50, "+10.05 yr (×2.1)", BLUE,
     (0, 24), 26),
    ("Building-type accuracy", 0.901, 0.897, "−0.004", BLUE, (0.5, 1.0), -22),
]
DOWNSTREAM = [
    ("M1 · Reference attributes\n(no vision component)", 0.492, 0.496, "+0.004", GREY),
    ("M2 · Direct image-to-EPC\n(DINOv2)", 0.597, 0.499, "−0.098", BLUE),
    ("M3 · Attribute-mediated\n(DINOv2)", 0.490, 0.490, "±0.000", ORANGE),
]
M0_POOLED, M0_LOCO = 0.479, 0.480  # Tables 4.5 / 4.8
F1_LIM = (0.44, 0.66)


def slope(ax, v0: float, v1: float, color: str, fmt: str, note: str,
          note_dy: float) -> None:
    ax.plot([0, 1], [v0, v1], color=color, lw=2, marker="o", ms=6, zorder=3)
    ax.text(-0.08, v0, fmt.format(v0), ha="right", va="center", fontsize=9,
            color=INK)
    ax.text(1.08, v1, fmt.format(v1), ha="left", va="center", fontsize=9,
            color=INK)
    ax.annotate(note, xy=(0.5, (v0 + v1) / 2), xytext=(0, note_dy),
                textcoords="offset points", ha="center", fontsize=9,
                color=color)


def style(ax, ylim: tuple[float, float]) -> None:
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 1], ["Pooled\n(4 cities)", "LOCO\n(Amsterdam)"])


def main() -> None:
    setup_mpl()
    fig = plt.figure(figsize=(10.5, 6.6))
    gs = fig.add_gridspec(2, 6, hspace=0.75, wspace=0.35,
                          top=0.86, bottom=0.08, left=0.05, right=0.97)

    # -- top row: upstream DINOv2 -----------------------------------------
    for k, (title, v0, v1, note, color, ylim, note_dy) in enumerate(UPSTREAM):
        ax = fig.add_subplot(gs[0, 3 * k:3 * k + 3])
        fmt = "{:.2f}" if v1 > 1 else "{:.3f}"
        slope(ax, v0, v1, color, fmt, note, note_dy=note_dy)
        style(ax, ylim)
        ax.set_title(title)

    # -- bottom row: downstream binary macro-F1 ---------------------------
    for k, (title, v0, v1, note, color) in enumerate(DOWNSTREAM):
        ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
        ax.plot([0, 1], [M0_POOLED, M0_LOCO], color=BASE, lw=1.2, ls="--",
                zorder=2)
        if k == 0:
            ax.text(0.5, M0_POOLED - 0.012, "M0 random guessing (0.48)",
                    ha="center", va="top", fontsize=8, color=BASE)
        note_dy = 10 if abs(v1 - v0) < 0.02 else -18
        slope(ax, v0, v1, color, "{:.3f}", note, note_dy=note_dy)
        style(ax, F1_LIM)
        ax.set_title(title)

    fig.text(0.5, 0.945, "Upstream attribute extraction (DINOv2)",
             ha="center", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.5, 0.455, "Downstream EPC classification — binary macro-F1",
             ha="center", fontsize=11, fontweight="bold", color=INK)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"F4_5_pooled_vs_loco.{ext}"
        fig.savefig(out)
        print(f"[fig] {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
