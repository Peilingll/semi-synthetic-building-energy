"""Fig. 4.3b -- Sankey flows from reference to predicted TABULA-NL construction period (hold-out).

One panel per vision configuration (DINOv2 -> ResNet-50 -> InternVL3-2B), stacked
vertically so the figure prints at text width with legible labels. Left nodes are
the reference periods (from BAG construction year), right nodes the predicted
periods (from the predicted year). Ribbon width is the number of hold-out
buildings; correctly assigned flows are TUM blue, misassigned flows TUM orange.

Data: the same hold-out prediction files as fig_ch4_3_cell_recall_heatmap.py.
Output (new file, the recall heatmap is untouched):
  reports/figures/ch4/F4_3_period_sankey.png / .svg

Run:  .venv/Scripts/python.exe scripts/fig_ch4_3_period_sankey.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MPath

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.tabula_matcher import classify_period  # noqa: E402

OUT_DIR = REPO / "reports" / "figures" / "ch4"
PERIODS = ["NL.01", "NL.02", "NL.03", "NL.04", "NL.05", "NL.06"]
LABELS = ["≤1964", "1965–74", "1975–91", "1992–2005", "2006–14", "≥2015"]
MODELS = {
    "DINOv2 (frozen)": "reports/stage1/dinov2_frozen/holdout_preds.parquet",
    "ResNet-50 (fine-tuned)": "reports/stage1/resnet50_ft/holdout_preds.parquet",
    "InternVL3-2B (zero-shot)": "reports/stage1/vlm_internvl3/v3_holdout_per_pand_id.parquet",
}
BLUE, ORANGE, INK, NODE = "#0065BD", "#E37222", "#1F2A37", "#4B5563"
FONT = 11


def flows(path: Path) -> np.ndarray:
    df = pd.read_parquet(path).dropna(subset=["true_bouwjaar", "pred_year"])
    tp = df["true_bouwjaar"].map(lambda y: classify_period(int(y)))
    pp = df["pred_year"].map(lambda y: classify_period(int(round(y))))
    m = np.zeros((6, 6), dtype=int)
    for a, b in zip(tp, pp):
        if a in PERIODS and b in PERIODS:
            m[PERIODS.index(a), PERIODS.index(b)] += 1
    return m


def ribbon(ax, x0, y0a, y0b, x1, y1a, y1b, color, alpha):
    """Filled band from left segment [y0a, y0b] to right segment [y1a, y1b]."""
    cx = (x0 + x1) / 2
    verts = [(x0, y0a), (cx, y0a), (cx, y1a), (x1, y1a),
             (x1, y1b), (cx, y1b), (cx, y0b), (x0, y0b), (x0, y0a)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def draw_panel(ax, m: np.ndarray, title: str):
    total = m.sum()
    gap = 0.05 * total  # vertical gap between nodes, in building units
    left_tot, right_tot = m.sum(axis=1), m.sum(axis=0)
    # node spans (top-down), only periods with buildings on that side get height > 0
    def spans(tots):
        y, out = 0.0, []
        for t in tots:
            out.append((y, y + t))
            y += t + (gap if t > 0 else gap * 0.35)
        return out, y
    lspan, lh = spans(left_tot)
    rspan, rh = spans(right_tot)
    H = max(lh, rh)
    x0, x1, w = 0.0, 1.0, 0.035
    ax.set_xlim(-0.55, 1.55)
    ax.axis("off")
    # ribbons: correct first (blue), then misassigned (orange) on top
    lcur = [s[0] for s in lspan]
    rcur = [s[0] for s in rspan]
    order = [(i, j) for i in range(6) for j in range(6)]
    for correct in (True, False):
        for i, j in order:
            c = m[i, j]
            if c == 0 or ((i == j) != correct):
                continue
            ya, yb = lcur[i], lcur[i] + c
            yc, yd = rcur[j], rcur[j] + c
            lcur[i] += c
            rcur[j] += c
            ribbon(ax, x0 + w, ya, yb, x1 - w, yc, yd,
                   BLUE if correct else ORANGE, 0.55 if correct else 0.75)
    # nodes and labels (labels are staggered so that thin nodes do not overlap)
    def staggered(spans, tots):
        centres = [(a + b) / 2 for a, b in spans]
        min_gap = 0.055 * H
        ys = []
        for k, c in enumerate(centres):
            y = c if not ys else max(c, ys[-1] + min_gap)
            ys.append(y)
        return ys
    ly, ry = staggered(lspan, left_tot), staggered(rspan, right_tot)
    for k, ((a, b), (c, d)) in enumerate(zip(lspan, rspan)):
        if left_tot[k]:
            ax.add_patch(Rectangle((x0, a), w, b - a, facecolor=NODE, edgecolor="none"))
            ax.plot([x0 - 0.01, x0 - 0.05], [(a + b) / 2, ly[k]], color="#9CA3AF", lw=0.8)
            ax.text(x0 - 0.06, ly[k], f"{LABELS[k]}  ({left_tot[k]})",
                    ha="right", va="center", fontsize=FONT, color=INK)
        if right_tot[k]:
            ax.add_patch(Rectangle((x1 - w, c), w, d - c, facecolor=NODE, edgecolor="none"))
            ax.plot([x1 + 0.01, x1 + 0.05], [(c + d) / 2, ry[k]], color="#9CA3AF", lw=0.8)
            ax.text(x1 + 0.06, ry[k], f"({right_tot[k]})  {LABELS[k]}",
                    ha="left", va="center", fontsize=FONT, color=INK)
    ax.set_ylim(max(H, max(ly), max(ry)) + 0.03 * H, -gap * 0.5)
    ax.text(x0, -gap * 0.35, "Reference period", ha="left", va="bottom",
            fontsize=FONT, fontweight="bold", color=INK)
    ax.text(x1, -gap * 0.35, "Predicted period", ha="right", va="bottom",
            fontsize=FONT, fontweight="bold", color=INK)
    acc = np.trace(m) / total
    ax.set_title(f"{title}   —   period agreement {acc:.2f}", fontsize=FONT + 1,
                 fontweight="bold", color=INK, loc="left", pad=14)


def main():
    plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": FONT})
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 10.2))
    for ax, (name, path) in zip(axes, MODELS.items()):
        draw_panel(ax, flows(REPO / path), name)
    # shared legend
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=BLUE, alpha=0.55, label="same period as reference"),
                        Patch(color=ORANGE, alpha=0.75, label="different period")],
               loc="lower center", ncol=2, frameon=False, fontsize=FONT,
               bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "F4_3_period_sankey.png", dpi=300, facecolor="white")
    fig.savefig(OUT_DIR / "F4_3_period_sankey.svg", facecolor="white")
    print("saved", OUT_DIR / "F4_3_period_sankey.png")


if __name__ == "__main__":
    main()
