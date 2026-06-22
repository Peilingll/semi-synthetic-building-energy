"""Shared plotting / IO helpers for the Stage 3 result notebook (09)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "reports" / "figures" / "stage3"
TABLE_DIR = REPO / "reports" / "tables" / "stage3"
REPORTS_DIR = REPO / "reports" / "stage3"

ENERGY_LABELS = ["A", "B", "C", "D", "E", "F", "G"]

# Route display order and colours (M1 = ceiling grey; M3 = decomposed warm; M2 = end-to-end cool).
ROUTE_COLORS = {
    "M0": "#BAB0AC",
    "M1": "#000000",
    "M3-DINOv2": "#F58518", "M3-ResNet50": "#E45756", "M3-VLMv3": "#B279A2",
    "M2-DINOv2": "#4C78A8", "M2-ResNet50": "#54A24B", "M2-VLM": "#72B7B2",
}


def setup_mpl() -> None:
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def save_fig(fig, name: str, subdir: str = ".") -> Path:
    out_dir = FIG_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    print(f"[fig] {png.relative_to(REPO)}  +  {pdf.name}")
    return png


def _df_to_md(df: pd.DataFrame, index: bool = False) -> str:
    if index:
        df = df.reset_index()

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}" if abs(v) < 1000 else f"{v:.2f}"
        return str(v)
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join([header, sep, *rows]) + "\n"


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    csv = TABLE_DIR / f"{name}.csv"
    md = TABLE_DIR / f"{name}.md"
    df.to_csv(csv, index=index)
    md.write_text(_df_to_md(df, index=index), encoding="utf-8")
    print(f"[tbl] {csv.relative_to(REPO)}  +  {md.name}")
    return csv


def load_metrics(route: str) -> dict:
    return json.loads((REPORTS_DIR / f"{route}_metrics.json").read_text(encoding="utf-8"))


def confusion_heatmap(cm: np.ndarray, labels: list[str], ax, title: str = "") -> None:
    cm = np.asarray(cm, dtype=float)
    rs = cm.sum(axis=1, keepdims=True)
    cmn = np.divide(cm, rs, out=np.zeros_like(cm), where=rs > 0)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    if title:
        ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
