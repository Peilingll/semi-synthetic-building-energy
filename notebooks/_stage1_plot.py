"""Shared plotting / IO helpers for Stage 1 result notebooks (04, 05, 06)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "reports" / "figures" / "stage1"
TABLE_DIR = REPO / "reports" / "tables" / "stage1"
REPORTS_DIR = REPO / "reports" / "stage1"

TYPE_LABELS = ["SFH", "TH", "MFH", "AB"]
CITY_LABELS = ["amsterdam", "rotterdam", "utrecht", "delft"]
PERIOD_LABELS = ["NL.01", "NL.02", "NL.03", "NL.04", "NL.05", "NL.06"]

TYPE_PALETTE = {
    "SFH": "#4C78A8",
    "TH":  "#F58518",
    "MFH": "#54A24B",
    "AB":  "#E45756",
}
CITY_PALETTE = {
    "amsterdam": "#4C78A8",
    "rotterdam": "#F58518",
    "utrecht":   "#54A24B",
    "delft":     "#E45756",
}


def setup_mpl() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_fig(fig, name: str, subdir: str) -> Path:
    out_dir = FIG_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    print(f"[fig] {png.relative_to(REPO)}  +  {pdf.name}")
    return png


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    csv = TABLE_DIR / f"{name}.csv"
    md = TABLE_DIR / f"{name}.md"
    df.to_csv(csv, index=index)
    md.write_text(_df_to_md(df, index=index), encoding="utf-8")
    print(f"[tbl] {csv.relative_to(REPO)}  +  {md.name}")
    return csv


def _df_to_md(df: pd.DataFrame, index: bool = False) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table without tabulate."""
    if index:
        df = df.reset_index()

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}" if abs(v) < 1000 else f"{v:.2f}"
        return str(v)

    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join([header, sep, *rows]) + "\n"


def confusion_heatmap(cm: np.ndarray, labels: list[str], ax, title: str = "") -> None:
    """Row-normalized 4x4 confusion matrix heatmap into ax."""
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if title:
        ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm_norm[i, j]
            n = int(cm[i, j])
            ax.text(j, i, f"{v:.2f}\n(n={n})",
                    ha="center", va="center",
                    color="white" if v > 0.5 else "black",
                    fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _load_metrics(*paths: Path) -> list[dict]:
    out = []
    for p in paths:
        if p.exists():
            out.append(json.loads(Path(p).read_text(encoding="utf-8")))
    return out


def compute_locked_limits() -> dict:
    """Compute shared y-axis limits from DINOv2 + InternVL3 metrics.

    Reads holdout metrics JSONs directly so 05 and 06 produce identical
    limits regardless of execution order. Returns a dict like:
        {"year_scatter": (1850, 2030),
         "year_mae_bar": (0, 32),
         "floors_mae_bar": (0, 0.8),
         "type_acc_bar": (0, 1)}
    """
    candidates = [
        REPORTS_DIR / "dinov2_frozen" / "holdout_metrics.json",
        REPORTS_DIR / "vlm_internvl3" / "v3_holdout_metrics.json",
    ]
    metrics_list = _load_metrics(*candidates)

    year_max = 0.0
    floors_max = 0.0
    for m in metrics_list:
        for cls_stats in m.get("per_class_year_floors", {}).values():
            year_max = max(year_max, float(cls_stats.get("year_mae", 0.0)))
            floors_max = max(floors_max, float(cls_stats.get("floors_mae", 0.0)))

    return {
        "year_scatter": (1850, 2030),
        "year_mae_bar": (0.0, math.ceil(year_max + 1)),
        "floors_mae_bar": (0.0, math.ceil(floors_max * 10) / 10 + 0.1),
        "type_acc_bar": (0.0, 1.0),
    }


def provenance_table(paths: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        p = Path(p)
        if p.exists():
            st = p.stat()
            rows.append({
                "file": str(p.relative_to(REPO)),
                "size_kb": round(st.st_size / 1024, 1),
                "mtime": pd.Timestamp(st.st_mtime, unit="s").strftime("%Y-%m-%d %H:%M"),
            })
        else:
            rows.append({"file": str(p), "size_kb": None, "mtime": "MISSING"})
    return pd.DataFrame(rows)
