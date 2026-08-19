"""Fig. 4.1 -- experimental sample vs eligible four-city registry reference pool.

Normalized (within-set percentage) distribution comparison, five panels:
  (a) city                   (b) building type       (c) floor-count buckets
  (d) TABULA-NL periods      (e) EPC labels, A+..A++++ merged into A
The sixth cell carries the shared legend.

Reference pool = all rows of stage1_gt.parquet (four cities, EPC present,
TABULA-NL matching succeeded; no image requirement), n = 124,784.
Experimental sample = pool INTERSECT svi_manifest.parquet, n = 10,086.
The sample is a subset of the pool, so the two bars are "my sample" vs
"the whole eligible stock", not two disjoint groups.

Run:  .venv/Scripts/python.exe scripts/fig_ch4_1_sample_vs_pool.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "notebooks"))
from _stage1_plot import setup_mpl, CITY_LABELS, PERIOD_LABELS, TYPE_LABELS  # noqa: E402

OUT_DIR = REPO / "reports" / "figures" / "ch4"

SAMPLE_COLOR = "#4C78A8"
POOL_COLOR = "#BAB0AC"
SAMPLE_NAME = "Experimental sample"
POOL_NAME = "Reference pool"

CITY_DISPLAY = {"amsterdam": "Amsterdam", "rotterdam": "Rotterdam",
                "utrecht": "Utrecht", "delft": "Delft"}
FLOOR_BUCKETS = ["1", "2", "3", "4", "5", "6", "7", "8+"]
EPC_LABELS = ["A", "B", "C", "D", "E", "F", "G"]
PERIOD_RANGES = {
    "NL.01": "≤1964", "NL.02": "1965–1974", "NL.03": "1975–1991",
    "NL.04": "1992–2005", "NL.05": "2006–2014", "NL.06": "≥2015",
}


def load_sets() -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = pd.read_parquet(REPO / "data/processed/stage1_gt.parquet")
    manifest = pd.read_parquet(REPO / "data/processed/svi_manifest.parquet")
    pool["pand_id"] = pool["pand_id"].astype(str)
    manifest["pand_id"] = manifest["pand_id"].astype(str)

    sample = pool[pool["pand_id"].isin(set(manifest["pand_id"]))].copy()

    pool["floor_bucket"] = pool["num_floors"].map(_floor_bucket)
    pool["epc"] = pool["Energieklasse"].map(_merge_epc)
    sample["floor_bucket"] = sample["num_floors"].map(_floor_bucket)
    sample["epc"] = sample["Energieklasse"].map(_merge_epc)
    return sample, pool


def _floor_bucket(n) -> str:
    n = int(n)
    return str(n) if n <= 7 else "8+"


def _merge_epc(label: str) -> str:
    """A+, A++, A+++, A++++ all collapse into A; B..G pass through."""
    return "A" if str(label).startswith("A") else str(label)


def _within_set_pct(df: pd.DataFrame, col: str, order: list[str]) -> np.ndarray:
    counts = df[col].value_counts()
    total = counts.sum()
    return np.array([counts.get(c, 0) / total * 100 for c in order])


def _panel(ax, sample, pool, col, order, xticklabels, title, show_ylabel):
    s_pct = _within_set_pct(sample, col, order)
    p_pct = _within_set_pct(pool, col, order)
    x = np.arange(len(order))
    w = 0.38

    ax.bar(x - w / 2, s_pct, w, color=SAMPLE_COLOR, label=SAMPLE_NAME)
    ax.bar(x + w / 2, p_pct, w, color=POOL_COLOR, label=POOL_NAME)

    ax.set_title(title)
    if show_ylabel:
        ax.set_ylabel("% of set")
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.yaxis.grid(True, color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    return pd.DataFrame({col: order, "sample_pct": s_pct.round(1),
                         "pool_pct": p_pct.round(1)})


def main() -> None:
    setup_mpl()
    sample, pool = load_sets()
    n_s, n_p = len(sample), len(pool)
    print(f"experimental sample = {n_s:>7,}")
    print(f"reference pool      = {n_p:>7,}  (sample is a subset: "
          f"{n_s / n_p * 100:.1f}%)")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)

    tables = [
        _panel(axes[0, 0], sample, pool, "city", CITY_LABELS,
               [CITY_DISPLAY[c] for c in CITY_LABELS], "(a) City", True),
        _panel(axes[0, 1], sample, pool, "building_type", TYPE_LABELS,
               TYPE_LABELS, "(b) Building type", False),
        _panel(axes[1, 0], sample, pool, "floor_bucket", FLOOR_BUCKETS,
               FLOOR_BUCKETS, "(c) Number of floors", True),
        _panel(axes[1, 1], sample, pool, "tabula_period", PERIOD_LABELS,
               [f"{p}\n{PERIOD_RANGES[p]}" for p in PERIOD_LABELS],
               "(d) TABULA-NL construction period", False),
    ]
    for t in tables:
        print()
        print(t.to_string(index=False))

    handles, _ = axes[0, 0].get_legend_handles_labels()
    labels = [f"{SAMPLE_NAME}  (n = {n_s:,})", f"{POOL_NAME}  (n = {n_p:,})"]
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"F4_1_sample_vs_pool.{ext}"
        fig.savefig(out)
        print(f"\n[fig] {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
