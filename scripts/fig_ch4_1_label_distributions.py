"""Fig. 4.1 -- label distributions of the experimental sample, stacked by city.

Same form as the Stage 1 F1 figure (x = city, stacked absolute building counts),
with thesis-facing panel titles and two changes:
  - city names capitalised
  - EPC A+ .. A++++ merged into a single class A, giving A-G

Experimental sample = stage1_gt.parquet INTERSECT svi_manifest.parquet,
n = 10,086 buildings.

Run:  .venv/Scripts/python.exe scripts/fig_ch4_1_label_distributions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "notebooks"))
from _stage1_plot import (  # noqa: E402
    setup_mpl, CITY_LABELS, PERIOD_LABELS, TYPE_LABELS, TYPE_PALETTE,
)

OUT_DIR = REPO / "reports" / "figures" / "ch4"

CITY_DISPLAY = {"amsterdam": "Amsterdam", "rotterdam": "Rotterdam",
                "utrecht": "Utrecht", "delft": "Delft"}
FLOOR_BUCKETS = ["1", "2", "3", "4", "5", "6", "7", "8+"]
EPC_LABELS = ["A", "B", "C", "D", "E", "F", "G"]
PERIOD_RANGES = {
    "NL.01": "≤1964", "NL.02": "1965–1974", "NL.03": "1975–1991",
    "NL.04": "1992–2005", "NL.05": "2006–2014", "NL.06": "≥2015",
}


def load_sample() -> pd.DataFrame:
    gt = pd.read_parquet(REPO / "data/processed/stage1_gt.parquet")
    manifest = pd.read_parquet(REPO / "data/processed/svi_manifest.parquet")
    gt["pand_id"] = gt["pand_id"].astype(str)
    manifest["pand_id"] = manifest["pand_id"].astype(str)

    df = gt[gt["pand_id"].isin(set(manifest["pand_id"]))].copy()
    df["floor_bucket"] = df["num_floors"].map(lambda n: str(int(n)) if int(n) <= 7 else "8+")
    df["epc"] = df["Energieklasse"].map(_merge_epc)
    return df


def _merge_epc(label: str) -> str:
    """A+, A++, A+++, A++++ all collapse into A; B..G pass through."""
    return "A" if str(label).startswith("A") else str(label)


def _panel(ax, df, col, order, title, colors=None, cmap=None, legend_labels=None,
           legend_title="", ncol=1):
    counts = (
        df.groupby(["city", col]).size()
        .unstack(fill_value=0)
        .reindex(index=CITY_LABELS)
        .reindex(columns=order, fill_value=0)
    )
    counts.index = [CITY_DISPLAY[c] for c in counts.index]
    if legend_labels is not None:
        counts.columns = legend_labels

    kwargs = {"color": colors} if colors is not None else {"colormap": cmap}
    counts.plot(kind="bar", stacked=True, ax=ax, width=0.7, **kwargs)

    totals = counts.sum(axis=1)
    for i, total in enumerate(totals):
        ax.text(i, total + totals.max() * 0.015, f"{int(total):,}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, totals.max() * 1.12)

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("buildings")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title=legend_title, loc="upper right", frameon=False, ncol=ncol,
              fontsize=8 if ncol > 1 else 9)
    return counts


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    setup_mpl()
    df = load_sample()
    print(f"experimental sample = {len(df):,} buildings")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    tables = [
        _panel(axes[0, 0], df, "building_type", TYPE_LABELS, "(a) Building type",
               colors=[TYPE_PALETTE[t] for t in TYPE_LABELS], legend_title="type"),
        _panel(axes[0, 1], df, "floor_bucket", FLOOR_BUCKETS, "(b) Number of floors",
               cmap="plasma", legend_title="floors", ncol=2),
        _panel(axes[1, 0], df, "tabula_period", PERIOD_LABELS, "(c) Construction period",
               cmap="viridis",
               legend_labels=[f"{p}  ({PERIOD_RANGES[p]})" for p in PERIOD_LABELS],
               legend_title="period", ncol=2),
        _panel(axes[1, 1], df, "epc", EPC_LABELS, "(d) Energy class",
               cmap="RdYlGn_r", legend_title="label", ncol=2),
    ]
    for t in tables:
        print()
        print(t.to_string())

    fig.suptitle(f"Experimental sample: {len(df):,} buildings", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"F4_1_label_distributions.{ext}"
        fig.savefig(out)
        print(f"\n[fig] {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
