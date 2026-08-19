"""Fig. 4.2 -- within-city composition, reference pool vs experimental sample.

Each bar is normalized within its own city (sums to 100%), so Delft (n = 173)
is directly comparable to Amsterdam (n = 8,011) -- absolute counts would render
the small cities as invisible slivers.

Layout, following the side-by-side pool/sample convention:
    row 1: building type    -- (a) reference pool | (b) experimental sample
    row 2: TABULA-NL period -- (c) reference pool | (d) experimental sample
    row 3: EPC label        -- (e) reference pool | (f) experimental sample

Run:  .venv/Scripts/python.exe scripts/fig_ch4_2_city_composition.py
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

CITY_DISPLAY = {"amsterdam": "Amsterdam", "rotterdam": "Rotterdam",
                "utrecht": "Utrecht", "delft": "Delft"}
PERIOD_RANGES = {
    "NL.01": "≤1964", "NL.02": "1965–1974", "NL.03": "1975–1991",
    "NL.04": "1992–2005", "NL.05": "2006–2014", "NL.06": "≥2015",
}
TYPE_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
PERIOD_COLORS = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(PERIOD_LABELS)))
EPC_LABELS = ["A", "B", "C", "D", "E", "F", "G"]
EPC_COLORS = plt.get_cmap("RdYlGn_r")(np.linspace(0.05, 0.95, len(EPC_LABELS)))


def load_sets() -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = pd.read_parquet(REPO / "data/processed/stage1_gt.parquet")
    manifest = pd.read_parquet(REPO / "data/processed/svi_manifest.parquet")
    pool["pand_id"] = pool["pand_id"].astype(str)
    manifest["pand_id"] = manifest["pand_id"].astype(str)
    pool["epc"] = pool["Energieklasse"].map(_merge_epc)
    sample = pool[pool["pand_id"].isin(set(manifest["pand_id"]))].copy()
    return sample, pool


def _merge_epc(label: str) -> str:
    """A+, A++, A+++, A++++ all collapse into A; B..G pass through."""
    return "A" if str(label).startswith("A") else str(label)


def _within_city_pct(df: pd.DataFrame, col: str, order: list[str]) -> pd.DataFrame:
    return (
        df.groupby("city")[col].value_counts(normalize=True)
        .unstack()
        .reindex(index=CITY_LABELS)
        .reindex(columns=order)
        .fillna(0.0) * 100
    )


def _panel(ax, df, col, order, colors, legend_labels, title, show_ylabel):
    pct = _within_city_pct(df, col, order)
    counts = df["city"].value_counts().reindex(CITY_LABELS)
    x = np.arange(len(CITY_LABELS))
    bottom = np.zeros(len(CITY_LABELS))

    for cat, color, lab in zip(order, colors, legend_labels):
        vals = pct[cat].to_numpy()
        ax.bar(x, vals, 0.62, bottom=bottom, color=color, label=lab,
               edgecolor="white", linewidth=0.5)
        bottom += vals

    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{CITY_DISPLAY[c]}\nn = {counts[c]:,}" for c in CITY_LABELS])
    if show_ylabel:
        ax.set_ylabel("% within city")
    ax.yaxis.grid(True, color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    return pct.round(1)


def main() -> None:
    setup_mpl()
    sample, pool = load_sets()

    fig, axes = plt.subplots(3, 2, figsize=(12, 13.5))

    type_legend = TYPE_LABELS
    period_legend = [f"{p}  ({PERIOD_RANGES[p]})" for p in PERIOD_LABELS]
    epc_legend = EPC_LABELS

    specs = [
        (axes[0, 0], pool, "building_type", TYPE_LABELS, TYPE_COLORS, type_legend,
         "(a) Building type — reference pool", True),
        (axes[0, 1], sample, "building_type", TYPE_LABELS, TYPE_COLORS, type_legend,
         "(b) Building type — experimental sample", False),
        (axes[1, 0], pool, "tabula_period", PERIOD_LABELS, PERIOD_COLORS, period_legend,
         "(c) Construction period — reference pool", True),
        (axes[1, 1], sample, "tabula_period", PERIOD_LABELS, PERIOD_COLORS, period_legend,
         "(d) Construction period — experimental sample", False),
        (axes[2, 0], pool, "epc", EPC_LABELS, EPC_COLORS, epc_legend,
         "(e) EPC label — reference pool", True),
        (axes[2, 1], sample, "epc", EPC_LABELS, EPC_COLORS, epc_legend,
         "(f) EPC label — experimental sample", False),
    ]
    tables = [(spec[2], spec[6], _panel(*spec)) for spec in specs]
    for col, title, t in tables:
        print(f"\n--- {title}")
        print(t.to_string())

    axes[0, 1].legend(title="Building type", loc="center left",
                      bbox_to_anchor=(1.02, 0.5), frameon=False)
    axes[1, 1].legend(title="TABULA-NL period", loc="center left",
                      bbox_to_anchor=(1.02, 0.5), frameon=False)
    axes[2, 1].legend(title="EPC label", loc="center left",
                      bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"F4_2_city_composition.{ext}"
        fig.savefig(out)
        print(f"\n[fig] {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
