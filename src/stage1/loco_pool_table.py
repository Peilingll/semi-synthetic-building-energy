"""LOCO companion outputs:

1. reports/tables/stage1/T5_loco_pool_composition.md — train/test pool
   composition (type x pool, period x pool, joint-cell coverage) for the
   LOCO split, cited by §3.6.1.
2. data/processed/loco_<city>/holdout_vlm_subset.parquet — LOCO hold-out
   INTERSECT pooled hold-out: the strictly comparable evaluation pool for
   zero-shot paradigms whose predictions only exist on the pooled hold-out.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
PROCESSED = REPO / "data" / "processed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-city", default="amsterdam")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    loco_dir = PROCESSED / f"loco_{args.holdout_city}"
    dev = pd.read_parquet(loco_dir / "dev_fold_indices.parquet")
    ho = pd.read_parquet(loco_dir / "holdout_test_pand_ids.parquet")
    pooled_ho = pd.read_parquet(PROCESSED / "holdout_test_pand_ids.parquet")

    # --- strictly comparable subset for zero-shot paradigms ---
    subset = ho[ho["pand_id"].astype(str).isin(set(pooled_ho["pand_id"].astype(str)))].copy()
    subset_path = loco_dir / "holdout_vlm_subset.parquet"
    subset.to_parquet(subset_path, index=False)
    logger.info("wrote %s (%d buildings = LOCO holdout INTERSECT pooled holdout)",
                subset_path, len(subset))

    # --- pool composition table ---
    dev = dev.assign(pool="train (R+U+D)")
    ho = ho.assign(pool=f"test ({args.holdout_city})")
    both = pd.concat([dev[["pand_id", "pool", "building_type", "tabula_period"]],
                      ho[["pand_id", "pool", "building_type", "tabula_period"]]]).reset_index(drop=True)
    both["cell"] = both["building_type"].astype(str) + "|" + both["tabula_period"].astype(str)

    def md_table(t: pd.DataFrame, index_name: str) -> list[str]:
        cols = list(t.columns)
        out = ["| " + " | ".join([index_name] + [str(c) for c in cols]) + " |",
               "|---" + "|---:" * len(cols) + "|"]
        def fmt(v):
            f = float(v)
            return str(int(f)) if f.is_integer() else str(v)
        for idx, row in t.iterrows():
            out.append("| " + " | ".join([str(idx)] + [fmt(v) for v in row]) + " |")
        return out

    def dist(col):
        t = pd.crosstab(both[col], both["pool"])
        pct = {c: (100 * t[c] / t[c].sum()).round(1) for c in t.columns}
        for c, v in pct.items():
            t[f"{c} %"] = v
        return t

    type_t, period_t = dist("building_type"), dist("tabula_period")
    cell_t = pd.crosstab(both["cell"], both["pool"]).sort_values(
        f"test ({args.holdout_city})", ascending=False)
    test_col = f"test ({args.holdout_city})"
    uncovered = cell_t[(cell_t[test_col] > 0) & (cell_t["train (R+U+D)"] == 0)]

    lines = [
        f"# Table 5 — LOCO-{args.holdout_city} pool composition",
        "",
        f"Train = all imaged R+U+D buildings (n={len(dev)}, pooled dev+holdout merged); "
        f"test = all imaged {args.holdout_city} buildings (n={len(ho)}).",
        f"Strictly comparable zero-shot subset (test INTERSECT pooled hold-out): n={len(subset)}.",
        "",
        "## Size class x pool", "",
        *md_table(type_t, "building_type"),
        "",
        "## TABULA period x pool", "",
        *md_table(period_t, "tabula_period"),
        "",
        "## Joint-cell coverage (sorted by test support)", "",
        *md_table(cell_t, "cell"),
        "",
        f"Cells present in test but absent from train: "
        f"{', '.join(uncovered.index) if len(uncovered) else 'none'}"
        f"{f' ({int(uncovered[test_col].sum())} test buildings)' if len(uncovered) else ''}.",
    ]
    out = REPO / "reports" / "tables" / "stage1" / "T5_loco_pool_composition.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
