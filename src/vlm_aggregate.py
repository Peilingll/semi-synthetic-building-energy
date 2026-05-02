"""Aggregate per-image VLM predictions into per-pand_id rows.

Inputs:
- `data/processed/vlm_predictions_openfacades.csv` (per-image, OpenFACADES schema)
- `data/openfacades_output/phase_c_delft_grid/merged/bag_openfacades_id_mapping.csv`
  (`building_id` → `pand_id` from spatial join)

Output:
- `data/processed/vlm_predictions_full.csv`
  one row per pand_id with majority-vote categoricals + median numerics + image-count metadata
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = REPO_ROOT / "data" / "processed" / "vlm_predictions_openfacades.csv"
DEFAULT_MAPPING = (
    REPO_ROOT
    / "data" / "openfacades_output" / "phase_c_delft_grid" / "merged"
    / "bag_openfacades_id_mapping.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "vlm_predictions_full.csv"

CATEGORICAL = ["building_type", "surface_material", "construction_material"]
NUMERIC = ["building_age", "floors"]


def majority_vote(s: pd.Series) -> tuple[str | None, float]:
    """Return (top label, share). Empty / all-NaN → (None, 0.0)."""
    s = s.dropna().astype(str)
    if len(s) == 0:
        return None, 0.0
    counts = s.value_counts()
    top_label = counts.index[0]
    share = float(counts.iloc[0]) / float(len(s))
    return top_label, share


def aggregate(predictions_csv: Path, mapping_csv: Path, output_csv: Path) -> pd.DataFrame:
    pred = pd.read_csv(predictions_csv, dtype={"building_id": str})
    mapping = pd.read_csv(mapping_csv, dtype={"building_id": str, "pand_id": str})
    logger.info("predictions: %d rows, %d unique building_id", len(pred), pred["building_id"].nunique())
    logger.info("mapping:     %d rows, %d unique pand_id (residential subset)",
                len(mapping), mapping["pand_id"].nunique(dropna=True))

    # Drop predictions that errored out
    ok = pred[pred["error"].isna() | (pred["error"] == "")].copy()
    logger.info("ok predictions: %d / %d", len(ok), len(pred))

    # Coerce numeric columns
    for col in NUMERIC:
        ok[col] = pd.to_numeric(ok[col], errors="coerce")

    # Merge on building_id → bring pand_id in
    merged = ok.merge(mapping[["building_id", "pand_id"]], on="building_id", how="left")
    n_no_pand = merged["pand_id"].isna().sum()
    if n_no_pand:
        logger.warning("%d image rows have no BAG pand_id (will be dropped)", n_no_pand)
    merged = merged.dropna(subset=["pand_id"])

    rows: list[dict] = []
    for pand_id, grp in merged.groupby("pand_id"):
        out = {"pand_id": str(pand_id), "n_images_used": len(grp)}
        # Categorical: majority vote + share
        for col in CATEGORICAL:
            label, share = majority_vote(grp[col])
            out[col] = label
            out[f"{col}_vote_share"] = round(share, 3)
        # Numeric: median (robust to one-off bad year guesses)
        for col in NUMERIC:
            vals = grp[col].dropna()
            out[col] = float(vals.median()) if len(vals) else None
        # Schema-aligned aliases for MVP compatibility
        out["construction_year"] = out.get("building_age")
        out["num_floors"] = out.get("floors")
        out["window_to_wall_ratio"] = None  # not in OpenFACADES training
        rows.append(out)

    agg = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    # Save zero-padded pand_id as string to keep leading zeros
    agg["pand_id"] = agg["pand_id"].astype(str).str.zfill(16)
    agg.to_csv(output_csv, index=False)
    logger.info("Wrote %d aggregated rows to %s", len(agg), output_csv)
    logger.info("Top building_type vote distribution:")
    logger.info(agg["building_type"].value_counts().head(10).to_string())
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    aggregate(args.predictions, args.mapping, args.output)


if __name__ == "__main__":
    main()
