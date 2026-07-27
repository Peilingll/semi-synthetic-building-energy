"""Evaluate aggregated VLM predictions against BAG ground truth.

Compares `data/processed/vlm_predictions_full.csv` (per-pand_id, OpenFACADES schema)
against `data/processed/residential_tabula_matched.parquet`:

| OpenFACADES        | BAG ground truth         | Metric             |
|--------------------|--------------------------|--------------------|
| building_type      | tabula_building_type     | binary AB vs other accuracy |
| construction_year  | bouwjaar                 | MAE, MdAE, ±10y / ±20y |
| num_floors         | num_floors_estimated     | MAE, ±0 / ±1       |
| surface_material   | (no BAG field)           | distribution only  |

Reports per-class precision/recall for AB vs house, plus the share of
non-residential predictions (`retail/office/industrial/...`) which indicates
where the model is confused.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRED = REPO_ROOT / "data" / "processed" / "vlm_predictions_full.csv"
DEFAULT_GT = REPO_ROOT / "data" / "processed" / "residential_tabula_matched.parquet"
DEFAULT_REPORT = REPO_ROOT / "data" / "processed" / "vlm_phase2_eval.json"

# OpenFACADES → BAG TABULA coarse mapping
# OF only has "apartments" / "house" for residential, no fine NL split
OF_RESIDENTIAL = {"apartments", "house"}
OF_NON_RESIDENTIAL = {"retail", "office", "hotel", "industrial", "religious",
                      "education", "public", "garage"}


def of_to_tabula(of_type: str | None) -> str | None:
    if of_type is None or pd.isna(of_type):
        return None
    of_type = str(of_type).lower()
    if of_type == "apartments":
        return "AB"
    if of_type == "house":
        return "house"
    if of_type in OF_NON_RESIDENTIAL:
        return f"non_residential ({of_type})"
    return f"unknown ({of_type})"


def evaluate(pred_csv: Path, gt_parquet: Path, report_path: Path) -> dict:
    pred = pd.read_csv(pred_csv, dtype={"pand_id": str})
    gt = pd.read_parquet(gt_parquet)
    gt["pand_id"] = gt["pand_id"].astype(str).str.zfill(16)
    pred["pand_id"] = pred["pand_id"].astype(str).str.zfill(16)

    n_with_pred = len(pred)
    n_residential = len(gt)
    coverage = n_with_pred / n_residential if n_residential else 0.0
    logger.info("Predictions: %d pand_id | BAG residential total: %d | coverage: %.1f%%",
                n_with_pred, n_residential, 100 * coverage)

    df = pred.merge(gt[["pand_id", "Gebouwtype", "tabula_building_type",
                        "bouwjaar", "num_floors_estimated"]],
                    on="pand_id", how="inner")
    logger.info("Merged eval rows: %d", len(df))

    report: dict = {
        "n_predictions": n_with_pred,
        "n_bag_residential": n_residential,
        "coverage_pct": round(100 * coverage, 2),
        "n_eval": len(df),
    }

    # === 1. Building type coarse classification ===
    df["of_class"] = df["building_type"].apply(of_to_tabula)
    df["gt_class"] = np.where(df["tabula_building_type"] == "AB", "AB", "house")

    # accuracy (only over rows with a residential prediction: apartments or house)
    valid = df["of_class"].isin(["AB", "house"])
    if valid.any():
        correct = (df.loc[valid, "of_class"] == df.loc[valid, "gt_class"]).mean()
        report["type_residential_accuracy"] = round(correct, 4)
        report["type_n_residential_pred"] = int(valid.sum())

        # Confusion matrix (residential predictions only)
        cm = pd.crosstab(df.loc[valid, "gt_class"], df.loc[valid, "of_class"], dropna=False)
        report["type_confusion"] = cm.to_dict()

    # Share of non-residential predictions (model confused / image issue)
    non_res = df["of_class"].astype(str).str.startswith("non_residential")
    report["type_non_residential_share"] = round(non_res.mean(), 4)
    if non_res.any():
        report["type_non_residential_breakdown"] = (
            df.loc[non_res, "building_type"].value_counts().to_dict()
        )

    # === 2. Construction year ===
    df["year_pred"] = pd.to_numeric(df["construction_year"], errors="coerce")
    yv = df.dropna(subset=["year_pred", "bouwjaar"]).copy()
    if len(yv):
        err = (yv["year_pred"] - yv["bouwjaar"]).abs()
        report["year_n"] = len(yv)
        report["year_mae"] = round(err.mean(), 2)
        report["year_mdae"] = round(err.median(), 2)
        report["year_within_10y_pct"] = round(100 * (err <= 10).mean(), 2)
        report["year_within_20y_pct"] = round(100 * (err <= 20).mean(), 2)

    # === 3. Floor count ===
    df["floor_pred"] = pd.to_numeric(df["num_floors"], errors="coerce")
    fv = df.dropna(subset=["floor_pred", "num_floors_estimated"]).copy()
    if len(fv):
        err = (fv["floor_pred"] - fv["num_floors_estimated"]).abs()
        report["floor_n"] = len(fv)
        report["floor_mae"] = round(err.mean(), 2)
        report["floor_exact_pct"] = round(100 * (err == 0).mean(), 2)
        report["floor_within_1_pct"] = round(100 * (err <= 1).mean(), 2)

    # === 4. Surface material distribution (no BAG ground truth) ===
    if "surface_material" in df.columns:
        report["surface_material_distribution"] = (
            df["surface_material"].value_counts(dropna=False).to_dict()
        )

    # === 5. n_images_used distribution ===
    if "n_images_used" in df.columns:
        ni = df["n_images_used"]
        report["n_images_per_pand_id"] = {
            "mean": round(ni.mean(), 2),
            "median": int(ni.median()),
            "min": int(ni.min()),
            "max": int(ni.max()),
        }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Report written to %s", report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = evaluate(args.predictions, args.ground_truth, args.report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
