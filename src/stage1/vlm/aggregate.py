"""Aggregate per-image VLM predictions into one row per pand_id.

For each building:
- pred_type: mode across the top-3 images, with
  vote_share = mode_count / n_parsed as a confidence indicator.
- pred_year: median (numeric central tendency).
- pred_floors: rounded median (kept as int so the metric matches the
  DINOv2 path's `floors_exact_pct` calculation, per the plan's Floors
  rounding decision).
- pred_period: derived from the median year via
  src.tabula_matcher.classify_period (prompt v3 no longer asks the VLM
  for a period — the same year->period mapping is applied to predictions
  and GT, per the spec's period-accuracy definition).

Schema aligns with `src/stage1/evaluate.py::evaluate_predictions` so the
output parquet can be evaluated with no further changes.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.tabula_matcher import classify_period

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
GT_PATH = REPO_ROOT / "data" / "processed" / "stage1_gt.parquet"


def _mode_with_share(values: pd.Series) -> tuple[str | None, float]:
    cleaned = values.dropna()
    if cleaned.empty:
        return None, 0.0
    vc = cleaned.value_counts()
    top = vc.index[0]
    share = float(vc.iloc[0]) / float(len(cleaned))
    return str(top), round(share, 3)


def _median_or_nan(values: pd.Series) -> float:
    cleaned = values.dropna()
    if cleaned.empty:
        return float("nan")
    return float(np.median(cleaned))


def aggregate(per_image: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    parsed = per_image[per_image["parse_ok"]].copy()
    logger.info(
        "per-image rows: %d total, %d parse_ok (%.1f%%)",
        len(per_image), len(parsed), 100 * len(parsed) / max(1, len(per_image)),
    )

    grouped = per_image.groupby("pand_id", sort=False)
    rows: list[dict] = []
    for pand_id, g in grouped:
        g_parsed = g[g["parse_ok"]]
        n_images = len(g)
        n_parsed = len(g_parsed)
        city = g["city"].iloc[0]

        pred_type, type_vs = _mode_with_share(g_parsed["pred_type"])
        pred_year_median = _median_or_nan(g_parsed["pred_year"])
        pred_floors_median = _median_or_nan(g_parsed["pred_floors"])

        pred_floors_int = int(round(pred_floors_median)) if not np.isnan(pred_floors_median) else None
        pred_period = (
            classify_period(int(round(pred_year_median)))
            if not np.isnan(pred_year_median) else None
        )

        rows.append({
            "pand_id": str(pand_id),
            "city": city,
            "n_images": int(n_images),
            "n_parsed": int(n_parsed),
            "pred_type": pred_type,
            "pred_type_vote_share": type_vs,
            "pred_year": float(pred_year_median) if not np.isnan(pred_year_median) else None,
            "pred_period": pred_period,
            "pred_floors": pred_floors_int,
            "pred_floors_median_raw": float(pred_floors_median) if not np.isnan(pred_floors_median) else None,
        })

    agg = pd.DataFrame(rows)
    n_unparsed = int((agg["n_parsed"] == 0).sum())
    logger.info("aggregated %d buildings (%d with zero parsed images)", len(agg), n_unparsed)

    gt = gt.copy()
    gt["pand_id"] = gt["pand_id"].astype(str)
    merged = agg.merge(
        gt[["pand_id", "building_type", "bouwjaar", "num_floors",
            "Energieklasse", "tabula_period"]],
        on="pand_id", how="left",
    )
    merged = merged.rename(columns={
        "building_type": "true_type",
        "bouwjaar": "true_bouwjaar",
        "num_floors": "true_num_floors",
        "tabula_period": "true_tabula_period",
    })
    n_no_gt = merged["true_type"].isna().sum()
    if n_no_gt:
        logger.warning("%d buildings have no GT row (will be dropped)", n_no_gt)
        merged = merged[merged["true_type"].notna()].reset_index(drop=True)

    merged["true_bouwjaar"] = merged["true_bouwjaar"].astype(float)
    merged["true_num_floors"] = merged["true_num_floors"].astype(float)

    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gt", default=GT_PATH, type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    per_image = pd.read_parquet(args.inp)
    gt = pd.read_parquet(args.gt)
    out = aggregate(per_image, gt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    logger.info("wrote %s (%d rows)", args.out, len(out))

    if "pred_type" in out.columns:
        share_with_pred = (out["n_parsed"] > 0).mean()
        logger.info("buildings with >=1 parsed image: %.3f", share_with_pred)
        logger.info("type vote_share mean: %.3f", out["pred_type_vote_share"].mean())


if __name__ == "__main__":
    main()
