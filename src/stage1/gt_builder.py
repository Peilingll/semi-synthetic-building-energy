"""Build Stage 1 ground-truth table from BAG + 3D BAG + EP-Online joined parquet.

Output schema (one row per pand_id):
    pand_id (str)            — join key to svi_manifest
    city (str)
    bouwjaar (int)           — year regression target
    Gebouwtype (str)         — raw BAG label, used for Stage 1 stratification
    building_type (cat4)     — SFH/TH/MFH/AB, via GEBOUWTYPE_MAP
    num_floors (int)         — from b3_bouwlagen (fallback to b3_h_50p/3 if missing)
    Energieklasse (str)      — Stage 2/3 stratification key; NOT a Stage 1 target
    tabula_period (str)      — NL.01..NL.06, recomputed from bouwjaar for cross-city
                               consistency (Delft source uses 'build_period' instead).

Multi-city mode (--cities all) concatenates the four per-city GTs into
`data/processed/stage1_gt.parquet` and also writes per-city files for
sanity checking.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.tabula_matcher import GEBOUWTYPE_MAP, classify_period

logger = logging.getLogger(__name__)

BOUWJAAR_MIN = 1800
BOUWJAAR_MAX = 2025
FLOOR_HEIGHT_M = 3.0
CITIES = ["delft", "utrecht", "rotterdam", "amsterdam"]


def build_stage1_gt(joined_path: Path, city: str) -> pd.DataFrame:
    df = pd.read_parquet(joined_path)
    logger.info("[%s] loaded %d rows from %s", city, len(df), joined_path.name)

    n0 = len(df)
    df = df[df["gebruiksdoel"].fillna("").str.contains("woonfunctie")]
    logger.info("[%s] woonfunctie: %d → %d", city, n0, len(df))

    n1 = len(df)
    df = df[df["Energieklasse"].notna() & (df["Energieklasse"].astype(str).str.strip() != "")]
    logger.info("[%s] Energieklasse not null: %d → %d", city, n1, len(df))

    n2 = len(df)
    df = df[df["Gebouwtype"].isin(GEBOUWTYPE_MAP.keys())]
    logger.info("[%s] Gebouwtype in map: %d → %d", city, n2, len(df))

    n3 = len(df)
    df = df[(df["bouwjaar"] >= BOUWJAAR_MIN) & (df["bouwjaar"] <= BOUWJAAR_MAX)]
    logger.info("[%s] bouwjaar [%d, %d]: %d → %d", city, BOUWJAAR_MIN, BOUWJAAR_MAX, n3, len(df))

    bouwlagen = df["b3_bouwlagen"]
    height_fallback = (df["b3_h_50p"] / FLOOR_HEIGHT_M).round()
    num_floors = bouwlagen.fillna(height_fallback)
    n_fallback = bouwlagen.isna().sum()
    if n_fallback > 0:
        logger.info("[%s] b3_bouwlagen missing for %d rows, used height/%.1f fallback",
                    city, n_fallback, FLOOR_HEIGHT_M)

    n4 = len(df)
    valid_floors = num_floors.notna() & (num_floors >= 1) & (num_floors <= 30)
    df = df[valid_floors]
    num_floors = num_floors[valid_floors]
    logger.info("[%s] num_floors [1, 30]: %d → %d", city, n4, len(df))

    tabula_period = df["bouwjaar"].apply(classify_period)
    n_no_period = tabula_period.isna().sum()
    if n_no_period > 0:
        logger.warning("[%s] %d rows with no tabula_period (will be dropped)", city, n_no_period)
        keep = tabula_period.notna()
        df = df[keep]
        num_floors = num_floors[keep]
        tabula_period = tabula_period[keep]

    out = pd.DataFrame({
        "pand_id": df["pand_id"].astype(str).values,
        "city": city,
        "bouwjaar": df["bouwjaar"].astype(int).values,
        "Gebouwtype": df["Gebouwtype"].astype(str).values,
        "building_type": df["Gebouwtype"].map(GEBOUWTYPE_MAP).astype("string").values,
        "num_floors": num_floors.astype(int).values,
        "Energieklasse": df["Energieklasse"].astype(str).values,
        "tabula_period": tabula_period.astype(str).values,
    })

    out = out.drop_duplicates(subset="pand_id").reset_index(drop=True)
    logger.info("[%s] final stage1_gt: %d rows", city, len(out))
    logger.info("[%s] building_type: %s", city, out["building_type"].value_counts().to_dict())
    logger.info("[%s] tabula_period: %s", city, out["tabula_period"].value_counts().to_dict())

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", required=True,
                        help="single city (delft/utrecht/rotterdam/amsterdam) or 'all'")
    parser.add_argument("--joined-path", default=None,
                        help="single-city only: override joined parquet path")
    parser.add_argument("--out", default=None,
                        help="single-city only: override output path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cities == "all":
        all_gts = []
        for city in CITIES:
            joined = repo_root / "data" / "processed" / city / "bag_3dbag_ep_joined.parquet"
            gt = build_stage1_gt(joined, city)
            per_city_out = out_dir / f"stage1_gt_{city}.parquet"
            gt.to_parquet(per_city_out, index=False)
            logger.info("wrote %s (%d rows)", per_city_out, len(gt))
            all_gts.append(gt)
        combined = pd.concat(all_gts, ignore_index=True)
        combined_out = out_dir / "stage1_gt.parquet"
        combined.to_parquet(combined_out, index=False)
        logger.info("wrote %s (%d rows, 4 cities)", combined_out, len(combined))
        logger.info("multi-city building_type: %s", combined["building_type"].value_counts().to_dict())
        logger.info("multi-city Energieklasse: %s", combined["Energieklasse"].value_counts().to_dict())
        logger.info("multi-city tabula_period: %s", combined["tabula_period"].value_counts().to_dict())
    else:
        if args.cities not in CITIES:
            raise SystemExit(f"unknown city {args.cities!r}; must be 'all' or one of {CITIES}")
        joined = Path(args.joined_path) if args.joined_path else (
            repo_root / "data" / "processed" / args.cities / "bag_3dbag_ep_joined.parquet"
        )
        out_path = Path(args.out) if args.out else (out_dir / f"stage1_gt_{args.cities}.parquet")
        gt = build_stage1_gt(joined, args.cities)
        gt.to_parquet(out_path, index=False)
        logger.info("wrote %s (%d rows)", out_path, len(gt))


if __name__ == "__main__":
    main()
