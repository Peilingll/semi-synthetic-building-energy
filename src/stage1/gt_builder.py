"""Build Stage 1 ground-truth table from BAG + 3D BAG + EP-Online joined parquet.

Output schema (one row per pand_id):
    pand_id (str)            — join key to svi_manifest
    city (str)
    bouwjaar (int)           — year regression target
    building_type (cat4)     — SFH/TH/MFH/AB, via GEBOUWTYPE_MAP
    num_floors (int)         — from b3_bouwlagen (fallback to height/3 if missing)
    Energieklasse (str)      — stratify key only, NOT a training target
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.tabula_matcher import GEBOUWTYPE_MAP

logger = logging.getLogger(__name__)

BOUWJAAR_MIN = 1800
BOUWJAAR_MAX = 2025
FLOOR_HEIGHT_M = 3.0


def build_stage1_gt(joined_path: Path, city: str) -> pd.DataFrame:
    df = pd.read_parquet(joined_path)
    logger.info("loaded %d rows from %s", len(df), joined_path)

    n0 = len(df)
    df = df[df["gebruiksdoel"].fillna("").str.contains("woonfunctie")]
    logger.info("woonfunctie filter: %d → %d", n0, len(df))

    n1 = len(df)
    df = df[df["Energieklasse"].notna() & (df["Energieklasse"].str.strip() != "")]
    logger.info("Energieklasse filter: %d → %d", n1, len(df))

    n2 = len(df)
    df = df[df["Gebouwtype"].isin(GEBOUWTYPE_MAP.keys())]
    logger.info("Gebouwtype filter: %d → %d", n2, len(df))

    n3 = len(df)
    df = df[(df["bouwjaar"] >= BOUWJAAR_MIN) & (df["bouwjaar"] <= BOUWJAAR_MAX)]
    logger.info("bouwjaar [%d, %d] filter: %d → %d", BOUWJAAR_MIN, BOUWJAAR_MAX, n3, len(df))

    bouwlagen = df["b3_bouwlagen"]
    height_fallback = (df["b3_h_50p"] / FLOOR_HEIGHT_M).round()
    num_floors = bouwlagen.fillna(height_fallback)
    n_fallback = bouwlagen.isna().sum()
    if n_fallback > 0:
        logger.info("b3_bouwlagen missing for %d rows, used height/%.1f fallback", n_fallback, FLOOR_HEIGHT_M)

    n4 = len(df)
    valid_floors = num_floors.notna() & (num_floors >= 1) & (num_floors <= 30)
    df = df[valid_floors]
    num_floors = num_floors[valid_floors]
    logger.info("valid num_floors [1, 30] filter: %d → %d", n4, len(df))

    out = pd.DataFrame({
        "pand_id": df["pand_id"].astype(str),
        "city": city,
        "bouwjaar": df["bouwjaar"].astype(int),
        "building_type": df["Gebouwtype"].map(GEBOUWTYPE_MAP).astype("string"),
        "num_floors": num_floors.astype(int).values,
        "Energieklasse": df["Energieklasse"].astype(str),
    })

    out = out.drop_duplicates(subset="pand_id").reset_index(drop=True)
    logger.info("final stage1_gt: %d rows", len(out))
    logger.info("building_type counts:\n%s", out["building_type"].value_counts().to_string())
    logger.info("Energieklasse counts:\n%s", out["Energieklasse"].value_counts().to_string())

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, choices=["delft", "utrecht", "rotterdam", "amsterdam"])
    parser.add_argument("--joined-path", default=None,
                        help="Path to bag_3dbag_ep_joined.parquet (default: data/processed/<city>/bag_3dbag_ep_joined.parquet)")
    parser.add_argument("--out", default=None,
                        help="Output path (default: data/processed/stage1_gt_<city>.parquet)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    repo_root = Path(__file__).resolve().parents[2]
    joined = Path(args.joined_path) if args.joined_path else (
        repo_root / "data" / "processed" / args.city / "bag_3dbag_ep_joined.parquet"
    )
    out_path = Path(args.out) if args.out else (
        repo_root / "data" / "processed" / f"stage1_gt_{args.city}.parquet"
    )

    gt = build_stage1_gt(joined, args.city)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gt.to_parquet(out_path, index=False)
    logger.info("wrote %s (%d rows)", out_path, len(gt))


if __name__ == "__main__":
    main()
