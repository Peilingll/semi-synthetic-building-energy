"""5-fold StratifiedGroupKFold split for Stage 1.

Restricts to buildings that have at least one image in svi_manifest, then
stratifies by Energieklasse with pand_id as group. Output one row per
training-eligible pand_id with assigned fold index 0-4.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)

N_SPLITS = 5
RANDOM_STATE = 42


def make_fold_indices(gt: pd.DataFrame, manifest: pd.DataFrame, city: str) -> pd.DataFrame:
    city_manifest = manifest[manifest["city"] == city]
    pand_ids_with_imgs = set(city_manifest["pand_id"].unique())

    eligible = gt[gt["pand_id"].isin(pand_ids_with_imgs)].reset_index(drop=True)
    logger.info("eligible (gt ∩ manifest): %d / %d gt rows", len(eligible), len(gt))

    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    eligible["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(skf.split(eligible, eligible["Energieklasse"], groups=eligible["pand_id"])):
        eligible.loc[val_idx, "fold"] = fold_idx

    assert (eligible["fold"] >= 0).all(), "some rows did not get assigned a fold"

    out = eligible[["pand_id", "fold"]].copy()

    for fold_idx in range(N_SPLITS):
        val_mask = eligible["fold"] == fold_idx
        n_val = val_mask.sum()
        types = eligible.loc[val_mask, "building_type"].value_counts().to_dict()
        logger.info("fold %d: val=%d  types=%s", fold_idx, n_val, types)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, choices=["delft", "utrecht", "rotterdam", "amsterdam"])
    parser.add_argument("--gt", default=None, help="Path to stage1_gt_<city>.parquet")
    parser.add_argument("--manifest", default=None, help="Path to svi_manifest.parquet")
    parser.add_argument("--out", default=None, help="Output path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    repo_root = Path(__file__).resolve().parents[2]
    gt_path = Path(args.gt) if args.gt else repo_root / "data" / "processed" / f"stage1_gt_{args.city}.parquet"
    manifest_path = Path(args.manifest) if args.manifest else repo_root / "data" / "processed" / "svi_manifest.parquet"
    out_path = Path(args.out) if args.out else repo_root / "data" / "processed" / f"fold_indices_{args.city}.parquet"

    gt = pd.read_parquet(gt_path)
    manifest = pd.read_parquet(manifest_path)
    folds = make_fold_indices(gt, manifest, args.city)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_parquet(out_path, index=False)
    logger.info("wrote %s (%d rows)", out_path, len(folds))


if __name__ == "__main__":
    main()
