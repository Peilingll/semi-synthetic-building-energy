"""Leave-One-City-Out (LOCO) splits for Stage 1/3 (RQ3b).

Hold-out = ALL buildings of one city (default Amsterdam).
Dev = every other city's buildings with images (the pooled dev/holdout
boundary inside the remaining cities is dissolved on purpose: with the
whole target city held out, that boundary no longer serves a purpose and
the source pool is scarce), split into 5 folds with the same
StratifiedGroupKFold protocol as the pooled splits.

Outputs mirror splits.py filenames inside data/processed/loco_<city>/ so
train.py / eval_holdout.py only need path overrides, no schema changes.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.stage1.splits import N_SPLITS, RANDOM_STATE, build_strata, compute_checksum

logger = logging.getLogger(__name__)


def split_loco(
    gt: pd.DataFrame,
    manifest: pd.DataFrame,
    holdout_city: str,
    dev_n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pids_with_imgs = set(manifest["pand_id"].astype(str).unique())
    eligible = gt[gt["pand_id"].astype(str).isin(pids_with_imgs)].copy().reset_index(drop=True)
    logger.info("training universe (gt INTERSECT manifest): %d / %d gt rows", len(eligible), len(gt))

    cities = set(eligible["city"].unique())
    assert holdout_city in cities, f"holdout city {holdout_city!r} not in {sorted(cities)}"

    holdout_df = eligible[eligible["city"] == holdout_city].copy().reset_index(drop=True)
    dev_df = eligible[eligible["city"] != holdout_city].copy().reset_index(drop=True)

    assert set(holdout_df["pand_id"]).isdisjoint(set(dev_df["pand_id"])), \
        "hold-out and dev share pand_ids — group leak!"
    assert (holdout_df["city"] == holdout_city).all()
    assert holdout_city not in set(dev_df["city"])

    logger.info("LOCO hold-out (%s): %d buildings (%.1f%%)", holdout_city,
                len(holdout_df), 100 * len(holdout_df) / len(eligible))
    logger.info("LOCO dev (%s): %d buildings", sorted(set(dev_df["city"])), len(dev_df))

    dev_df["strata"] = build_strata(dev_df)
    logger.info("dev unique strata: %d", dev_df["strata"].nunique())

    sgkf_dev = StratifiedGroupKFold(
        n_splits=dev_n_splits, shuffle=True, random_state=random_state,
    )
    dev_df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(sgkf_dev.split(
        dev_df, dev_df["strata"], groups=dev_df["pand_id"],
    )):
        dev_df.iloc[val_idx, dev_df.columns.get_loc("fold")] = fold_idx
    assert (dev_df["fold"] >= 0).all(), "some dev rows did not get a fold"

    for f in range(dev_n_splits):
        val_mask = dev_df["fold"] == f
        city_dist = dev_df.loc[val_mask, "city"].value_counts().to_dict()
        type_dist = dev_df.loc[val_mask, "building_type"].value_counts().to_dict()
        logger.info("dev fold %d: val=%d cities=%s types=%s", f, val_mask.sum(), city_dist, type_dist)

    return holdout_df, dev_df


def write_outputs(
    holdout_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    out_dir: Path,
    holdout_city: str,
    random_state: int,
    n_splits: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    holdout_out = holdout_df[["pand_id", "city", "building_type", "Energieklasse", "tabula_period"]].copy()
    holdout_out["split"] = f"loco_{holdout_city}_holdout"
    holdout_path = out_dir / "holdout_test_pand_ids.parquet"
    holdout_out.to_parquet(holdout_path, index=False)
    logger.info("wrote %s (%d rows)", holdout_path, len(holdout_out))

    checksum = compute_checksum(holdout_out["pand_id"].astype(str).tolist())
    checksum_path = out_dir / "holdout_test_pand_ids.checksum.txt"
    checksum_path.write_text(
        f"sha256_prefix: {checksum}\n"
        f"n_buildings:   {len(holdout_out)}\n"
        f"holdout_city:  {holdout_city}\n"
        f"random_state:  {random_state}\n"
        f"n_splits:      {n_splits}\n"
        f"strata_keys:   city,Gebouwtype,Energieklasse,tabula_period (dev folds only)\n"
        f"created_with:  src/stage1/loco_splits.py\n",
        encoding="utf-8",
    )
    logger.info("hold-out SHA256 prefix: %s (n=%d)", checksum, len(holdout_out))

    dev_out = dev_df[["pand_id", "fold", "city", "building_type", "Energieklasse", "tabula_period"]].copy()
    dev_path = out_dir / "dev_fold_indices.parquet"
    dev_out.to_parquet(dev_path, index=False)
    logger.info("wrote %s (%d rows, %d folds)", dev_path, len(dev_out), dev_out["fold"].nunique())

    return {
        "holdout_path": holdout_path,
        "dev_path": dev_path,
        "checksum": checksum,
        "n_holdout": len(holdout_out),
        "n_dev": len(dev_out),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-city", default="amsterdam")
    parser.add_argument("--gt", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    repo_root = Path(__file__).resolve().parents[2]
    gt_path = Path(args.gt) if args.gt else repo_root / "data" / "processed" / "stage1_gt.parquet"
    manifest_path = Path(args.manifest) if args.manifest else (
        repo_root / "data" / "processed" / "svi_manifest.parquet"
    )
    out_dir = Path(args.out_dir) if args.out_dir else (
        repo_root / "data" / "processed" / f"loco_{args.holdout_city}"
    )

    gt = pd.read_parquet(gt_path)
    manifest = pd.read_parquet(manifest_path)

    holdout_df, dev_df = split_loco(
        gt, manifest,
        holdout_city=args.holdout_city,
        dev_n_splits=args.n_splits,
        random_state=args.seed,
    )
    info = write_outputs(holdout_df, dev_df, out_dir, args.holdout_city, args.seed, args.n_splits)
    logger.info("done. hold-out %d  dev %d  checksum %s",
                info["n_holdout"], info["n_dev"], info["checksum"])


if __name__ == "__main__":
    main()
