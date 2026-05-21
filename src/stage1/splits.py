"""Stage 1 Phase B splits: 20% hold-out test + 80% dev 5-fold CV.

Two-stage StratifiedGroupKFold:
1. Holdout: SGKF(n_splits=5).split() first iteration -- val side is the
   20% hold-out test set, frozen and SHA256-hashed.
2. Dev folds: another SGKF(n_splits=5) inside the remaining 80%, producing
   per-pand_id fold indices 0-4.

Stratify keys = city x Gebouwtype x Energieklasse x tabula_period
(union of Stage 1 target Gebouwtype and Stage 2/3 target Energieklasse so
the same hold-out works for all three stages).

Group = pand_id so all images of one building stay on one side.
"""

import argparse
import hashlib
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_SPLITS = 5


def build_strata(df: pd.DataFrame) -> pd.Series:
    return (
        df["city"].astype(str) + "_"
        + df["Gebouwtype"].astype(str) + "_"
        + df["Energieklasse"].astype(str) + "_"
        + df["tabula_period"].astype(str)
    )


def split_holdout_and_dev(
    gt: pd.DataFrame,
    manifest: pd.DataFrame,
    holdout_n_splits: int = N_SPLITS,
    dev_n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pids_with_imgs = set(manifest["pand_id"].astype(str).unique())
    eligible = gt[gt["pand_id"].astype(str).isin(pids_with_imgs)].copy().reset_index(drop=True)
    logger.info("training universe (gt INTERSECT manifest): %d / %d gt rows", len(eligible), len(gt))

    eligible["strata"] = build_strata(eligible)
    logger.info("unique strata: %d", eligible["strata"].nunique())

    sgkf_holdout = StratifiedGroupKFold(
        n_splits=holdout_n_splits, shuffle=True, random_state=random_state,
    )
    train_idx, holdout_idx = next(sgkf_holdout.split(
        eligible, eligible["strata"], groups=eligible["pand_id"],
    ))
    holdout_df = eligible.iloc[holdout_idx].copy().reset_index(drop=True)
    dev_df = eligible.iloc[train_idx].copy().reset_index(drop=True)

    assert set(holdout_df["pand_id"]).isdisjoint(set(dev_df["pand_id"])), \
        "hold-out and dev share pand_ids — group leak!"

    logger.info("hold-out test: %d buildings (%.1f%%)", len(holdout_df),
                100 * len(holdout_df) / len(eligible))
    logger.info("dev set: %d buildings (%.1f%%)", len(dev_df),
                100 * len(dev_df) / len(eligible))

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
        n_val = val_mask.sum()
        city_dist = dev_df.loc[val_mask, "city"].value_counts().to_dict()
        type_dist = dev_df.loc[val_mask, "building_type"].value_counts().to_dict()
        logger.info("dev fold %d: val=%d cities=%s types=%s", f, n_val, city_dist, type_dist)

    return holdout_df, dev_df


def compute_checksum(pand_ids: list[str]) -> str:
    payload = ",".join(sorted(str(p) for p in pand_ids))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def write_outputs(
    holdout_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    holdout_out = holdout_df[["pand_id", "city", "building_type", "Energieklasse", "tabula_period"]].copy()
    holdout_out["split"] = "holdout_test"
    holdout_path = out_dir / "holdout_test_pand_ids.parquet"
    holdout_out.to_parquet(holdout_path, index=False)
    logger.info("wrote %s (%d rows)", holdout_path, len(holdout_out))

    checksum = compute_checksum(holdout_out["pand_id"].astype(str).tolist())
    checksum_path = out_dir / "holdout_test_pand_ids.checksum.txt"
    checksum_path.write_text(
        f"sha256_prefix: {checksum}\n"
        f"n_buildings:   {len(holdout_out)}\n"
        f"random_state:  {RANDOM_STATE}\n"
        f"n_splits:      {N_SPLITS}\n"
        f"strata_keys:   city,Gebouwtype,Energieklasse,tabula_period\n"
        f"created_with:  src/stage1/splits.py\n",
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
        "checksum_path": checksum_path,
        "checksum": checksum,
        "n_holdout": len(holdout_out),
        "n_dev": len(dev_out),
    }


def main():
    parser = argparse.ArgumentParser()
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
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "data" / "processed"

    gt = pd.read_parquet(gt_path)
    manifest = pd.read_parquet(manifest_path)

    holdout_df, dev_df = split_holdout_and_dev(
        gt, manifest,
        holdout_n_splits=args.n_splits, dev_n_splits=args.n_splits,
        random_state=args.seed,
    )
    info = write_outputs(holdout_df, dev_df, out_dir)
    logger.info("done. hold-out %d  dev %d  checksum %s",
                info["n_holdout"], info["n_dev"], info["checksum"])


if __name__ == "__main__":
    main()
