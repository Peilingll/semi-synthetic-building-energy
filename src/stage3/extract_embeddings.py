"""M2: extract frozen DINOv2 per-building embeddings for dev + holdout.

Runs the Stage 1 frozen DINOv2 backbone (no training) over each building's
street-view images, mean-pools over valid images (same aggregation as Stage 1),
and saves one 768-d embedding per building. These feed the M2 end-to-end head.

ENV: conda `stage1-gpu` (torch + CUDA). Deterministic (val) transforms.

Usage:
    python -m src.stage3.extract_embeddings
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.stage1.dataset import Stage1ImageDataset, collate_fn
from src.stage1.models import DINOV2_FEAT_DIM, DINOv2FrozenMLP

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
OUT = REPO / "reports" / "stage3"

BATCH_BUILDINGS = 8   # 8 buildings x <=8 imgs = <=64 fwd imgs; safe on 8GB (no_grad)
N_FOLDS = 5


@torch.no_grad()
def embed_loader(loader: DataLoader, backbone, device) -> pd.DataFrame:
    rows_pid, rows_emb = [], []
    for batch in loader:
        imgs = batch["images"].to(device)          # [B, K, 3, H, W]
        mask = batch["valid_mask"].to(device)      # [B, K]
        B, K, C, H, W = imgs.shape

        flat_mask = mask.reshape(B * K)
        packed = imgs.reshape(B * K, C, H, W)[flat_mask]   # only valid imgs
        feats_packed = backbone(packed)                    # [n_valid, 768]

        feats = feats_packed.new_zeros(B * K, feats_packed.shape[-1])
        feats[flat_mask] = feats_packed
        feats = feats.view(B, K, -1)

        m = mask.unsqueeze(-1).float()
        pooled = (feats * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)   # [B, 768]

        rows_pid.extend(batch["pand_id"])
        rows_emb.append(pooled.cpu().numpy())
        torch.cuda.empty_cache()

    emb = np.concatenate(rows_emb, axis=0)
    df = pd.DataFrame(emb, columns=[f"e{i}" for i in range(emb.shape[1])])
    df.insert(0, "pand_id", rows_pid)
    return df


def make_loader(split: str, fold: int | None) -> DataLoader:
    ds = Stage1ImageDataset(
        manifest_path=PROC / "svi_manifest.parquet",
        gt_path=PROC / "stage1_gt.parquet",
        split=split,
        dev_fold_indices_path=PROC / "dev_fold_indices.parquet",
        holdout_pand_ids_path=PROC / "holdout_test_pand_ids.parquet",
        fold=fold,
    )
    return DataLoader(ds, batch_size=BATCH_BUILDINGS, shuffle=False,
                      num_workers=2, collate_fn=collate_fn)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device=%s feat_dim=%d", device, DINOV2_FEAT_DIM)
    backbone = DINOv2FrozenMLP().backbone.to(device).eval()

    # dev: union of the 5 val folds (covers all 8,068 with deterministic transforms)
    dev_parts = []
    for f in range(N_FOLDS):
        df = embed_loader(make_loader("val", f), backbone, device)
        dev_parts.append(df)
        logger.info("dev fold %d embedded: %d buildings", f, len(df))
    dev = pd.concat(dev_parts, ignore_index=True).drop_duplicates("pand_id")
    dev.to_parquet(OUT / "embeddings_dev.parquet", index=False)
    logger.info("wrote embeddings_dev.parquet: %d buildings", len(dev))

    ho = embed_loader(make_loader("holdout", None), backbone, device)
    ho.to_parquet(OUT / "embeddings_holdout.parquet", index=False)
    logger.info("wrote embeddings_holdout.parquet: %d buildings", len(ho))


if __name__ == "__main__":
    main()
