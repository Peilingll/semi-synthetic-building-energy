"""Streaming PIL dataset for Stage 1 Phase B.

Schema (one __getitem__ returns one building):
- Reads svi_manifest.parquet (4 cities), stage1_gt.parquet (4 cities concat),
  and either dev_fold_indices.parquet (split='train'/'val') or
  holdout_test_pand_ids.parquet (split='holdout').
- Pads to MAX_IMAGES with zeros + valid_mask so the model can mask-pool.
- Transforms follow spec section 3.2.4: RandomResizedCrop + HorizontalFlip +
  ColorJitter for train; deterministic Resize/CenterCrop for val/holdout.
"""

import logging
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

MAX_IMAGES = 8
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BUILDING_TYPE_TO_IDX = {"SFH": 0, "TH": 1, "MFH": 2, "AB": 3}
IDX_TO_BUILDING_TYPE = {v: k for k, v in BUILDING_TYPE_TO_IDX.items()}


def build_transforms(split: Literal["train", "val", "holdout"]) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(IMG_SIZE + 32),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class Stage1ImageDataset(Dataset):
    """Stage 1 Phase B four-city dataset."""

    def __init__(
        self,
        manifest_path: Path,
        gt_path: Path,
        split: Literal["train", "val", "holdout"],
        dev_fold_indices_path: Path | None = None,
        holdout_pand_ids_path: Path | None = None,
        fold: int | None = None,
        year_mean: float | None = None,
        year_std: float | None = None,
        floors_mean: float | None = None,
        floors_std: float | None = None,
    ):
        self.split = split
        self.transform = build_transforms(split)

        manifest = pd.read_parquet(manifest_path)
        gt = pd.read_parquet(gt_path)

        if split == "holdout":
            assert holdout_pand_ids_path is not None, "holdout split needs holdout_pand_ids_path"
            holdout = pd.read_parquet(holdout_pand_ids_path)
            keep_pids = set(holdout["pand_id"].astype(str))
        else:
            assert dev_fold_indices_path is not None and fold is not None, \
                "train/val splits need dev_fold_indices_path and fold"
            dev = pd.read_parquet(dev_fold_indices_path)
            assert "fold" in dev.columns
            if split == "train":
                keep_pids = set(dev.loc[dev["fold"] != fold, "pand_id"].astype(str))
            else:
                keep_pids = set(dev.loc[dev["fold"] == fold, "pand_id"].astype(str))

            if holdout_pand_ids_path is not None and holdout_pand_ids_path.exists():
                holdout_pids = set(pd.read_parquet(holdout_pand_ids_path)["pand_id"].astype(str))
                leak = keep_pids & holdout_pids
                assert not leak, f"hold-out leak: {len(leak)} pand_ids in {split} split"

        gt["pand_id"] = gt["pand_id"].astype(str)
        manifest["pand_id"] = manifest["pand_id"].astype(str)
        gt_sub = gt[gt["pand_id"].isin(keep_pids)].copy()
        manifest_sub = manifest[manifest["pand_id"].isin(keep_pids)].copy()

        self.images_by_pand: dict[str, list[str]] = (
            manifest_sub.groupby("pand_id")["file_path"]
            .apply(lambda s: list(s)[:MAX_IMAGES])
            .to_dict()
        )

        gt_sub = gt_sub[gt_sub["pand_id"].isin(self.images_by_pand)].reset_index(drop=True)
        self.gt = gt_sub

        if year_mean is None:
            self.year_mean = float(gt_sub["bouwjaar"].mean())
            self.year_std = float(gt_sub["bouwjaar"].std() or 1.0)
            self.floors_mean = float(gt_sub["num_floors"].mean())
            self.floors_std = float(gt_sub["num_floors"].std() or 1.0)
        else:
            self.year_mean = year_mean
            self.year_std = year_std
            self.floors_mean = floors_mean
            self.floors_std = floors_std

        logger.info(
            "Stage1ImageDataset[split=%s fold=%s]: %d buildings, %d images total, cities=%s",
            split, fold, len(self.gt),
            sum(len(v) for v in self.images_by_pand.values()),
            self.gt["city"].value_counts().to_dict(),
        )

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for the 4 building_type classes."""
        counts = self.gt["building_type"].value_counts()
        weights = torch.tensor(
            [1.0 / max(int(counts.get(label, 0)), 1) for label in ["SFH", "TH", "MFH", "AB"]],
            dtype=torch.float32,
        )
        weights = weights / weights.sum() * len(weights)
        return weights

    def __len__(self) -> int:
        return len(self.gt)

    def __getitem__(self, idx: int) -> dict:
        row = self.gt.iloc[idx]
        pand_id = row["pand_id"]
        paths = self.images_by_pand[pand_id]
        n = len(paths)

        imgs = torch.zeros(MAX_IMAGES, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
        mask = torch.zeros(MAX_IMAGES, dtype=torch.bool)
        for i, p in enumerate(paths):
            img = Image.open(p).convert("RGB")
            imgs[i] = self.transform(img)
            mask[i] = True

        target_type = BUILDING_TYPE_TO_IDX[row["building_type"]]
        target_year_norm = (row["bouwjaar"] - self.year_mean) / self.year_std
        target_floors_norm = (row["num_floors"] - self.floors_mean) / self.floors_std

        return {
            "images": imgs,
            "valid_mask": mask,
            "target_type": torch.tensor(target_type, dtype=torch.long),
            "target_year_norm": torch.tensor(target_year_norm, dtype=torch.float32),
            "target_floors_norm": torch.tensor(target_floors_norm, dtype=torch.float32),
            "bouwjaar": torch.tensor(row["bouwjaar"], dtype=torch.float32),
            "num_floors": torch.tensor(row["num_floors"], dtype=torch.float32),
            "pand_id": pand_id,
            "city": row["city"],
            "n_images": n,
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "images": torch.stack([b["images"] for b in batch]),
        "valid_mask": torch.stack([b["valid_mask"] for b in batch]),
        "target_type": torch.stack([b["target_type"] for b in batch]),
        "target_year_norm": torch.stack([b["target_year_norm"] for b in batch]),
        "target_floors_norm": torch.stack([b["target_floors_norm"] for b in batch]),
        "bouwjaar": torch.stack([b["bouwjaar"] for b in batch]),
        "num_floors": torch.stack([b["num_floors"] for b in batch]),
        "pand_id": [b["pand_id"] for b in batch],
        "city": [b["city"] for b in batch],
        "n_images": torch.tensor([b["n_images"] for b in batch], dtype=torch.long),
    }
