"""Streaming PIL dataset for Stage 1.

One __getitem__ returns one building (pand_id) with up to MAX_IMAGES images,
padded to a fixed length with a mask. Transforms follow spec §3.2.4 (random
resized crop + horizontal flip + color jitter, no vertical flip).
"""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
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


def build_transforms(split: Literal["train", "val"]) -> transforms.Compose:
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
    """Dataset of buildings; each sample is one pand_id with up to 8 images."""

    def __init__(
        self,
        manifest_path: Path,
        gt_path: Path,
        fold_indices_path: Path,
        city: str,
        split: Literal["train", "val"],
        fold: int,
        year_mean: float | None = None,
        year_std: float | None = None,
        floors_mean: float | None = None,
        floors_std: float | None = None,
    ):
        self.split = split
        self.transform = build_transforms(split)

        manifest = pd.read_parquet(manifest_path)
        manifest = manifest[manifest["city"] == city].copy()
        gt = pd.read_parquet(gt_path)
        folds = pd.read_parquet(fold_indices_path)

        if split == "train":
            keep_pids = folds.loc[folds["fold"] != fold, "pand_id"]
        else:
            keep_pids = folds.loc[folds["fold"] == fold, "pand_id"]

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
            "Stage1ImageDataset[%s fold=%d split=%s]: %d buildings, %d images total",
            city, fold, split, len(self.gt),
            sum(len(v) for v in self.images_by_pand.values()),
        )

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
        "n_images": torch.tensor([b["n_images"] for b in batch], dtype=torch.long),
    }
