"""Stage 1 training loop (single fold).

Usage:
    python -m src.stage1.train --city delft --fold 0 --epochs 30
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.stage1.dataset import (
    IDX_TO_BUILDING_TYPE,
    Stage1ImageDataset,
    collate_fn,
)
from src.stage1.models import build_model

logger = logging.getLogger(__name__)


def run_epoch(
    model, loader, optimizer, scaler, device, train: bool,
) -> dict:
    model.train(train)
    total_loss = 0.0
    total_ce = 0.0
    total_year_mse = 0.0
    total_floors_mse = 0.0
    n = 0
    all_pred_type = []
    all_target_type = []

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        mask = batch["valid_mask"].to(device, non_blocking=True)
        t_type = batch["target_type"].to(device, non_blocking=True)
        t_year = batch["target_year_norm"].to(device, non_blocking=True)
        t_floors = batch["target_floors_norm"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(scaler is not None)):
            out = model(images, mask)
            loss_ce = F.cross_entropy(out["logits_type"], t_type)
            loss_year = F.mse_loss(out["pred_year_norm"], t_year)
            loss_floors = F.mse_loss(out["pred_floors_norm"], t_floors)
            loss = loss_ce + loss_year + loss_floors

        if train:
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_ce += loss_ce.item() * bs
        total_year_mse += loss_year.item() * bs
        total_floors_mse += loss_floors.item() * bs
        n += bs

        all_pred_type.append(out["logits_type"].argmax(-1).detach().cpu().numpy())
        all_target_type.append(t_type.detach().cpu().numpy())

    pred = np.concatenate(all_pred_type)
    target = np.concatenate(all_target_type)
    macro_f1 = f1_score(target, pred, average="macro", labels=list(IDX_TO_BUILDING_TYPE.keys()), zero_division=0)
    acc = float((pred == target).mean())

    return {
        "loss": total_loss / n,
        "ce": total_ce / n,
        "year_mse": total_year_mse / n,
        "floors_mse": total_floors_mse / n,
        "type_acc": acc,
        "type_macro_f1": float(macro_f1),
    }


def predict(model, loader, device, year_mean, year_std, floors_mean, floors_std) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            mask = batch["valid_mask"].to(device)
            out = model(images, mask)
            pred_type = out["logits_type"].argmax(-1).cpu().numpy()
            pred_year_norm = out["pred_year_norm"].cpu().numpy()
            pred_floors_norm = out["pred_floors_norm"].cpu().numpy()
            pred_year = pred_year_norm * year_std + year_mean
            pred_floors = pred_floors_norm * floors_std + floors_mean

            for i, pid in enumerate(batch["pand_id"]):
                rows.append({
                    "pand_id": pid,
                    "pred_type_idx": int(pred_type[i]),
                    "pred_type": IDX_TO_BUILDING_TYPE[int(pred_type[i])],
                    "pred_year": float(pred_year[i]),
                    "pred_floors": float(pred_floors[i]),
                    "true_type": IDX_TO_BUILDING_TYPE[int(batch["target_type"][i])],
                    "true_bouwjaar": float(batch["bouwjaar"][i]),
                    "true_num_floors": float(batch["num_floors"][i]),
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, choices=["delft", "utrecht", "rotterdam", "amsterdam"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--model", default="dinov2_frozen")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "data" / "processed" / "svi_manifest.parquet"
    gt_path = repo_root / "data" / "processed" / f"stage1_gt_{args.city}.parquet"
    folds_path = repo_root / "data" / "processed" / f"fold_indices_{args.city}.parquet"

    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "reports" / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = repo_root / "models" / "stage1"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_tag = f"{args.city}_{args.model}_fold{args.fold}"
    ckpt_path = ckpt_dir / f"{run_tag}.pt"
    preds_path = out_dir / f"{run_tag}_val_preds.parquet"
    history_path = out_dir / f"{run_tag}_history.json"

    common = dict(
        manifest_path=manifest_path, gt_path=gt_path, fold_indices_path=folds_path,
        city=args.city, fold=args.fold,
    )
    train_ds = Stage1ImageDataset(split="train", **common)
    val_ds = Stage1ImageDataset(
        split="val",
        year_mean=train_ds.year_mean, year_std=train_ds.year_std,
        floors_mean=train_ds.floors_mean, floors_std=train_ds.floors_std,
        **common,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if (device == "cuda" and not args.no_amp) else None

    best_f1 = -1.0
    best_epoch = -1
    epochs_since_best = 0
    history = []

    logger.info("device=%s amp=%s lr=%g batch=%d", device, scaler is not None, args.lr, args.batch_size)
    logger.info("year_mean=%.2f year_std=%.2f floors_mean=%.2f floors_std=%.2f",
                train_ds.year_mean, train_ds.year_std, train_ds.floors_mean, train_ds.floors_std)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        tr = run_epoch(model, train_loader, optimizer, scaler, device, train=True)
        va = run_epoch(model, val_loader, None, None, device, train=False)
        scheduler.step()
        epoch_dt = time.time() - epoch_t0

        record = {
            "epoch": epoch, "epoch_sec": round(epoch_dt, 1),
            "lr": scheduler.get_last_lr()[0],
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}": v for k, v in va.items()},
        }
        history.append(record)
        logger.info(
            "epoch %d/%d  %.1fs  tr_loss=%.4f tr_f1=%.3f tr_acc=%.3f  va_loss=%.4f va_f1=%.3f va_acc=%.3f",
            epoch, args.epochs, epoch_dt,
            tr["loss"], tr["type_macro_f1"], tr["type_acc"],
            va["loss"], va["type_macro_f1"], va["type_acc"],
        )

        if va["type_macro_f1"] > best_f1:
            best_f1 = va["type_macro_f1"]
            best_epoch = epoch
            epochs_since_best = 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "year_mean": train_ds.year_mean, "year_std": train_ds.year_std,
                "floors_mean": train_ds.floors_mean, "floors_std": train_ds.floors_std,
                "val_macro_f1": va["type_macro_f1"], "args": vars(args),
            }, ckpt_path)
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                logger.info("early stopping at epoch %d (best epoch %d, val_f1=%.3f)", epoch, best_epoch, best_f1)
                break

    total_dt = time.time() - t0
    logger.info("done. total %.1fs. best epoch %d val_f1=%.3f. ckpt: %s", total_dt, best_epoch, best_f1, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    preds = predict(model, val_loader, device,
                    train_ds.year_mean, train_ds.year_std,
                    train_ds.floors_mean, train_ds.floors_std)
    preds.to_parquet(preds_path, index=False)
    logger.info("wrote %s (%d rows)", preds_path, len(preds))

    with open(history_path, "w") as f:
        json.dump({
            "args": vars(args), "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
            "total_sec": round(total_dt, 1),
            "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device == "cuda" else None,
            "history": history,
        }, f, indent=2)
    logger.info("wrote %s", history_path)


if __name__ == "__main__":
    main()
