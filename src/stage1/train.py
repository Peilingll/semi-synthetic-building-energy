"""Stage 1 Phase B training loop.

Reads four-city manifest + stage1_gt.parquet + dev_fold_indices.parquet.
Supports --fold N (single fold) or --all-folds (sequential 0..4).
Class-weighted CE, regularised AdamW, cosine schedule, AMP fp16,
EarlyStopping on val macro-F1.
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


def run_epoch(model, loader, optimizer, scaler, device, train: bool, class_weights: torch.Tensor) -> dict:
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
            loss_ce = F.cross_entropy(out["logits_type"], t_type, weight=class_weights)
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
                    "city": batch["city"][i],
                    "pred_type_idx": int(pred_type[i]),
                    "pred_type": IDX_TO_BUILDING_TYPE[int(pred_type[i])],
                    "pred_year": float(pred_year[i]),
                    "pred_floors": float(pred_floors[i]),
                    "true_type": IDX_TO_BUILDING_TYPE[int(batch["target_type"][i])],
                    "true_bouwjaar": float(batch["bouwjaar"][i]),
                    "true_num_floors": float(batch["num_floors"][i]),
                })
    return pd.DataFrame(rows)


def train_one_fold(args, fold: int, paths: dict, run_tag_prefix: str = "pooled") -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "reports" / "stage1" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = repo_root / "models" / "stage1"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_tag = f"{run_tag_prefix}_{args.model}_fold{fold}"  # checkpoint name keeps the model
    report_tag = f"{run_tag_prefix}_fold{fold}"  # report folder already carries it
    ckpt_path = ckpt_dir / f"{run_tag}.pt"
    preds_path = out_dir / f"{report_tag}_val_preds.parquet"
    history_path = out_dir / f"{report_tag}_history.json"

    common = dict(
        manifest_path=paths["manifest"], gt_path=paths["gt"],
        dev_fold_indices_path=paths["dev_folds"],
        holdout_pand_ids_path=paths["holdout"],
        fold=fold,
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
    if args.head_lr is not None and hasattr(model, "param_groups"):
        # discriminative lr: pretrained backbone at --lr, randomly initialised
        # trunk/heads at --head-lr (norm/bias excluded from weight decay)
        optimizer = torch.optim.AdamW(
            model.param_groups(args.lr, args.head_lr, args.weight_decay),
        )
        logger.info("param groups: backbone lr=%g head lr=%g", args.lr, args.head_lr)
    else:
        optimizer = torch.optim.AdamW(
            model.trainable_parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if (device == "cuda" and not args.no_amp) else None

    class_weights = train_ds.class_weights().to(device)
    logger.info("class_weights (SFH/TH/MFH/AB): %s", class_weights.tolist())

    best_f1 = -1.0
    best_epoch = -1
    epochs_since_best = 0
    history = []

    logger.info(
        "fold=%d device=%s amp=%s lr=%g wd=%g batch=%d patience=%d",
        fold, device, scaler is not None, args.lr, args.weight_decay,
        args.batch_size, args.patience,
    )
    logger.info(
        "year_mean=%.2f year_std=%.2f floors_mean=%.2f floors_std=%.2f",
        train_ds.year_mean, train_ds.year_std,
        train_ds.floors_mean, train_ds.floors_std,
    )

    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        tr = run_epoch(model, train_loader, optimizer, scaler, device, True, class_weights)
        va = run_epoch(model, val_loader, None, None, device, False, class_weights)
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
            "fold %d epoch %d/%d %.1fs  tr_loss=%.4f tr_f1=%.3f tr_acc=%.3f  va_loss=%.4f va_f1=%.3f va_acc=%.3f",
            fold, epoch, args.epochs, epoch_dt,
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
                "class_weights": class_weights.cpu().tolist(),
                "val_macro_f1": va["type_macro_f1"], "args": vars(args),
                "fold": fold,
            }, ckpt_path)
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                logger.info("fold %d early stop at epoch %d (best %d, val_f1=%.3f)",
                            fold, epoch, best_epoch, best_f1)
                break

    total_dt = time.time() - t0
    logger.info("fold %d done. total %.1fs. best epoch %d val_f1=%.3f.",
                fold, total_dt, best_epoch, best_f1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    preds = predict(model, val_loader, device,
                    train_ds.year_mean, train_ds.year_std,
                    train_ds.floors_mean, train_ds.floors_std)
    preds.to_parquet(preds_path, index=False)
    logger.info("fold %d wrote %s (%d rows)", fold, preds_path, len(preds))

    summary = {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "total_sec": round(total_dt, 1),
        "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device == "cuda" else None,
        "history": history,
        "args": vars(args),
    }
    history_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("fold %d wrote %s", fold, history_path)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dinov2_frozen")
    parser.add_argument("--fold", type=int, default=None, help="single fold (0-4); ignored if --all-folds/--folds")
    parser.add_argument("--all-folds", action="store_true", help="run folds 0..4 sequentially")
    parser.add_argument("--folds", type=str, default=None,
                        help="comma-separated fold list, e.g. '1,2,3,4' (skip an already-run fold)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--head-lr", type=float, default=None,
                        help="separate lr for trunk+heads (models exposing param_groups); None = uniform --lr")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    repo_root = Path(__file__).resolve().parents[2]
    paths = {
        "manifest": repo_root / "data" / "processed" / "svi_manifest.parquet",
        "gt": repo_root / "data" / "processed" / "stage1_gt.parquet",
        "dev_folds": repo_root / "data" / "processed" / "dev_fold_indices.parquet",
        "holdout": repo_root / "data" / "processed" / "holdout_test_pand_ids.parquet",
    }
    for k, p in paths.items():
        assert p.exists(), f"missing input: {k} = {p}"

    if args.all_folds:
        folds = list(range(5))
    elif args.folds:
        folds = [int(f) for f in args.folds.split(",")]
        assert all(0 <= f <= 4 for f in folds), f"folds out of range: {folds}"
    else:
        assert args.fold is not None, "must pass --fold N, --folds list or --all-folds"
        folds = [args.fold]

    all_summaries = []
    t0 = time.time()
    for fold in folds:
        summary = train_one_fold(args, fold, paths)
        all_summaries.append(summary)

    if len(folds) > 1:
        logger.info("===== ALL FOLDS DONE in %.1f min =====", (time.time() - t0) / 60)
        f1s = [s["best_val_macro_f1"] for s in all_summaries]
        logger.info("per-fold best val_f1: %s", [round(x, 3) for x in f1s])
        logger.info("mean ± std: %.3f ± %.3f", float(np.mean(f1s)), float(np.std(f1s)))


if __name__ == "__main__":
    main()
