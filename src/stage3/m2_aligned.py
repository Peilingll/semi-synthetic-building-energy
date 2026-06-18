"""Aligned M2: end-to-end DINOv2 -> energy label, mirroring the Stage 1 DINOv2
training recipe (same frozen backbone + trunk, class-weighted CE, AdamW, cosine,
AMP, early stop on val energy macro-F1, 5-fold). Only the output is the 7-class
Energieklasse head; no type/year/floor, no TABULA.

Hold-out: pick the best fold by val energy macro-F1 (same protocol as
eval_holdout.py), evaluate once on the hold-out, restricted to the same common
set as Stage 3 M1/M3, and compare.

ENV: conda stage1-gpu.
    python -m src.stage3.m2_aligned --all-folds
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
    ENERGY_LABELS,
    IDX_TO_ENERGY,
    Stage1ImageDataset,
    collate_fn,
    energy_to_idx,
)
from src.stage1.models import build_model
from src.stage2.metrics import evaluate

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
REPORTS = REPO / "reports" / "stage3"
CKPT_DIR = REPO / "models" / "stage3"
N_ENERGY = 7
MODEL = "dinov2_energy"


def energy_class_weights(ds: Stage1ImageDataset) -> torch.Tensor:
    counts = np.zeros(N_ENERGY, dtype=np.float64)
    for ek in ds.gt["Energieklasse"]:
        i = energy_to_idx(ek)
        if i >= 0:
            counts[i] += 1
    w = 1.0 / np.clip(counts, 1, None)
    w = w / w.sum() * N_ENERGY
    return torch.tensor(w, dtype=torch.float32)


def run_epoch(model, loader, optimizer, scaler, device, train, class_weights) -> dict:
    model.train(train)
    total_loss, n = 0.0, 0
    preds, targets = [], []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        mask = batch["valid_mask"].to(device, non_blocking=True)
        t = batch["target_energy"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train), \
             torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(scaler is not None)):
            out = model(images, mask)
            loss = F.cross_entropy(out["logits_energy"], t, weight=class_weights)
        if train:
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
        bs = images.size(0)
        total_loss += loss.item() * bs
        n += bs
        preds.append(out["logits_energy"].argmax(-1).detach().cpu().numpy())
        targets.append(t.detach().cpu().numpy())
    p, y = np.concatenate(preds), np.concatenate(targets)
    return {
        "loss": total_loss / n,
        "macro_f1": float(f1_score(y, p, labels=list(range(N_ENERGY)), average="macro", zero_division=0)),
        "acc": float((p == y).mean()),
    }


@torch.no_grad()
def predict_energy(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        out = model(batch["images"].to(device), batch["valid_mask"].to(device))
        pred = out["logits_energy"].argmax(-1).cpu().numpy()
        for i, pid in enumerate(batch["pand_id"]):
            rows.append({
                "pand_id": pid, "city": batch["city"][i],
                "pred": IDX_TO_ENERGY[int(pred[i])],
                "true": IDX_TO_ENERGY[int(batch["target_energy"][i])],
            })
    return pd.DataFrame(rows)


def loaders(split, fold, args, **stats):
    ds = Stage1ImageDataset(
        manifest_path=PROC / "svi_manifest.parquet", gt_path=PROC / "stage1_gt.parquet",
        split=split, dev_fold_indices_path=PROC / "dev_fold_indices.parquet",
        holdout_pand_ids_path=PROC / "holdout_test_pand_ids.parquet",
        fold=fold, energy_target=True, **stats,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=(split == "train"),
                    collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    return ds, dl


def train_one_fold(args, fold, device) -> dict:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, train_dl = loaders("train", fold, args)
    _, val_dl = loaders("val", fold, args)

    model = build_model(MODEL).to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if (device == "cuda" and not args.no_amp) else None
    cw = energy_class_weights(train_ds).to(device)
    logger.info("fold %d energy class_weights: %s", fold, [round(x, 2) for x in cw.tolist()])

    ckpt_path = CKPT_DIR / f"pooled_{MODEL}_fold{fold}.pt"
    best_f1, best_epoch, since = -1.0, -1, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_dl, optimizer, scaler, device, True, cw)
        va = run_epoch(model, val_dl, None, None, device, False, cw)
        scheduler.step()
        if device == "cuda":
            torch.cuda.empty_cache()
        logger.info("fold %d ep %d/%d tr_loss=%.4f tr_f1=%.3f va_loss=%.4f va_f1=%.3f va_acc=%.3f",
                    fold, epoch, args.epochs, tr["loss"], tr["macro_f1"], va["loss"], va["macro_f1"], va["acc"])
        if va["macro_f1"] > best_f1:
            best_f1, best_epoch, since = va["macro_f1"], epoch, 0
            torch.save({"model_state": model.state_dict(), "fold": fold,
                        "val_macro_f1": best_f1, "epoch": epoch, "args": vars(args)}, ckpt_path)
        else:
            since += 1
            if since >= args.patience:
                logger.info("fold %d early stop ep %d (best %d f1=%.3f)", fold, epoch, best_epoch, best_f1)
                break
    logger.info("fold %d done %.1fs best ep %d val_f1=%.3f", fold, time.time() - t0, best_epoch, best_f1)
    return {"fold": fold, "best_val_macro_f1": best_f1, "ckpt": str(ckpt_path)}


def evaluate_holdout(args, device):
    # best fold by val energy macro-F1 (same protocol as eval_holdout.py)
    best = max(range(5), key=lambda f: torch.load(
        CKPT_DIR / f"pooled_{MODEL}_fold{f}.pt", map_location="cpu", weights_only=False)["val_macro_f1"])
    ckpt = torch.load(CKPT_DIR / f"pooled_{MODEL}_fold{best}.pt", map_location=device, weights_only=False)
    logger.info("holdout: best fold %d val_f1=%.3f", best, ckpt["val_macro_f1"])
    model = build_model(MODEL).to(device)
    model.load_state_dict(ckpt["model_state"])

    _, ho_dl = loaders("holdout", None, args)
    preds = predict_energy(model, ho_dl, device)

    # restrict to the same common set as Stage 3 M1/M3
    m1 = pd.read_parquet(REPORTS / "M1_holdout_preds.parquet")
    common = set(m1["pand_id"].astype(str))
    preds["pand_id"] = preds["pand_id"].astype(str)
    preds = preds[preds["pand_id"].isin(common)].reset_index(drop=True)

    rep = evaluate(preds[["true", "pred"]], with_ci=True)
    rep["route"] = "M2-DINOv2-aligned"
    rep["source_fold"] = int(best)
    rep["dev_val_macro_f1"] = float(ckpt["val_macro_f1"])
    REPORTS.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(REPORTS / "M2-DINOv2_holdout_preds.parquet", index=False)
    (REPORTS / "M2-DINOv2_metrics.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(r):
        return json.load(open(REPORTS / f"{r}_metrics.json"))
    m1m, m3m = load("M1"), load("M3-DINOv2")
    print(f"\n{'route':22s} {'macroF1':>8s} {'kappa':>7s} {'acc':>7s}")
    for nm, r in [("M1 (GT)", m1m), ("M3-DINOv2 (decomp)", m3m), ("M2-DINOv2 (aligned)", rep)]:
        print(f"{nm:22s} {r['macro_f1']:8.4f} {r['quadratic_kappa']:7.4f} {r['accuracy']:7.4f}")
    print(f"\nM2(aligned) - M3 : mF1 {rep['macro_f1']-m3m['macro_f1']:+.4f}  kappa {rep['quadratic_kappa']-m3m['quadratic_kappa']:+.4f}")
    print(f"M2(aligned) - M1 : mF1 {rep['macro_f1']-m1m['macro_f1']:+.4f}  kappa {rep['quadratic_kappa']-m1m['quadratic_kappa']:+.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all-folds", action="store_true")
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--holdout-only", action="store_true", help="skip training, just eval existing ckpts")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.holdout_only:
        folds = list(range(5)) if args.all_folds else [args.fold]
        assert folds[0] is not None, "pass --all-folds or --fold N"
        t0 = time.time()
        for f in folds:
            train_one_fold(args, f, device)
        logger.info("===== training done in %.1f min =====", (time.time() - t0) / 60)
    evaluate_holdout(args, device)


if __name__ == "__main__":
    main()
