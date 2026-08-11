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
    Stage1ImageDataset,
    collate_fn,
    energy_to_idx,
    idx_to_energy,
    n_energy_classes,
)
from src.stage1.models import build_model
from src.stage2.metrics import evaluate

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
PROC = REPO / "data" / "processed"
REPORTS = REPO / "reports" / "stage3"
CKPT_DIR = REPO / "models" / "stage3"
MODEL = "dinov2_energy"
ROUTE_TAG = {"dinov2_energy": "M2-DINOv2", "resnet50_energy": "M2-ResNet50"}


def task_suffix(task: str) -> str:
    return "" if task == "7class" else f"_{task}"


def energy_class_weights(ds: Stage1ImageDataset) -> torch.Tensor:
    n_cls = n_energy_classes(ds.energy_task)
    counts = np.zeros(n_cls, dtype=np.float64)
    for ek in ds.gt["Energieklasse"]:
        i = energy_to_idx(ek, ds.energy_task)
        if i >= 0:
            counts[i] += 1
    w = 1.0 / np.clip(counts, 1, None)
    w = w / w.sum() * n_cls
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
    n_cls = class_weights.numel()
    return {
        "loss": total_loss / n,
        "macro_f1": float(f1_score(y, p, labels=list(range(n_cls)), average="macro", zero_division=0)),
        "acc": float((p == y).mean()),
    }


@torch.no_grad()
def predict_energy(model, loader, device, task: str = "7class") -> pd.DataFrame:
    model.eval()
    i2e = idx_to_energy(task)
    rows = []
    for batch in loader:
        out = model(batch["images"].to(device), batch["valid_mask"].to(device))
        logits = out["logits_energy"]
        pred = logits.argmax(-1).cpu().numpy()
        # P(positive class) for the binary ROC/PR curves; unused for 7-class.
        proba = torch.softmax(logits.float(), dim=-1)[:, -1].cpu().numpy()
        for i, pid in enumerate(batch["pand_id"]):
            rows.append({
                "pand_id": pid, "city": batch["city"][i],
                "pred": i2e[int(pred[i])],
                "true": i2e[int(batch["target_energy"][i])],
                "proba": float(proba[i]),
            })
    return pd.DataFrame(rows)


def loaders(split, fold, args, **stats):
    ds = Stage1ImageDataset(
        manifest_path=PROC / "svi_manifest.parquet", gt_path=PROC / "stage1_gt.parquet",
        split=split, dev_fold_indices_path=PROC / "dev_fold_indices.parquet",
        holdout_pand_ids_path=PROC / "holdout_test_pand_ids.parquet",
        fold=fold, energy_target=True, energy_task=args.task, **stats,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=(split == "train"),
                    collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    return ds, dl


def train_one_fold(args, fold, device) -> dict:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, train_dl = loaders("train", fold, args)
    _, val_dl = loaders("val", fold, args)

    model = build_model(args.model, n_energy_classes(args.task)).to(device)
    if args.head_lr is not None and hasattr(model, "param_groups"):
        optimizer = torch.optim.AdamW(model.param_groups(args.lr, args.head_lr, args.weight_decay))
        logger.info("discriminative lr: backbone=%g head=%g", args.lr, args.head_lr)
    else:
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if (device == "cuda" and not args.no_amp) else None
    cw = energy_class_weights(train_ds).to(device)
    logger.info("fold %d energy class_weights: %s", fold, [round(x, 2) for x in cw.tolist()])

    ckpt_path = CKPT_DIR / f"pooled_{args.model}{task_suffix(args.task)}_fold{fold}.pt"
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
    tag = ROUTE_TAG[args.model]          # e.g. "M2-ResNet50"
    m3_route = tag.replace("M2-", "M3-")  # corresponding decomposed route
    sfx = task_suffix(args.task)

    def ckpt_path(f):
        return CKPT_DIR / f"pooled_{args.model}{sfx}_fold{f}.pt"

    # best fold by val energy macro-F1 (same protocol as eval_holdout.py).
    # --eval-folds narrows the candidate set when a run was interrupted; the
    # result is then a deviation from the 5-fold protocol and is written under a
    # _provisional name so it cannot be mistaken for the real one.
    folds = ([int(f) for f in args.eval_folds.split(",")] if args.eval_folds
             else list(range(5)))
    out_sfx = sfx  # sfx keeps naming the INPUTS (checkpoints, M1 preds)
    if sorted(folds) != list(range(5)):
        out_sfx += "_provisional"
        logger.warning("evaluating best-of-%d folds %s, NOT the 5-fold protocol",
                       len(folds), folds)
    best = max(folds, key=lambda f: torch.load(
        ckpt_path(f), map_location="cpu", weights_only=False)["val_macro_f1"])
    ckpt = torch.load(ckpt_path(best), map_location=device, weights_only=False)
    logger.info("holdout: best fold %d val_f1=%.3f", best, ckpt["val_macro_f1"])
    model = build_model(args.model, n_energy_classes(args.task)).to(device)
    model.load_state_dict(ckpt["model_state"])

    _, ho_dl = loaders("holdout", None, args)
    preds = predict_energy(model, ho_dl, device, args.task)

    # restrict to the same common set as Stage 3 M1/M3
    m1 = pd.read_parquet(REPORTS / f"M1{sfx}_holdout_preds.parquet")
    common = set(m1["pand_id"].astype(str))
    preds["pand_id"] = preds["pand_id"].astype(str)
    preds = preds[preds["pand_id"].isin(common)].reset_index(drop=True)

    proba = preds["proba"] if args.task == "binary" else None
    rep = evaluate(preds[["true", "pred"]], with_ci=True, task=args.task, proba=proba)
    rep["route"] = f"{tag}-aligned"
    rep["source_fold"] = int(best)
    rep["dev_val_macro_f1"] = float(ckpt["val_macro_f1"])
    rep["note"] = "full fine-tune end-to-end (Stage 1 ResNet recipe)" if args.model == "resnet50_energy" else "aligned"
    rep["eval_folds"] = folds
    REPORTS.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(REPORTS / f"{tag}{out_sfx}_holdout_preds.parquet", index=False)
    (REPORTS / f"{tag}{out_sfx}_metrics.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(r):
        return json.load(open(REPORTS / f"{r}{sfx}_metrics.json"))
    m1m, m3m = load("M1"), load(m3_route)
    print(f"\n{'route':22s} {'macroF1':>8s} {'kappa':>7s} {'acc':>7s}")
    for nm, r in [("M1 (GT)", m1m), (f"{m3_route} (decomp)", m3m), (f"{tag} (aligned)", rep)]:
        print(f"{nm:22s} {r['macro_f1']:8.4f} {r['quadratic_kappa']:7.4f} {r['accuracy']:7.4f}")
    print(f"\nM2(aligned) - M3 : mF1 {rep['macro_f1']-m3m['macro_f1']:+.4f}  kappa {rep['quadratic_kappa']-m3m['quadratic_kappa']:+.4f}")
    print(f"M2(aligned) - M1 : mF1 {rep['macro_f1']-m1m['macro_f1']:+.4f}  kappa {rep['quadratic_kappa']-m1m['quadratic_kappa']:+.4f}")


# ---------------------------------------------------------------------------
# Cached-embedding mode: frozen backbone is re-run once (extract_embeddings),
# then the SAME neural head (trunk 768->256 + GELU + Dropout + 7-class) is
# trained on cached embeddings. Identical architecture + CE + AdamW + cosine +
# early-stop recipe as the full path; the ONLY deviation from Stage 1 is no
# train-time augmentation (frozen backbone => augmentation effect is marginal).
# ~seconds vs ~10 h, because the backbone is not re-forwarded every epoch.
# ---------------------------------------------------------------------------

EMB_COLS = [f"e{i}" for i in range(768)]


class EnergyHead(torch.nn.Module):
    """trunk + head, identical to DINOv2FrozenEnergy's trunk + head_energy."""

    def __init__(self, in_dim=768, hidden=256, n=7, dropout=0.3):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout))
        self.head = torch.nn.Linear(hidden, n)

    def forward(self, x):
        return self.head(self.trunk(x))


def _cw_from_labels(y: np.ndarray, n_cls: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_cls).astype(np.float64)
    w = 1.0 / np.clip(counts, 1, None)
    return torch.tensor(w / w.sum() * n_cls, dtype=torch.float32)


def _train_head_cached(Xtr, ytr, Xva, yva, args, device):
    n_cls = n_energy_classes(args.task)
    head = EnergyHead(n=n_cls).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    cw = _cw_from_labels(ytr, n_cls).to(device)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    n, bs = len(Xtr_t), 256
    best_f1, best_state, since = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(head(Xtr_t[idx]), ytr_t[idx], weight=cw).backward()
            opt.step()
        sch.step()
        head.eval()
        with torch.no_grad():
            vp = head(Xva_t).argmax(-1).cpu().numpy()
        f1 = f1_score(yva, vp, labels=list(range(n_cls)), average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, since = f1, 0
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
        else:
            since += 1
            if since >= args.patience:
                break
    return best_f1, best_state


def main_cached(args, device):
    sfx = task_suffix(args.task)
    gt = pd.read_parquet(PROC / "stage1_gt.parquet")
    gt["pand_id"] = gt["pand_id"].astype(str)
    lab = {p: energy_to_idx(e, args.task) for p, e in zip(gt["pand_id"], gt["Energieklasse"])}
    folds = pd.read_parquet(PROC / "dev_fold_indices.parquet")
    folds["pand_id"] = folds["pand_id"].astype(str)
    fold_map = dict(zip(folds["pand_id"], folds["fold"]))

    dev = pd.read_parquet(REPORTS / "embeddings_dev.parquet")
    dev["pand_id"] = dev["pand_id"].astype(str)
    dev["fold"] = dev["pand_id"].map(fold_map)
    dev["y"] = dev["pand_id"].map(lab)
    ho = pd.read_parquet(REPORTS / "embeddings_holdout.parquet")
    ho["pand_id"] = ho["pand_id"].astype(str)
    ho["y"] = ho["pand_id"].map(lab)

    states, f1s = {}, {}
    for f in range(5):
        tr, va = dev[dev["fold"] != f], dev[dev["fold"] == f]
        f1, st = _train_head_cached(tr[EMB_COLS].values, tr["y"].values.astype(int),
                                    va[EMB_COLS].values, va["y"].values.astype(int), args, device)
        states[f], f1s[f] = st, f1
        logger.info("cached fold %d val energy macro-F1=%.4f", f, f1)

    best = max(f1s, key=f1s.get)
    logger.info("best fold %d (val f1=%.4f)", best, f1s[best])
    i2e = idx_to_energy(args.task)
    head = EnergyHead(n=n_energy_classes(args.task)).to(device)
    head.load_state_dict(states[best])
    head.eval()
    with torch.no_grad():
        ho_logits = head(torch.tensor(ho[EMB_COLS].values, dtype=torch.float32, device=device))
        ho_pred = ho_logits.argmax(-1).cpu().numpy()
        ho_proba = torch.softmax(ho_logits.float(), dim=-1)[:, -1].cpu().numpy()

    preds = pd.DataFrame({
        "pand_id": ho["pand_id"].values,
        "true": [i2e[i] for i in ho["y"].values.astype(int)],
        "pred": [i2e[i] for i in ho_pred],
        "proba": ho_proba,
    })
    m1 = pd.read_parquet(REPORTS / f"M1{sfx}_holdout_preds.parquet")
    common = set(m1["pand_id"].astype(str))
    preds = preds[preds["pand_id"].isin(common)].reset_index(drop=True)

    proba = preds["proba"] if args.task == "binary" else None
    rep = evaluate(preds[["true", "pred"]], with_ci=True, task=args.task, proba=proba)
    rep["route"] = "M2-DINOv2-aligned-cached"
    rep["source_fold"] = int(best)
    rep["dev_val_macro_f1"] = float(f1s[best])
    rep["note"] = "frozen DINOv2 cached embeddings + Stage1-style neural head; no train-time augmentation"
    REPORTS.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(REPORTS / f"M2-DINOv2{sfx}_holdout_preds.parquet", index=False)
    (REPORTS / f"M2-DINOv2{sfx}_metrics.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(r):
        return json.load(open(REPORTS / f"{r}{sfx}_metrics.json"))
    m1m, m3m = load("M1"), load("M3-DINOv2")
    print(f"\n{'route':24s} {'macroF1':>8s} {'kappa':>7s} {'acc':>7s}")
    for nm, r in [("M1 (GT)", m1m), ("M3-DINOv2 (decomp)", m3m), ("M2-DINOv2 (aligned)", rep)]:
        print(f"{nm:24s} {r['macro_f1']:8.4f} {r['quadratic_kappa']:7.4f} {r['accuracy']:7.4f}")
    print(f"\nM2 - M3 : mF1 {rep['macro_f1']-m3m['macro_f1']:+.4f}  kappa {rep['quadratic_kappa']-m3m['quadratic_kappa']:+.4f}")
    print(f"M2 - M1 : mF1 {rep['macro_f1']-m1m['macro_f1']:+.4f}  kappa {rep['quadratic_kappa']-m1m['quadratic_kappa']:+.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dinov2_energy",
                   help="dinov2_energy (cached/frozen) or resnet50_energy (full fine-tune)")
    p.add_argument("--task", default="7class", choices=["7class", "binary"],
                   help="7-class A-G (original) or binary A-C | D-G (Sun cut)")
    p.add_argument("--head-lr", type=float, default=None,
                   help="separate lr for trunk+head (models with param_groups, e.g. resnet50_energy)")
    p.add_argument("--cached", action="store_true",
                   help="train neural head on cached embeddings (fast; no augmentation; dinov2 only)")
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
    p.add_argument("--eval-folds", default=None,
                   help="comma list of folds to select the hold-out model from "
                        "(default all 5; anything else writes a _provisional result)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.cached:
        main_cached(args, device)
        return

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
