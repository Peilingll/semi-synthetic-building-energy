"""LOCO control: is the cross-city drop about the city, or about sample size?

The headline LOCO-Amsterdam run changes two things at once. It trains on
R+U+D (2,075 buildings) instead of the pooled dev set (8,068) AND it removes
Amsterdam from training. So the degradation it reports -- ROC-AUC 0.647 -> 0.528
downstream, joint-cell accuracy below the majority baseline upstream -- cannot be
attributed to either cause on its own.

This holds sample size fixed and varies only the city composition:

  arm A "loco"     train = the 2,075 R+U+D buildings (no Amsterdam)
  arm B "matched"  train = 2,075 sampled from all four cities, minus the test set

Both predict the same 1,595-building Amsterdam test set (AMS INTERSECT pooled
hold-out), both train the same head on the same cached DINOv2 embeddings with the
same protocol, so the only difference is whether Amsterdam is in the training
pool.

Note the direction of the composition difference, which is the opposite of the
intuition: Amsterdam is ~80% of the imaged stock, so arm B's pool is MORE
homogeneous than arm A's (roughly 90% apartments against 65%). Arm A trains on
the more diverse pool but a mismatched one; arm B on a narrower pool that looks
like the deployment site. The comparison therefore asks whether matching the
deployment distribution beats training-pool diversity at equal n.

This is an internal A/B. Both arms use the cached-embedding head rather than the
full augmented Stage 1 recipe, so the absolute numbers are not interchangeable
with `T4_joint_cell_loco_amsterdam.md`; only the A-vs-B gap is the result.

Usage:
    "D:/conda_envs/stage1-gpu/python.exe" -m src.stage1.loco_control
"""

import argparse
import json
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from src.stage2.features import REPO_ROOT
from src.tabula_matcher import classify_period

logger = logging.getLogger(__name__)

PROC = REPO_ROOT / "data" / "processed"
STAGE3 = REPO_ROOT / "reports" / "stage3"
TABLES = REPO_ROOT / "reports" / "tables" / "stage1"

EMB_COLS = [f"e{i}" for i in range(768)]
TYPES = ["SFH", "TH", "MFH", "AB"]
TYPE_TO_IDX = {t: i for i, t in enumerate(TYPES)}
N_FOLDS = 5
SEED = 0


class AttrHead(torch.nn.Module):
    """trunk + type/year heads, mirroring DINOv2FrozenMLP's trunk and heads."""

    def __init__(self, in_dim=768, hidden=256, dropout=0.3):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout))
        self.head_type = torch.nn.Linear(hidden, len(TYPES))
        self.head_year = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.head_type(h), self.head_year(h).squeeze(-1)


def load_embeddings() -> pd.DataFrame:
    frames = [pd.read_parquet(STAGE3 / f"embeddings_{s}.parquet")
              for s in ("dev", "holdout")]
    e = pd.concat(frames, ignore_index=True)
    e["pand_id"] = e["pand_id"].astype(str)
    return e.drop_duplicates("pand_id").reset_index(drop=True)


def train_arm(tr: pd.DataFrame, te: pd.DataFrame, args, device) -> pd.DataFrame:
    """5-fold on the arm's own pool, best fold by val type macro-F1, predict test."""
    X = tr[EMB_COLS].to_numpy(dtype=np.float32)
    y_type = tr["type_idx"].to_numpy()
    y_year = tr["year_norm"].to_numpy(dtype=np.float32)
    rng = np.random.default_rng(SEED)
    folds = rng.permutation(len(tr)) % N_FOLDS

    Xte = torch.tensor(te[EMB_COLS].to_numpy(dtype=np.float32), device=device)
    best_f1, best_state = -1.0, None
    for f in range(N_FOLDS):
        trm, vam = folds != f, folds == f
        head = AttrHead().to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        Xtr = torch.tensor(X[trm], device=device)
        ttr = torch.tensor(y_type[trm], dtype=torch.long, device=device)
        ytr = torch.tensor(y_year[trm], device=device)
        Xva = torch.tensor(X[vam], device=device)
        n, bs = len(Xtr), 256
        fold_best, fold_state, since = -1.0, None, 0
        for _ in range(args.epochs):
            head.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                lt, ly = head(Xtr[idx])
                opt.zero_grad()
                (F.cross_entropy(lt, ttr[idx]) + F.l1_loss(ly, ytr[idx])).backward()
                opt.step()
            sch.step()
            head.eval()
            with torch.no_grad():
                vp = head(Xva)[0].argmax(-1).cpu().numpy()
            f1 = f1_score(y_type[vam], vp, labels=list(range(len(TYPES))),
                          average="macro", zero_division=0)
            if f1 > fold_best:
                fold_best, since = f1, 0
                fold_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            else:
                since += 1
                if since >= args.patience:
                    break
        logger.info("  fold %d val type macro-F1 %.4f", f, fold_best)
        if fold_best > best_f1:
            best_f1, best_state = fold_best, fold_state

    head = AttrHead().to(device)
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        lt, ly = head(Xte)
    return pd.DataFrame({
        "pand_id": te["pand_id"].values,
        "pred_type": [TYPES[i] for i in lt.argmax(-1).cpu().numpy()],
        "pred_year": ly.cpu().numpy() * args.year_std + args.year_mean,
    }), best_f1


def score(pred: pd.DataFrame, te: pd.DataFrame) -> dict:
    d = pred.merge(te[["pand_id", "building_type", "bouwjaar", "tabula_period"]],
                   on="pand_id")
    pred_period = d["pred_year"].apply(classify_period).astype(str)
    true_cell = d["building_type"].astype(str) + "|" + d["tabula_period"].astype(str)
    pred_cell = d["pred_type"].astype(str) + "|" + pred_period
    joint = (pred_cell == true_cell)
    recalls = [joint[true_cell == c].mean() for c in true_cell.unique()]
    return {
        "n": int(len(d)),
        "type_acc": float((d["pred_type"] == d["building_type"]).mean()),
        "type_macro_f1": float(f1_score(d["building_type"], d["pred_type"],
                                        labels=TYPES, average="macro", zero_division=0)),
        "year_mae": float((d["pred_year"] - d["bouwjaar"]).abs().mean()),
        "period_acc": float((pred_period == d["tabula_period"].astype(str)).mean()),
        "joint_acc": float(joint.mean()),
        "majority_cell": float(true_cell.value_counts(normalize=True).iloc[0]),
        "macro_cell_recall": float(np.mean(recalls)),
        "cells": int(true_cell.nunique()),
    }


def composition(df: pd.DataFrame) -> str:
    ab = (df["building_type"].astype(str) == "AB").mean()
    n01 = (df["tabula_period"].astype(str) == "NL.01").mean()
    ams = (df["city"].astype(str) == "amsterdam").mean()
    return f"AB {ab:.1%} / NL.01 {n01:.1%} / Amsterdam {ams:.1%}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=15)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gt = pd.read_parquet(PROC / "stage1_gt.parquet")
    gt["pand_id"] = gt["pand_id"].astype(str)
    emb = load_embeddings()
    df = emb.merge(gt[["pand_id", "city", "building_type", "bouwjaar",
                       "tabula_period"]], on="pand_id", how="inner")
    df = df[df["building_type"].isin(TYPES)].reset_index(drop=True)
    df["type_idx"] = df["building_type"].map(TYPE_TO_IDX)

    test_ids = set(pd.read_parquet(PROC / "loco_amsterdam" / "holdout_vlm_subset.parquet")
                   ["pand_id"].astype(str))
    loco_ids = set(pd.read_parquet(PROC / "loco_amsterdam" / "dev_fold_indices.parquet")
                   ["pand_id"].astype(str))

    te = df[df["pand_id"].isin(test_ids)].reset_index(drop=True)
    pool = df[~df["pand_id"].isin(test_ids)].reset_index(drop=True)
    arm_a = pool[pool["pand_id"].isin(loco_ids)].reset_index(drop=True)
    assert not (set(arm_a["pand_id"]) & test_ids), "arm A leaks into the test set"
    arm_b = pool.sample(n=len(arm_a), random_state=SEED).reset_index(drop=True)

    # year normalisation from arm A, applied to both, so the two heads see the
    # same target scale and the MAEs are directly comparable
    args.year_mean = float(arm_a["bouwjaar"].mean())
    args.year_std = float(arm_a["bouwjaar"].std() or 1.0)
    for d in (arm_a, arm_b):
        d["year_norm"] = (d["bouwjaar"] - args.year_mean) / args.year_std

    logger.info("test  n=%d  %s", len(te), composition(te))
    logger.info("arm A n=%d  %s", len(arm_a), composition(arm_a))
    logger.info("arm B n=%d  %s", len(arm_b), composition(arm_b))

    results = {}
    for name, tr in (("A_loco", arm_a), ("B_matched", arm_b)):
        logger.info("training %s", name)
        pred, val_f1 = train_arm(tr, te, args, device)
        r = score(pred, te)
        r["val_type_macro_f1"] = val_f1
        r["train_composition"] = composition(tr)
        results[name] = r
        logger.info("%s: type_acc=%.4f year_mae=%.2f joint=%.4f",
                    name, r["type_acc"], r["year_mae"], r["joint_acc"])

    a, b = results["A_loco"], results["B_matched"]
    L = [
        f"# T9 — LOCO control: city shift or sample size? (test n={a['n']:,})",
        "",
        "Both arms train **2,075** buildings on the same cached DINOv2 embeddings with "
        "the same head and protocol, and predict the same Amsterdam test set. The only "
        "difference is whether Amsterdam is in the training pool, so sample size is no "
        "longer confounded with city composition.",
        "",
        f"- test set: {a['n']:,} buildings — {composition(te)}",
        f"- arm A (loco, no Amsterdam): {a['train_composition']}",
        f"- arm B (matched, all four cities): {b['train_composition']}",
        "",
        "Amsterdam is ~80% of the imaged stock, so arm B's pool is the *more* "
        "homogeneous one. Arm A trains on a more diverse but mismatched pool; arm B on "
        "a narrower pool that resembles the deployment site.",
        "",
        "| arm | type acc | type macro-F1 | year MAE | period acc | joint cell | macro-cell recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in (("A — loco (no Amsterdam)", a), ("B — matched (all cities)", b)):
        L.append(f"| {name} | {r['type_acc']:.4f} | {r['type_macro_f1']:.4f} | "
                 f"{r['year_mae']:.2f} | {r['period_acc']:.4f} | {r['joint_acc']:.4f} | "
                 f"{r['macro_cell_recall']:.4f} |")
    L += [
        f"| **B − A** | {b['type_acc'] - a['type_acc']:+.4f} | "
        f"{b['type_macro_f1'] - a['type_macro_f1']:+.4f} | "
        f"{b['year_mae'] - a['year_mae']:+.2f} | "
        f"{b['period_acc'] - a['period_acc']:+.4f} | "
        f"{b['joint_acc'] - a['joint_acc']:+.4f} | "
        f"{b['macro_cell_recall'] - a['macro_cell_recall']:+.4f} |",
        "",
        f"Majority-cell baseline on this test set: **{a['majority_cell']:.4f}** "
        f"({a['cells']} occupied cells).",
        "",
        "Reading: a large B − A means the cross-city gap is real at fixed n. A gap near "
        "zero means the headline LOCO degradation is a sample-size effect and the "
        "cross-city claim does not survive.",
        "",
        "Both arms use the cached-embedding head, not the full augmented Stage 1 recipe, "
        "so these absolute values are not interchangeable with "
        "`T4_joint_cell_loco_amsterdam.md`; the A-vs-B gap is the result.",
        "",
    ]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "T9_loco_control.md").write_text("\n".join(L), encoding="utf-8")
    (STAGE3 / "loco_control.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote %s", TABLES / "T9_loco_control.md")
    for k in ("type_acc", "type_macro_f1", "year_mae", "period_acc", "joint_acc"):
        print(f"{k:16s} A={a[k]:8.4f}  B={b[k]:8.4f}  B-A={b[k]-a[k]:+.4f}")


if __name__ == "__main__":
    main()
