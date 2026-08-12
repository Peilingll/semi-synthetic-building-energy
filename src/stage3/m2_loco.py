"""M2-DINOv2 (aligned-cached) under the LOCO split: end-to-end embedding ->
energy label, trained on the LOCO source cities (R+U+D), evaluated on the
held-out city (Amsterdam). Fills the "LOCO never ran M2" gap.

Protocol mirrors m2_aligned.main_cached exactly (same EnergyHead, HP, 5-fold
best-of selection), only the split files and output names differ:
  folds    data/processed/loco_<city>/dev_fold_indices.parquet
  holdout  data/processed/loco_<city>/holdout_test_pand_ids.parquet
  outputs  M2-DINOv2{sfx}_loco_<city>_{holdout_preds.parquet,metrics.json}

Embeddings are reused from the pooled extract (embeddings_dev + embeddings_
holdout cover all 10,086 buildings; the LOCO train/test pools are subsets).

ENV: conda stage1-gpu (torch). CPU is fine, the head is tiny.
    python -m src.stage3.m2_loco --task 7class
    python -m src.stage3.m2_loco --task binary
"""

import argparse
import json
import logging

import pandas as pd
import torch

from src.stage1.dataset import energy_to_idx, idx_to_energy, n_energy_classes
from src.stage2.metrics import evaluate
from src.stage3.m2_aligned import (
    EMB_COLS,
    PROC,
    REPORTS,
    EnergyHead,
    _train_head_cached,
    task_suffix,
)

logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="7class", choices=["7class", "binary"])
    p.add_argument("--run-tag", default="loco_amsterdam")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=10)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sfx = task_suffix(args.task)
    splits = PROC / args.run_tag

    gt = pd.read_parquet(PROC / "stage1_gt.parquet")
    gt["pand_id"] = gt["pand_id"].astype(str)
    lab = {p_: energy_to_idx(e, args.task) for p_, e in zip(gt["pand_id"], gt["Energieklasse"])}

    emb = pd.concat([
        pd.read_parquet(REPORTS / "embeddings_dev.parquet"),
        pd.read_parquet(REPORTS / "embeddings_holdout.parquet"),
    ], ignore_index=True)
    emb["pand_id"] = emb["pand_id"].astype(str)
    emb = emb.drop_duplicates("pand_id")
    emb["y"] = emb["pand_id"].map(lab)

    folds = pd.read_parquet(splits / "dev_fold_indices.parquet")
    folds["pand_id"] = folds["pand_id"].astype(str)
    dev = emb.merge(folds[["pand_id", "fold"]], on="pand_id", how="inner")
    ho_ids = pd.read_parquet(splits / "holdout_test_pand_ids.parquet")
    ho = emb[emb["pand_id"].isin(set(ho_ids["pand_id"].astype(str)))].reset_index(drop=True)
    logger.info("LOCO dev=%d (of %d)  holdout=%d (of %d)  device=%s",
                len(dev), len(folds), len(ho), len(ho_ids), device)

    states, f1s = {}, {}
    for f in range(5):
        tr, va = dev[dev["fold"] != f], dev[dev["fold"] == f]
        f1, st = _train_head_cached(tr[EMB_COLS].values, tr["y"].values.astype(int),
                                    va[EMB_COLS].values, va["y"].values.astype(int), args, device)
        states[f], f1s[f] = st, f1
        logger.info("loco cached fold %d val energy macro-F1=%.4f", f, f1)

    best = max(f1s, key=f1s.get)
    logger.info("best fold %d (val f1=%.4f)", best, f1s[best])
    i2e = idx_to_energy(args.task)
    head = EnergyHead(n=n_energy_classes(args.task)).to(device)
    head.load_state_dict(states[best])
    head.eval()
    with torch.no_grad():
        logits = head(torch.tensor(ho[EMB_COLS].values, dtype=torch.float32, device=device))
        pred = logits.argmax(-1).cpu().numpy()
        proba = torch.softmax(logits.float(), dim=-1)[:, -1].cpu().numpy()

    preds = pd.DataFrame({
        "pand_id": ho["pand_id"].values,
        "true": [i2e[i] for i in ho["y"].values.astype(int)],
        "pred": [i2e[i] for i in pred],
        "proba": proba,
    })
    m1 = pd.read_parquet(REPORTS / f"M1{sfx}_{args.run_tag}_holdout_preds.parquet")
    common = set(m1["pand_id"].astype(str))
    preds = preds[preds["pand_id"].isin(common)].reset_index(drop=True)
    logger.info("restricted to M1 common set: n=%d", len(preds))

    rep = evaluate(preds[["true", "pred"]], with_ci=True, task=args.task,
                   proba=preds["proba"] if args.task == "binary" else None)
    rep["route"] = "M2-DINOv2-aligned-cached"
    rep["run_tag"] = args.run_tag
    rep["source_fold"] = int(best)
    rep["dev_val_macro_f1"] = float(f1s[best])
    rep["note"] = ("frozen DINOv2 cached embeddings + Stage1-style neural head; "
                   "LOCO split (train=source cities, test=held-out city)")
    preds.to_parquet(REPORTS / f"M2-DINOv2{sfx}_{args.run_tag}_holdout_preds.parquet", index=False)
    (REPORTS / f"M2-DINOv2{sfx}_{args.run_tag}_metrics.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(r):
        return json.load(open(REPORTS / f"{r}{sfx}_{args.run_tag}_metrics.json"))
    rows = [("M1 (GT)", load("M1")), ("M3-DINOv2 (decomp)", load("M3-DINOv2")),
            ("M2-DINOv2 (cached)", rep)]
    print(f"\n{'route':22s} {'macroF1':>8s} {'kappa':>7s} {'acc':>7s}" +
          ("  roc_auc" if args.task == "binary" else ""))
    for nm, r in rows:
        line = f"{nm:22s} {r['macro_f1']:8.4f} {r['quadratic_kappa']:7.4f} {r['accuracy']:7.4f}"
        if args.task == "binary" and r.get("roc_auc") is not None:
            line += f"  {r['roc_auc']:7.4f}"
        print(line)


if __name__ == "__main__":
    main()
