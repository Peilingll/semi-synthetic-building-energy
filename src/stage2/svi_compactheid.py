"""Can SVI extract compactheid? Train a regression head on cached frozen DINOv2
embeddings to predict compactheid (BAG surface/volume ratio), 5-fold OOF.

Two-level evaluation:
  1) extraction quality: predicted vs GT compactheid (R2 / MAE)
  2) downstream usefulness: feed PREDICTED compactheid into the energy LightGBM
     and compare macro-F1/kappa to (a) S_full and (b) S_full + GT compactheid.

Reuses reports/stage3/embeddings_dev.parquet (768-d frozen DINOv2, mean-pooled).
"""

import csv
import logging

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.metrics import mean_absolute_error, r2_score

from src.stage2.features import REPO_ROOT, build_master_table
from src.stage2.metrics import evaluate
from src.stage2.train_eval import FIXED_PARAMS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

EMB_COLS = [f"e{i}" for i in range(768)]
RAW_EP = REPO_ROOT / "data" / "raw" / "v20260401_v4_csv" / "v20260401_v4_csv.csv"


def _num(x):
    x = (x or "").strip().replace(",", ".")
    try:
        return float(x)
    except ValueError:
        return np.nan


def load_compactheid() -> dict:
    rows = []
    with open(RAW_EP, encoding="utf-8-sig") as f:
        f.readline(); f.readline()
        h = f.readline().rstrip("\n").split(";")
        idx = {c: i for i, c in enumerate(h)}
        iP, iC, iReg = idx["BAGPandIDs"], idx["Compactheid"], idx["Registratiedatum"]
        for row in csv.reader(f, delimiter=";"):
            if len(row) <= max(iP, iC, iReg):
                continue
            rows.append((str(row[iP]).split(",")[0].zfill(16), _num(row[iC]), row[iReg]))
    ep = pd.DataFrame(rows, columns=["pand_id", "compactheid", "reg"]).sort_values("reg").drop_duplicates("pand_id", keep="last")
    ep = ep[(ep["compactheid"] >= 0.3) & (ep["compactheid"] <= 5)]
    return dict(zip(ep["pand_id"], ep["compactheid"]))


class RegHead(torch.nn.Module):
    """Same trunk shape as Stage 1 DINOv2 head, single regression output."""
    def __init__(self, in_dim=768, hidden=256, dropout=0.3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_head_oof(X, y, folds, device, epochs=80, lr=1e-3):
    pred = np.empty(len(X))
    for f in range(5):
        tr, va = folds != f, folds == f
        mu, sd = y[tr].mean(), y[tr].std() + 1e-6
        head = RegHead().to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
        Xtr = torch.tensor(X[tr], dtype=torch.float32, device=device)
        ytr = torch.tensor((y[tr] - mu) / sd, dtype=torch.float32, device=device)
        Xva = torch.tensor(X[va], dtype=torch.float32, device=device)
        n, bs = len(Xtr), 256
        for ep in range(epochs):
            head.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                torch.nn.functional.smooth_l1_loss(head(Xtr[idx]), ytr[idx]).backward()
                opt.step()
        head.eval()
        with torch.no_grad():
            pred[va] = head(Xva).cpu().numpy() * sd + mu
    return pred


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    comp = load_compactheid()
    m = build_master_table(); m["pand_id"] = m["pand_id"].astype(str)
    emb = pd.read_parquet(REPORTS := REPO_ROOT / "reports" / "stage3" / "embeddings_dev.parquet")
    emb["pand_id"] = emb["pand_id"].astype(str)
    m = m.merge(emb, on="pand_id", how="inner")
    m["compactheid"] = m["pand_id"].map(comp)
    m = m.dropna(subset=["compactheid"]).reset_index(drop=True)
    logger.info("buildings with embedding + compactheid: %d", len(m))

    folds = m["fold"].to_numpy()
    X = m[EMB_COLS].to_numpy(dtype=np.float32)
    yC = m["compactheid"].to_numpy(dtype=np.float32)

    # --- Level 1: extraction quality ---
    pred = train_head_oof(X, yC, folds, device)
    r2, mae = r2_score(yC, pred), mean_absolute_error(yC, pred)
    logger.info("\n[L1 extraction] SVI->compactheid  R2=%.3f  MAE=%.3f  (range %.2f-%.2f)",
                r2, mae, np.percentile(yC, 5), np.percentile(yC, 95))
    m["pred_compactheid"] = pred

    # --- Level 2: downstream usefulness ---
    y = m["energy_class"]
    base = ["building_type", "bouwjaar", "u_wall", "u_roof", "u_floor", "u_window", "num_floors", "city"]
    cats = ["building_type", "city"]

    def lgbm_oof(cols):
        Xd = m[cols].copy()
        for c in cats:
            Xd[c] = Xd[c].astype("category")
        oof = np.empty(len(m), dtype=object)
        for f in range(5):
            tr, va = folds != f, folds == f
            cl = LGBMClassifier(**FIXED_PARAMS)
            cl.fit(Xd[tr], y[tr], categorical_feature=[c for c in cats if c in cols])
            oof[va] = cl.predict(Xd[va])
        return evaluate(pd.DataFrame({"true": y.values, "pred": oof}), with_ci=False)

    r_base = lgbm_oof(base)
    r_pred = lgbm_oof(base + ["pred_compactheid"])
    r_gt = lgbm_oof(base + ["compactheid"])
    logger.info("\n[L2 downstream energy] (n=%d)", len(m))
    for n, r in [("S_full (base)", r_base),
                 ("+ SVI-pred compactheid", r_pred),
                 ("+ GT compactheid (ceiling)", r_gt)]:
        logger.info("  %-28s macroF1=%.4f kappa=%.4f acc=%.4f",
                    n, r["macro_f1"], r["quadratic_kappa"], r["accuracy"])


if __name__ == "__main__":
    main()
