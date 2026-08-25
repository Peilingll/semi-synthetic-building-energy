"""A01b — two fills requested 2026-08-24 (draft V9 Tables 4.9 / 4.10).

1) Binary counterparts of the clean-geometry additions in A01 T6: the same
   dev-pool 5-fold OOF LightGBM protocol (fixed HP), target = energy_binary
   (A-C | D-G), objective = binary. Runs: S_full base, + shape_factor,
   + floor_area_estimated, + all four clean 3DBAG features
   (shape_factor, floor_area_estimated, shared_ratio, volume).

2) L1 extractability probe for the CLEAN shape factor (3DBAG envelope/volume):
   RegHead on cached frozen DINOv2 embeddings, 5-fold OOF R2 — same head,
   folds and protocol as svi_compactheid.py, only the target differs
   (the audited-out EP `Compactheid` is replaced by the 3DBAG value).

Usage: .venv/Scripts/python.exe -m src.audit.a01b_geometry_binary_probe
Output: reports/tables/audit/A01b_geometry_binary_probe.md
"""

import logging

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.metrics import mean_absolute_error, r2_score

from src.audit.a01_compactheid_source import load_bag_geometry
from src.stage2.features import REPO_ROOT, build_master_table
from src.stage2.metrics import evaluate
from src.stage2.svi_compactheid import EMB_COLS, train_head_oof
from src.stage2.train_eval import FIXED_PARAMS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUT = REPO_ROOT / "reports" / "tables" / "audit" / "A01b_geometry_binary_probe.md"

BASE = ["building_type", "bouwjaar", "u_wall", "u_roof", "u_floor", "u_window",
        "num_floors", "city"]
CATS = ["building_type", "city"]
CLEAN4 = ["shape_factor", "floor_area_estimated", "shared_ratio", "volume"]


def main() -> None:
    geom = load_bag_geometry()[["pand_id", *CLEAN4]]

    mt = build_master_table()
    mt["pand_id"] = mt["pand_id"].astype(str).str.zfill(16)
    mt = mt.merge(geom, on="pand_id", how="left")
    folds = mt["fold"].to_numpy()
    yb = mt["energy_binary"]

    params = dict(FIXED_PARAMS)
    params["objective"] = "binary"

    def lgbm_oof_binary(cols):
        Xd = mt[cols].copy()
        for c in CATS:
            if c in cols:
                Xd[c] = Xd[c].astype("category")
        oof = np.empty(len(mt), dtype=object)
        for f in range(5):
            tr, va = folds != f, folds == f
            cl = LGBMClassifier(**params)
            cl.fit(Xd[tr], yb[tr], categorical_feature=[c for c in CATS if c in cols])
            oof[va] = cl.predict(Xd[va])
        return evaluate(pd.DataFrame({"true": yb.values, "pred": oof}),
                        with_ci=False, task="binary")

    L = ["# A01b — Binary clean-geometry additions + clean shape-factor probe", "",
         "Part 1: same protocol as A01 T6 (dev pool, 5-fold OOF LightGBM, fixed HP), "
         "task = binary (A-C | D-G, objective=binary, class_weight=None).", "",
         "| run | macro-F1 | accuracy | d macro-F1 |", "|---|---:|---:|---:|"]

    r0 = lgbm_oof_binary(BASE)
    logger.info("base (S_full, binary): mF1 %.4f", r0["macro_f1"])
    L += [f"| S_full base | {r0['macro_f1']:.4f} | {r0['accuracy']:.4f} | — |"]
    results = {"base": r0["macro_f1"]}
    for name, cols in [("+ shape_factor", BASE + ["shape_factor"]),
                       ("+ floor_area_estimated", BASE + ["floor_area_estimated"]),
                       ("+ all four clean 3DBAG", BASE + CLEAN4)]:
        r = lgbm_oof_binary(cols)
        results[name] = r["macro_f1"]
        logger.info("%s: mF1 %.4f (d %+.4f)", name, r["macro_f1"],
                    r["macro_f1"] - r0["macro_f1"])
        L += [f"| {name} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} | "
              f"{r['macro_f1'] - r0['macro_f1']:+.4f} |"]

    # ---- Part 2: probe ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = pd.read_parquet(REPO_ROOT / "reports" / "stage3" / "embeddings_dev.parquet")
    emb["pand_id"] = emb["pand_id"].astype(str).str.zfill(16)
    m = mt[["pand_id", "fold", "shape_factor"]].merge(emb, on="pand_id", how="inner")
    m = m.dropna(subset=["shape_factor"]).reset_index(drop=True)
    logger.info("probe: buildings with embedding + clean shape_factor: %d", len(m))

    X = m[EMB_COLS].to_numpy(dtype=np.float32)
    y = m["shape_factor"].to_numpy(dtype=np.float32)
    pred = train_head_oof(X, y, m["fold"].to_numpy(), device)
    r2, mae = r2_score(y, pred), mean_absolute_error(y, pred)
    logger.info("[L1 probe] frozen DINOv2 -> clean shape_factor  R2=%.3f  MAE=%.3f", r2, mae)

    L += ["", "Part 2: L1 extractability probe (RegHead on frozen DINOv2 embeddings, "
          "5-fold OOF; same protocol as svi_compactheid.py), target = 3DBAG "
          "shape_factor (envelope_area / volume, the audited clean variant).", "",
          f"| target | n | OOF R2 | MAE |", "|---|---:|---:|---:|",
          f"| clean shape_factor | {len(m)} | {r2:.3f} | {mae:.3f} |"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
