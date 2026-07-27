"""Measure what the 2026-07-27 TABULA correction changed.

Diffs the superseded hand-made table (`legacy/tabula_nl_handmade.csv`) against
the regenerated canonical one (`tabula_nl.csv`, see `build_lookup.py`).

The U-values are a deterministic function of (building_type, tabula_period), and
both of those are already model features, so a tree model should be indifferent
to their numeric level. This script checks that claim instead of assuming it,
and quantifies the change for the physical (H_tr) use where it does matter.

Run: .venv/Scripts/python.exe -m src.tabula.impact_check
"""

import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

import src.stage2.features as F
from src.stage2.features import PROCESSED, build_master_table
from src.stage2.metrics import evaluate
from src.stage2.train_eval import FIXED_PARAMS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OLD = PROCESSED / "legacy" / "tabula_nl_handmade.csv"
NEW = PROCESSED / "tabula_nl.csv"


def run_ml(tabula_path):
    m = build_master_table(tabula_path=tabula_path)
    folds, y = m["fold"].to_numpy(), m["energy_class"]
    out = {}
    for name, cols in [("S_lookup", F.S_LOOKUP), ("S_full", F.S_FULL)]:
        X = m[cols].copy()
        cats = [c for c in F.CATEGORICAL if c in cols]
        for c in cats:
            X[c] = X[c].astype("category")
        oof = np.empty(len(m), dtype=object)
        for f in range(5):
            tr, va = folds != f, folds == f
            cl = LGBMClassifier(**FIXED_PARAMS)
            cl.fit(X[tr], y[tr], categorical_feature=cats)
            oof[va] = cl.predict(X[va])
        out[name] = evaluate(pd.DataFrame({"true": y.values, "pred": oof}), with_ci=False)
    return out


def run_physical(tabula_path):
    """Median specific transmission loss per cell, and stock total."""
    import src.audit.a06_archetype_vs_measured as a6
    tab = pd.read_csv(tabula_path)
    tab["key"] = tab["building_type"] + "|" + tab["period"]
    a6.U = tab.set_index("key")[["u_wall", "u_roof", "u_floor", "u_window"]]
    geo = a6.load_geometry()
    gt = pd.read_parquet(PROCESSED / "stage1_gt.parquet")
    gt["pand_id"] = gt["pand_id"].astype(str).str.zfill(16)
    gt["cell"] = gt["building_type"].astype(str) + "|" + gt["tabula_period"].astype(str)
    d = gt[["pand_id", "cell"]].merge(geo, on="pand_id", how="inner")
    h = a6.h_tr(d["cell"].to_numpy(), d)
    ok = np.isfinite(h)
    return dict(median_h=float(np.median(h[ok])),
                stock_wk=float((h[ok] * d["floor_area_estimated"].to_numpy()[ok]).sum()))


def main():
    a, b = run_ml(OLD), run_ml(NEW)
    logger.info("\n%-10s %14s %14s %9s | %11s %9s | %9s %9s",
                "run", "macroF1 old", "macroF1 new", "delta",
                "kappa old", "new", "acc old", "new")
    for k in a:
        logger.info("%-10s %14.4f %14.4f %+9.4f | %11.4f %9.4f | %9.4f %9.4f",
                    k, a[k]["macro_f1"], b[k]["macro_f1"],
                    b[k]["macro_f1"] - a[k]["macro_f1"],
                    a[k]["quadratic_kappa"], b[k]["quadratic_kappa"],
                    a[k]["accuracy"], b[k]["accuracy"])

    pa, pb = run_physical(OLD), run_physical(NEW)
    logger.info("\nphysical use (H_tr over the four-city stock):")
    logger.info("  median H_tr   %.3f -> %.3f W/(K.m2)   (%.1f%%)",
                pa["median_h"], pb["median_h"],
                100 * (pb["median_h"] / pa["median_h"] - 1))
    logger.info("  stock total   %.3e -> %.3e W/K        (%.1f%%)",
                pa["stock_wk"], pb["stock_wk"],
                100 * (pb["stock_wk"] / pa["stock_wk"] - 1))


if __name__ == "__main__":
    main()
