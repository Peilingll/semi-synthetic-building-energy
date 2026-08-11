"""Regression variant of Stage 3: predict continuous PrimaireFossieleEnergie
(kWh/m2.yr) with LightGBM, then bin back to A-G via official NTA8800 residential
boundaries. Compares against the direct-classification baseline.

Routes mirror Stage 3:
- M1-reg : GT attributes -> regress kWh -> bin A-G
- M3-reg : vision attributes -> regress kWh -> bin A-G  (DINOv2 / ResNet50 / VLMv3)

Reports regression MAE / R2 (kWh) AND the binned macro-F1 / kappa, on the same
hold-out common set as the classification Stage 3.

Usage: uv run python -m src.stage3.regression_kwh
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.stage2.features import (
    BINARY_CUT_KWH, BINARY_LABELS, CATEGORICAL, PROCESSED, REPO_ROOT, S_FULL,
    build_master_table, feature_matrix, to_binary,
)
from src.stage2.metrics import evaluate
from src.stage2.train_eval import task_suffix
from src.stage3.features import build_m1_holdout, build_m3_holdout
from src.stage3.run_stage3 import M3_PREDS, resolve_m3_preds

logger = logging.getLogger(__name__)
REPORTS = REPO_ROOT / "reports" / "stage3"
TABLES = REPO_ROOT / "reports" / "tables" / "stage3"

# Official NTA8800 residential upper bounds (kWh/m2.yr); A merges A+..A++++.
# label = first class whose upper bound >= pf.
PF_BINS = [("A", 160.0), ("B", 190.0), ("C", 250.0), ("D", 290.0),
           ("E", 335.0), ("F", 380.0), ("G", float("inf"))]
# The Sun cut sits exactly on the C boundary, so the binary task needs one
# threshold rather than a ladder.
PF_BINS_BINARY = [(BINARY_LABELS[0], BINARY_CUT_KWH), (BINARY_LABELS[1], float("inf"))]

REG_PARAMS = dict(
    objective="regression_l1", n_estimators=400, learning_rate=0.05, num_leaves=31,
    min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1,
)


def pf_to_label(pf: np.ndarray, task: str = "7class") -> np.ndarray:
    bins = PF_BINS if task == "7class" else PF_BINS_BINARY
    out = np.empty(len(pf), dtype=object)
    for i, v in enumerate(pf):
        for lab, ub in bins:
            if v <= ub:
                out[i] = lab
                break
    return out


def load_kwh() -> dict:
    ep = pd.read_parquet(PROCESSED / "ep_kwh.parquet")
    ep["pand_id"] = ep["pand_id"].astype(str)
    return dict(zip(ep["pand_id"], ep["pf_kwh"]))


def _cats(master):
    X, _ = feature_matrix(master, "S_full")
    return {c: X[c].cat.categories for c in CATEGORICAL if c in X.columns}


def _Xho(df, cat_dtypes):
    X = df[S_FULL].copy()
    for col, cats in cat_dtypes.items():
        X[col] = pd.Categorical(X[col], categories=cats)
    return X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="pooled")
    parser.add_argument("--dev-folds", type=Path, default=None)
    parser.add_argument("--holdout", type=Path, default=None)
    parser.add_argument("--models", default=",".join(M3_PREDS))
    parser.add_argument("--out-suffix", default="")
    parser.add_argument("--task", default="7class", choices=["7class", "binary"])
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tag = task_suffix(args.task)
    tag += "" if args.run_tag == "pooled" else f"_{args.run_tag}"
    tag += args.out_suffix

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    kwh = load_kwh()

    # --- train regressor on dev GT features -> pf_kwh ---
    master = build_master_table(dev_path=args.dev_folds)
    master["pf"] = master["pand_id"].astype(str).map(kwh)
    n0 = len(master)
    master = master[master["pf"].notna()].reset_index(drop=True)
    logger.info("dev with kWh: %d / %d", len(master), n0)

    X, cat = feature_matrix(master, "S_full")
    reg = LGBMRegressor(**REG_PARAMS)
    reg.fit(X, master["pf"], categorical_feature=cat)
    cat_dtypes = _cats(master)

    # --- common hold-out set (same as classification Stage 3) + true labels/kWh ---
    m1 = build_m1_holdout(args.holdout)
    m1["pf"] = m1["pand_id"].astype(str).map(kwh)
    common = set(pd.read_parquet(REPORTS / f"M1{tag}_holdout_preds.parquet")["pand_id"].astype(str))

    def reg_eval(df, name):
        d = df[df["pand_id"].astype(str).isin(common) & df["pf"].notna()].copy()
        pred_pf = reg.predict(_Xho(d, cat_dtypes))
        pred_lab = pf_to_label(np.asarray(pred_pf), args.task)
        true_series = d["energy_class"]
        true_lab = (to_binary(true_series) if args.task == "binary" else true_series).values
        mae = mean_absolute_error(d["pf"], pred_pf)
        r2 = r2_score(d["pf"], pred_pf)
        # A regressed kWh IS the score, so it doubles as the ranking for AUC:
        # higher predicted demand = more likely D-G.
        rep = evaluate(pd.DataFrame({"true": true_lab, "pred": pred_lab}), with_ci=True,
                       task=args.task,
                       proba=np.asarray(pred_pf) if args.task == "binary" else None)
        rep.update({"route": name, "n": int(len(d)), "kwh_mae": round(float(mae), 2),
                    "kwh_r2": round(float(r2), 4)})
        (REPORTS / f"{name}{tag}_metrics.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("%s: MAE=%.1f R2=%.3f macroF1=%.4f kappa=%.4f acc=%.4f",
                    name, mae, r2, rep["macro_f1"], rep["quadratic_kappa"], rep["accuracy"])
        return rep

    results = {"M1-reg": reg_eval(m1, "M1-reg")}
    for route, path in resolve_m3_preds(args.run_tag, models).items():
        d = build_m3_holdout(path, args.holdout)
        d["energy_class"] = d["pand_id"].astype(str).map(
            dict(zip(m1["pand_id"].astype(str), m1["energy_class"])))
        d["pf"] = d["pand_id"].astype(str).map(kwh)
        results[f"{route}-reg"] = reg_eval(d, f"{route}-reg")

    # --- table: regression vs classification ---
    def load_cls(n):
        return json.load(open(REPORTS / f"{n}{tag}_metrics.json"))
    cls = {n: load_cls(n) for n in ["M1"] + models}

    split_name = "hold-out" if args.run_tag == "pooled" else args.run_tag
    L = [f"# Table 3-reg — regression-to-kWh vs direct classification ({split_name})",
         "",
         "| Route | kWh MAE | kWh R² | macro-F1 (reg→bin) | κ (reg→bin) | macro-F1 (cls) | κ (cls) |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    pairs = [("M1-reg", "M1")] + [(f"{m}-reg", m) for m in models]
    for rn, cn in pairs:
        r, c = results[rn], cls[cn]
        L.append(f"| {rn} | {r['kwh_mae']:.1f} | {r['kwh_r2']:.3f} | {r['macro_f1']:.4f} | "
                 f"{r['quadratic_kappa']:.4f} | {c['macro_f1']:.4f} | {c['quadratic_kappa']:.4f} |")
    boundary = ("official NTA8800 residential (A≤160 B≤190 C≤250 D≤290 E≤335 F≤380 "
                "G>380 kWh/m²·yr)" if args.task == "7class" else
                f"one threshold at the C boundary, {BINARY_CUT_KWH:.0f} kWh/m²·yr")
    L += ["", f"Boundaries: {boundary}.",
          "Regression objective = L1 (MAE) on the registered primary fossil energy "
          "(`PrimaireFossieleEnergieEMGForfaitair` with fallback — audit A04)."]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"T3reg_regression_vs_classification{tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    for rn, cn in pairs:
        r, c = results[rn], cls[cn]
        print(f"{rn:18s} MAE={r['kwh_mae']:6.1f} R2={r['kwh_r2']:+.3f} "
              f"reg->bin mF1={r['macro_f1']:.4f} k={r['quadratic_kappa']:.4f} | "
              f"cls mF1={c['macro_f1']:.4f} k={c['quadratic_kappa']:.4f}")


if __name__ == "__main__":
    main()
