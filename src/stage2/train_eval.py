"""Stage 2 training + evaluation: 5-fold out-of-fold (OOF) LightGBM.

For one ablation run, trains a fixed-HP LightGBM on 4 folds and predicts the
held-out fold, looping over all 5 folds, then stitches the predictions back
into one 8,068-row OOF table for metric computation.

LightGBM HP are FIXED across all runs (no per-run tuning) so that differences
between runs reflect feature content only. class_weight=None (decision locked
2026-06-17): quadratic kappa is the ordinal metric and balancing hurts it.

Usage:
    uv run python -m src.stage2.train_eval --run S_full
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.stage2.features import (
    REPO_ROOT,
    build_master_table,
    feature_matrix,
)
from src.stage2.metrics import evaluate

logger = logging.getLogger(__name__)

REPORTS_DIR = REPO_ROOT / "reports" / "stage2"

FIXED_PARAMS = dict(
    objective="multiclass",
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    class_weight=None,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

N_FOLDS = 5


def train_oof(master: pd.DataFrame, run: str) -> pd.DataFrame:
    """5-fold OOF predictions for one run. Returns df[pand_id, fold, true, pred]."""
    X_all, cat_features = feature_matrix(master, run)
    y_all = master["energy_class"]
    folds = master["fold"].to_numpy()

    oof_pred = np.empty(len(master), dtype=object)
    for f in range(N_FOLDS):
        tr = folds != f
        va = folds == f
        clf = LGBMClassifier(**FIXED_PARAMS)
        clf.fit(X_all[tr], y_all[tr], categorical_feature=cat_features or "auto")
        oof_pred[va] = clf.predict(X_all[va])
        logger.info("[%s] fold %d: train=%d val=%d", run, f, tr.sum(), va.sum())

    return pd.DataFrame({
        "pand_id": master["pand_id"].values,
        "fold": folds,
        "true": y_all.values,
        "pred": oof_pred,
    })


def majority_oof(master: pd.DataFrame) -> pd.DataFrame:
    """M0 baseline: predict each training fold's most-frequent class."""
    folds = master["fold"].to_numpy()
    y_all = master["energy_class"]
    oof_pred = np.empty(len(master), dtype=object)
    for f in range(N_FOLDS):
        tr = folds != f
        maj = y_all[tr].value_counts().idxmax()
        oof_pred[folds == f] = maj
    return pd.DataFrame({
        "pand_id": master["pand_id"].values,
        "fold": folds,
        "true": y_all.values,
        "pred": oof_pred,
    })


def run_single(run: str, master: pd.DataFrame | None = None,
               with_ci: bool = True, save: bool = True) -> dict:
    master = build_master_table() if master is None else master
    oof = majority_oof(master) if run == "M0" else train_oof(master, run)
    report = evaluate(oof, with_ci=with_ci)
    report["run"] = run
    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        oof.to_parquet(REPORTS_DIR / f"{run}_oof_preds.parquet", index=False)
        (REPORTS_DIR / f"{run}_metrics.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("[%s] macro_f1=%.4f kappa=%.4f acc=%.4f → %s",
                    run, report["macro_f1"], report["quadratic_kappa"],
                    report["accuracy"], REPORTS_DIR)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run name or M0")
    parser.add_argument("--no-ci", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run_single(args.run, with_ci=not args.no_ci)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("confusion_matrix",)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
