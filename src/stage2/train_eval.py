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
    BINARY_POSITIVE,
    REPO_ROOT,
    build_master_table,
    feature_matrix,
    target_col,
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


def params_for(task: str) -> dict:
    """Same fixed HP for both tasks; only the LightGBM objective differs."""
    p = dict(FIXED_PARAMS)
    if task == "binary":
        p["objective"] = "binary"
    return p


def task_suffix(task: str) -> str:
    return "" if task == "7class" else f"_{task}"


def train_oof(master: pd.DataFrame, run: str, task: str = "7class") -> pd.DataFrame:
    """5-fold OOF predictions for one run.

    Returns df[pand_id, fold, true, pred] (+ `proba` = P(D-G) for the binary task).
    """
    X_all, cat_features = feature_matrix(master, run)
    y_all = master[target_col(task)]
    folds = master["fold"].to_numpy()

    oof_pred = np.empty(len(master), dtype=object)
    oof_proba = np.full(len(master), np.nan)
    for f in range(N_FOLDS):
        tr = folds != f
        va = folds == f
        clf = LGBMClassifier(**params_for(task))
        clf.fit(X_all[tr], y_all[tr], categorical_feature=cat_features or "auto")
        oof_pred[va] = clf.predict(X_all[va])
        if task == "binary":
            pos = list(clf.classes_).index(BINARY_POSITIVE)
            oof_proba[va] = clf.predict_proba(X_all[va])[:, pos]
        logger.info("[%s] fold %d: train=%d val=%d", run, f, tr.sum(), va.sum())

    out = pd.DataFrame({
        "pand_id": master["pand_id"].values,
        "fold": folds,
        "true": y_all.values,
        "pred": oof_pred,
    })
    if task == "binary":
        out["proba"] = oof_proba
    return out


def majority_oof(master: pd.DataFrame, task: str = "7class") -> pd.DataFrame:
    """M0 baseline: predict each training fold's most-frequent class."""
    folds = master["fold"].to_numpy()
    y_all = master[target_col(task)]
    oof_pred = np.empty(len(master), dtype=object)
    oof_proba = np.full(len(master), np.nan)
    for f in range(N_FOLDS):
        tr = folds != f
        maj = y_all[tr].value_counts().idxmax()
        oof_pred[folds == f] = maj
        # A constant baseline has no ranking information: a constant score gives
        # AUC 0.5 by definition, which is the honest reading.
        oof_proba[folds == f] = (y_all[tr] == BINARY_POSITIVE).mean()
    out = pd.DataFrame({
        "pand_id": master["pand_id"].values,
        "fold": folds,
        "true": y_all.values,
        "pred": oof_pred,
    })
    if task == "binary":
        out["proba"] = oof_proba
    return out


def run_single(run: str, master: pd.DataFrame | None = None,
               with_ci: bool = True, save: bool = True,
               task: str = "7class") -> dict:
    master = build_master_table() if master is None else master
    oof = (majority_oof(master, task) if run == "M0"
           else train_oof(master, run, task))
    proba = oof["proba"] if "proba" in oof.columns else None
    report = evaluate(oof, with_ci=with_ci, task=task, proba=proba)
    report["run"] = run
    if save:
        sfx = task_suffix(task)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        oof.to_parquet(REPORTS_DIR / f"{run}{sfx}_oof_preds.parquet", index=False)
        (REPORTS_DIR / f"{run}{sfx}_metrics.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("[%s|%s] macro_f1=%.4f kappa=%.4f acc=%.4f -> %s",
                    run, task, report["macro_f1"], report["quadratic_kappa"],
                    report["accuracy"], REPORTS_DIR)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run name or M0")
    parser.add_argument("--task", default="7class", choices=["7class", "binary"])
    parser.add_argument("--no-ci", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run_single(args.run, with_ci=not args.no_ci, task=args.task)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("confusion_matrix",)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
