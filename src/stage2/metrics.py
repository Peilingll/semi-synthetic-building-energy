"""Stage 2 metrics: macro-F1, quadratic-weighted Cohen's kappa, per-class
breakdown, confusion matrix, and bootstrap 95% CI.

Operates on a DataFrame with columns `true` and `pred`. Two label spaces:

- task="7class"  A..G, the original ordinal task (quadratic kappa is meaningful)
- task="binary"  A-C | D-G, the Sun 2026 cut (decision 2026-08-10). Quadratic
  kappa degenerates to plain Cohen's kappa on two labels, so it is still
  reported under the same key for continuity, and MCC / balanced accuracy /
  ROC-AUC / PR-AUC are added. AUC needs a `proba` column (P(positive class)),
  which only the retrained binary models can supply.

The binary pool is ~70/30, so a majority baseline already scores acc 0.70:
accuracy must never be read without M0 alongside it.

Reuses the building-level bootstrap resampler from Stage 1.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.stage1.evaluate import bootstrap_ci
from src.stage2.features import BINARY_POSITIVE, ENERGY_LABELS, labels_for

logger = logging.getLogger(__name__)


def macro_f1(df: pd.DataFrame, labels: list[str] | None = None) -> float:
    return float(f1_score(df["true"], df["pred"],
                          labels=labels or ENERGY_LABELS, average="macro", zero_division=0))


def quadratic_kappa(df: pd.DataFrame, labels: list[str] | None = None) -> float:
    return float(cohen_kappa_score(df["true"], df["pred"],
                                   labels=labels or ENERGY_LABELS, weights="quadratic"))


def accuracy(df: pd.DataFrame) -> float:
    return float(accuracy_score(df["true"], df["pred"]))


def balanced_accuracy(df: pd.DataFrame) -> float:
    return float(balanced_accuracy_score(df["true"], df["pred"]))


def mcc(df: pd.DataFrame) -> float:
    return float(matthews_corrcoef(df["true"], df["pred"]))


def roc_auc(df: pd.DataFrame) -> float:
    y = (df["true"] == BINARY_POSITIVE).astype(int)
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, df["proba"]))


def pr_auc(df: pd.DataFrame) -> float:
    y = (df["true"] == BINARY_POSITIVE).astype(int)
    if y.nunique() < 2:
        return float("nan")
    return float(average_precision_score(y, df["proba"]))


def evaluate(df: pd.DataFrame, with_ci: bool = True, task: str = "7class",
             proba: np.ndarray | pd.Series | None = None) -> dict:
    """Full metric report for one run's predictions.

    proba = P(positive class) per row, binary task only; enables ROC-AUC/PR-AUC.
    """
    labels = labels_for(task)
    if proba is not None:
        df = df.copy()
        df["proba"] = np.asarray(proba, dtype=float)

    report: dict = {
        "task": task,
        "n_eval": int(len(df)),
        "macro_f1": round(macro_f1(df, labels), 4),
        "quadratic_kappa": round(quadratic_kappa(df, labels), 4),
        "accuracy": round(accuracy(df), 4),
    }
    if task == "binary":
        report["balanced_accuracy"] = round(balanced_accuracy(df), 4)
        report["mcc"] = round(mcc(df), 4)
        if "proba" in df.columns:
            report["roc_auc"] = round(roc_auc(df), 4)
            report["pr_auc"] = round(pr_auc(df), 4)

    precision, recall, f1, support = precision_recall_fscore_support(
        df["true"], df["pred"], labels=labels, zero_division=0,
    )
    report["per_class"] = {
        label: {"precision": round(float(p), 4), "recall": round(float(r), 4),
                "f1": round(float(f), 4), "support": int(s)}
        for label, p, r, f, s in zip(labels, precision, recall, f1, support)
    }

    cm = confusion_matrix(df["true"], df["pred"], labels=labels)
    report["confusion_matrix"] = {"labels": labels, "matrix": cm.tolist()}

    if with_ci:
        ci = {
            "macro_f1": bootstrap_ci(df, lambda d: macro_f1(d, labels)),
            "quadratic_kappa": bootstrap_ci(df, lambda d: quadratic_kappa(d, labels)),
        }
        if task == "binary":
            ci["balanced_accuracy"] = bootstrap_ci(df, balanced_accuracy)
            if "proba" in df.columns:
                ci["roc_auc"] = bootstrap_ci(df, roc_auc)
        report["bootstrap_95ci"] = ci
    return report
