"""Stage 2 metrics: macro-F1, quadratic-weighted Cohen's kappa, per-class
breakdown, confusion matrix, and bootstrap 95% CI.

Operates on a DataFrame with columns `true` and `pred` (energy classes A..G).
Reuses the building-level bootstrap resampler from Stage 1.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.stage1.evaluate import bootstrap_ci
from src.stage2.features import ENERGY_LABELS

logger = logging.getLogger(__name__)


def macro_f1(df: pd.DataFrame) -> float:
    return float(f1_score(df["true"], df["pred"],
                          labels=ENERGY_LABELS, average="macro", zero_division=0))


def quadratic_kappa(df: pd.DataFrame) -> float:
    return float(cohen_kappa_score(df["true"], df["pred"],
                                   labels=ENERGY_LABELS, weights="quadratic"))


def accuracy(df: pd.DataFrame) -> float:
    return float(accuracy_score(df["true"], df["pred"]))


def evaluate(df: pd.DataFrame, with_ci: bool = True) -> dict:
    """Full Stage 2 metric report for one run's OOF predictions."""
    report: dict = {
        "n_eval": int(len(df)),
        "macro_f1": round(macro_f1(df), 4),
        "quadratic_kappa": round(quadratic_kappa(df), 4),
        "accuracy": round(accuracy(df), 4),
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        df["true"], df["pred"], labels=ENERGY_LABELS, zero_division=0,
    )
    report["per_class"] = {
        label: {"precision": round(float(p), 4), "recall": round(float(r), 4),
                "f1": round(float(f), 4), "support": int(s)}
        for label, p, r, f, s in zip(ENERGY_LABELS, precision, recall, f1, support)
    }

    cm = confusion_matrix(df["true"], df["pred"], labels=ENERGY_LABELS)
    report["confusion_matrix"] = {"labels": ENERGY_LABELS, "matrix": cm.tolist()}

    if with_ci:
        report["bootstrap_95ci"] = {
            "macro_f1": bootstrap_ci(df, macro_f1),
            "quadratic_kappa": bootstrap_ci(df, quadratic_kappa),
        }
    return report
