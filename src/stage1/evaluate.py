"""Stage 1 evaluation: compute metrics from val_preds.parquet.

Outputs spec Table 1 columns (type Acc, type Macro-F1, year MAE, floors MAE)
plus auxiliary diagnostics (RMSE, ±10y/±20y, confusion matrix, per-class P/R).
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)

TYPE_LABELS = ["SFH", "TH", "MFH", "AB"]


def evaluate_predictions(preds: pd.DataFrame) -> dict:
    report: dict = {
        "n_eval": len(preds),
    }

    y_true_type = preds["true_type"].tolist()
    y_pred_type = preds["pred_type"].tolist()

    report["type_acc"] = round(accuracy_score(y_true_type, y_pred_type), 4)
    report["type_macro_f1"] = round(
        f1_score(y_true_type, y_pred_type, labels=TYPE_LABELS, average="macro", zero_division=0),
        4,
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_type, y_pred_type, labels=TYPE_LABELS, zero_division=0,
    )
    report["type_per_class"] = {
        label: {"precision": round(float(p), 4), "recall": round(float(r), 4),
                "f1": round(float(f), 4), "support": int(s)}
        for label, p, r, f, s in zip(TYPE_LABELS, precision, recall, f1, support)
    }
    cm = confusion_matrix(y_true_type, y_pred_type, labels=TYPE_LABELS)
    report["type_confusion_matrix"] = {
        "labels": TYPE_LABELS,
        "matrix": cm.tolist(),
    }

    year_err = (preds["pred_year"] - preds["true_bouwjaar"]).abs()
    year_sq = (preds["pred_year"] - preds["true_bouwjaar"]) ** 2
    report["year_mae"] = round(float(year_err.mean()), 2)
    report["year_rmse"] = round(float(np.sqrt(year_sq.mean())), 2)
    report["year_mdae"] = round(float(year_err.median()), 2)
    report["year_within_10y_pct"] = round(100 * float((year_err <= 10).mean()), 2)
    report["year_within_20y_pct"] = round(100 * float((year_err <= 20).mean()), 2)

    floors_pred_round = preds["pred_floors"].round()
    floors_err = (floors_pred_round - preds["true_num_floors"]).abs()
    report["floors_mae"] = round(float((preds["pred_floors"] - preds["true_num_floors"]).abs().mean()), 3)
    report["floors_exact_pct"] = round(100 * float((floors_err == 0).mean()), 2)
    report["floors_within_1_pct"] = round(100 * float((floors_err <= 1).mean()), 2)

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    preds = pd.read_parquet(args.preds)
    report = evaluate_predictions(preds)

    out_path = args.out if args.out else args.preds.with_name(args.preds.stem.replace("_val_preds", "_metrics") + ".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote %s", out_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
