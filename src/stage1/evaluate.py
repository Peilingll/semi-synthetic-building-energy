"""Stage 1 evaluation: spec Table 1 metrics plus bootstrap CI and slicing.

Reads one or more `*_val_preds.parquet` files (per fold) and produces:
- main metrics with 95% bootstrap CI (type_acc, macro_f1, year_mae, floors_mae)
- per-class precision/recall/f1
- per-city slice (acc/macro_f1/year_mae/floors_mae)
- confusion matrix

Usage:
    # single fold metrics
    python -m src.stage1.evaluate --preds reports/stage1/pooled_dinov2_frozen_fold0_val_preds.parquet
    # aggregate across all 5 folds (concat val_preds, treat as one pooled-CV report)
    python -m src.stage1.evaluate --aggregate "reports/stage1/pooled_dinov2_frozen_fold*_val_preds.parquet"
"""

import argparse
import glob
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
N_BOOTSTRAP = 1000
CI = 0.95


def type_macro_f1(df: pd.DataFrame) -> float:
    return float(f1_score(df["true_type"], df["pred_type"],
                          labels=TYPE_LABELS, average="macro", zero_division=0))


def type_accuracy(df: pd.DataFrame) -> float:
    return float(accuracy_score(df["true_type"], df["pred_type"]))


def year_mae(df: pd.DataFrame) -> float:
    return float((df["pred_year"] - df["true_bouwjaar"]).abs().mean())


def floors_mae(df: pd.DataFrame) -> float:
    return float((df["pred_floors"] - df["true_num_floors"]).abs().mean())


def bootstrap_ci(
    df: pd.DataFrame,
    metric_fn,
    n_bootstrap: int = N_BOOTSTRAP,
    ci: float = CI,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(df)
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        boots.append(metric_fn(df.iloc[idx]))
    boots = np.array(boots, dtype=float)
    boots = boots[~np.isnan(boots)]
    if len(boots) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_boot_valid": 0}
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return {
        "mean": float(np.mean(boots)),
        "lo": round(lo, 4),
        "hi": round(hi, 4),
        "n_boot_valid": int(len(boots)),
    }


def evaluate_predictions(preds: pd.DataFrame, with_ci: bool = True) -> dict:
    report: dict = {
        "n_eval": len(preds),
    }

    report["type_acc"] = round(type_accuracy(preds), 4)
    report["type_macro_f1"] = round(type_macro_f1(preds), 4)

    precision, recall, f1, support = precision_recall_fscore_support(
        preds["true_type"], preds["pred_type"], labels=TYPE_LABELS, zero_division=0,
    )
    report["type_per_class"] = {
        label: {"precision": round(float(p), 4), "recall": round(float(r), 4),
                "f1": round(float(f), 4), "support": int(s)}
        for label, p, r, f, s in zip(TYPE_LABELS, precision, recall, f1, support)
    }
    cm = confusion_matrix(preds["true_type"], preds["pred_type"], labels=TYPE_LABELS)
    report["type_confusion_matrix"] = {"labels": TYPE_LABELS, "matrix": cm.tolist()}

    year_err = (preds["pred_year"] - preds["true_bouwjaar"]).abs()
    year_sq = (preds["pred_year"] - preds["true_bouwjaar"]) ** 2
    report["year_mae"] = round(float(year_err.mean()), 2)
    report["year_rmse"] = round(float(np.sqrt(year_sq.mean())), 2)
    report["year_mdae"] = round(float(year_err.median()), 2)
    report["year_within_10y_pct"] = round(100 * float((year_err <= 10).mean()), 2)
    report["year_within_20y_pct"] = round(100 * float((year_err <= 20).mean()), 2)

    floors_pred_round = preds["pred_floors"].round()
    floors_err_round = (floors_pred_round - preds["true_num_floors"]).abs()
    floors_err = (preds["pred_floors"] - preds["true_num_floors"]).abs()
    report["floors_mae"] = round(float(floors_err.mean()), 3)
    report["floors_exact_pct"] = round(100 * float((floors_err_round == 0).mean()), 2)
    report["floors_within_1_pct"] = round(100 * float((floors_err_round <= 1).mean()), 2)

    if "pred_period" in preds.columns and "true_tabula_period" in preds.columns:
        period_valid = preds[["pred_period", "true_tabula_period"]].dropna()
        if len(period_valid):
            period_correct = (period_valid["pred_period"] == period_valid["true_tabula_period"]).mean()
            report["period_acc"] = round(float(period_correct), 4)
            report["period_n_eval"] = int(len(period_valid))

    if with_ci:
        report["bootstrap_95ci"] = {
            "type_acc": bootstrap_ci(preds, type_accuracy),
            "type_macro_f1": bootstrap_ci(preds, type_macro_f1),
            "year_mae": bootstrap_ci(preds, year_mae),
            "floors_mae": bootstrap_ci(preds, floors_mae),
        }

    return report


def per_city_breakdown(preds: pd.DataFrame) -> dict:
    if "city" not in preds.columns:
        return {}
    out: dict = {}
    for city, grp in preds.groupby("city"):
        out[city] = {
            "n": int(len(grp)),
            "type_acc": round(type_accuracy(grp), 4),
            "type_macro_f1": round(type_macro_f1(grp), 4),
            "year_mae": round(year_mae(grp), 2),
            "floors_mae": round(floors_mae(grp), 3),
        }
    return out


def per_class_year_floor_breakdown(preds: pd.DataFrame) -> dict:
    """For each true_type class, report year/floors MAE separately."""
    out: dict = {}
    for cls in TYPE_LABELS:
        grp = preds[preds["true_type"] == cls]
        if len(grp) == 0:
            out[cls] = {"n": 0}
            continue
        out[cls] = {
            "n": int(len(grp)),
            "year_mae": round(year_mae(grp), 2),
            "floors_mae": round(floors_mae(grp), 3),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", type=Path, default=None, help="single val_preds.parquet")
    parser.add_argument("--aggregate", type=str, default=None,
                        help="glob pattern, e.g. 'reports/stage1/pooled_dinov2_frozen_fold*_val_preds.parquet'")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--no-ci", action="store_true", help="skip bootstrap CI (faster)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.aggregate:
        files = sorted(glob.glob(args.aggregate))
        if not files:
            raise SystemExit(f"no files matched: {args.aggregate}")
        logger.info("aggregating %d files: %s", len(files), files)
        dfs = [pd.read_parquet(f) for f in files]
        preds = pd.concat(dfs, ignore_index=True)
        default_out_name = "aggregate_metrics.json"
    elif args.preds:
        preds = pd.read_parquet(args.preds)
        default_out_name = args.preds.stem.replace("_val_preds", "_metrics") + ".json"
    else:
        raise SystemExit("must pass --preds or --aggregate")

    report = evaluate_predictions(preds, with_ci=not args.no_ci)
    report["per_city"] = per_city_breakdown(preds)
    report["per_class_year_floors"] = per_class_year_floor_breakdown(preds)

    out_path = args.out if args.out else (
        (args.preds.parent if args.preds else Path("reports/stage1")) / default_out_name
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote %s", out_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
