"""Stage 3 orchestration: M0 / M1 / M3-{DINOv2,ResNet50,VLMv3} on the hold-out,
the M3-M1 gap (RQ2), and error-propagation traceback for the best M3.

Usage:
    uv run python -m src.stage3.run_stage3
"""

import json
import logging
from pathlib import Path

import pandas as pd

from src.stage2.features import REPO_ROOT
from src.stage2.metrics import evaluate
from src.stage3.features import build_m1_holdout, build_m3_holdout
from src.stage3.routes import predict_route, train_m1_model
from src.tabula_matcher import classify_period

logger = logging.getLogger(__name__)

REPORTS = REPO_ROOT / "reports" / "stage3"
TABLES = REPO_ROOT / "reports" / "tables" / "stage3"

M3_PREDS = {
    "M3-DINOv2": "reports/stage1/dinov2_frozen/holdout_preds.parquet",
    "M3-ResNet50": "reports/stage1/resnet50_ft/holdout_preds.parquet",
    "M3-VLMv3": "reports/stage1/vlm_internvl3/v3_holdout_per_pand_id.parquet",
}


def error_propagation(m3df: pd.DataFrame, pred_label: pd.Series, true_label: pd.Series) -> dict:
    """For misclassified M3 buildings, attribute the error to vision vs downstream."""
    import numpy as np
    mis = pred_label.to_numpy() != true_label.to_numpy()
    sub = m3df[mis]
    type_wrong = sub["building_type"].astype(str).to_numpy() != sub["true_type"].astype(str).to_numpy()
    pred_period = sub["bouwjaar"].apply(classify_period).astype(str).to_numpy()
    true_period = sub["true_bouwjaar"].apply(classify_period).astype(str).to_numpy()
    period_wrong = pred_period != true_period
    attr_error = type_wrong | period_wrong
    return {
        "n_total": int(len(m3df)),
        "n_misclassified": int(mis.sum()),
        "type_wrong": int(type_wrong.sum()),
        "period_wrong": int(period_wrong.sum()),
        "both_wrong": int((type_wrong & period_wrong).sum()),
        "vision_attr_error": int(attr_error.sum()),
        "attrs_correct_downstream_error": int((~attr_error).sum()),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    REPORTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    clf, cat_dtypes, majority = train_m1_model()

    # Build each route's hold-out frame (pand_id, true label, pred label).
    m1 = build_m1_holdout()
    m1_pred = predict_route(clf, m1, cat_dtypes)

    m3_frames, m3_preds = {}, {}
    for name, path in M3_PREDS.items():
        df = build_m3_holdout(path)
        m3_frames[name] = df
        m3_preds[name] = predict_route(clf, df, cat_dtypes)

    # Common building set across all routes for strictly comparable M3-M1.
    common = set(m1["pand_id"])
    for df in m3_frames.values():
        common &= set(df["pand_id"])
    logger.info("common hold-out buildings across all routes: %d", len(common))

    def route_df(frame, pred):
        m = frame["pand_id"].isin(common)
        return pd.DataFrame({
            "pand_id": frame.loc[m, "pand_id"].values,
            "true": frame.loc[m, "energy_class"].values,
            "pred": pred[m].values,
        }).sort_values("pand_id").reset_index(drop=True)

    routes = {}
    routes["M1"] = route_df(m1, m1_pred)
    for name in M3_PREDS:
        routes[name] = route_df(m3_frames[name], m3_preds[name])
    # M0: dev-majority for every common building (use M1's true labels/ids).
    routes["M0"] = routes["M1"].assign(pred=majority)

    # Evaluate.
    reports = {}
    for name, df in routes.items():
        rep = evaluate(df, with_ci=True)
        rep["route"] = name
        reports[name] = rep
        df.to_parquet(REPORTS / f"{name}_holdout_preds.parquet", index=False)
        (REPORTS / f"{name}_metrics.json").write_text(
            json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    m1_f1 = reports["M1"]["macro_f1"]
    m1_k = reports["M1"]["quadratic_kappa"]

    # Table 3.
    order = ["M0", "M1", "M3-DINOv2", "M3-ResNet50", "M3-VLMv3"]
    lines = [
        f"# Table 3 — Stage 3 pipeline comparison (hold-out, n={len(common)})",
        "",
        "| Route | macro-F1 | 95% CI | κ | acc | M3−M1 mF1 | M3−M1 κ |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for name in order:
        r = reports[name]
        ci = r["bootstrap_95ci"]["macro_f1"]
        gap_f1 = "" if name in ("M0", "M1") else f"{r['macro_f1'] - m1_f1:+.4f}"
        gap_k = "" if name in ("M0", "M1") else f"{r['quadratic_kappa'] - m1_k:+.4f}"
        lines.append(f"| {name} | {r['macro_f1']:.4f} | [{ci['lo']:.3f}, {ci['hi']:.3f}] | "
                     f"{r['quadratic_kappa']:.4f} | {r['accuracy']:.4f} | {gap_f1} | {gap_k} |")
    (TABLES / "T3_main.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Error propagation for the best M3 (highest macro-F1).
    best = max(M3_PREDS, key=lambda n: reports[n]["macro_f1"])
    m = m3_frames[best]["pand_id"].isin(common)
    ep = error_propagation(
        m3_frames[best][m].reset_index(drop=True),
        m3_preds[best][m].reset_index(drop=True),
        pd.Series(m3_frames[best].loc[m, "energy_class"].values),
    )
    ep["best_m3"] = best
    (REPORTS / "error_propagation.json").write_text(
        json.dumps(ep, indent=2, ensure_ascii=False), encoding="utf-8")
    el = [
        f"# Table — Error propagation for best M3 ({best}), hold-out",
        "",
        f"- misclassified: {ep['n_misclassified']} / {ep['n_total']}",
        f"- attributable to vision attribute error: {ep['vision_attr_error']} "
        f"(type {ep['type_wrong']}, period {ep['period_wrong']}, both {ep['both_wrong']})",
        f"- attrs correct but LightGBM wrong (downstream): {ep['attrs_correct_downstream_error']}",
    ]
    (TABLES / "T3_error_propagation.md").write_text("\n".join(el) + "\n", encoding="utf-8")

    # ASCII-safe console summary (full tables with unicode are in the .md files).
    for name in order:
        r = reports[name]
        gap = "" if name in ("M0", "M1") else f" gap_mF1={r['macro_f1']-m1_f1:+.4f} gap_k={r['quadratic_kappa']-m1_k:+.4f}"
        print(f"{name:13s} macroF1={r['macro_f1']:.4f} kappa={r['quadratic_kappa']:.4f} acc={r['accuracy']:.4f}{gap}")
    print(f"error_propagation[{best}]: {json.dumps(ep, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
