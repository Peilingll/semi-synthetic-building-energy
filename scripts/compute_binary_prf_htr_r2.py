"""Two Table revisions requested on 2026-08-21.

1. Binary macro precision / recall / F1 @0.5 for every route in Table 4.4,
   read from the per-class entries of the existing stage-3 metrics files
   (macro = unweighted mean of the A-C and D-G values).
2. Building-level R^2 of predicted vs reference H'tr for Table 4.6, reusing
   the geometry, lookup and cell logic of src/stage3/htr_instrument.py
   (WWR = 0.25, same building filter as the published MAE).

Usage: .venv/Scripts/python.exe scripts/compute_binary_prf_htr_r2.py
Output: reports/stage3/binary_prf_htr_r2.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.stage3.htr_instrument import MODELS, WWR_MAIN, h_tr, load_geometry  # noqa: E402
from src.tabula_matcher import classify_period  # noqa: E402

BINARY_METRICS = {
    "M1": "M1_binary_metrics.json",
    "M2-DINOv2": "M2-DINOv2_binary_metrics.json",
    "M2-ResNet50": "M2-ResNet50_binary_metrics.json",
    "M2-VLM-binprompt": "M2-VLM-binprompt_binary_metrics.json",
    "M3-DINOv2": "M3-DINOv2_binary_metrics.json",
    "M3-ResNet50": "M3-ResNet50_binary_metrics.json",
    "M3-VLMv3": "M3-VLMv3_binary_metrics.json",
}


def macro_prf(metrics: dict) -> dict:
    pc = metrics["per_class"]
    return {
        "macro_precision": round(sum(v["precision"] for v in pc.values()) / len(pc), 4),
        "macro_recall": round(sum(v["recall"] for v in pc.values()) / len(pc), 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "per_class": {k: {m: v[m] for m in ("precision", "recall", "f1")} for k, v in pc.items()},
        "n_eval": metrics["n_eval"],
    }


def main() -> None:
    out = {"binary_prf_at_0.5": {}, "htr_r2": {}}

    for route, fname in BINARY_METRICS.items():
        m = json.load(open(REPO / "reports/stage3" / fname))
        out["binary_prf_at_0.5"][route] = macro_prf(m)

    # M0 uniform random guessing, analytic expectation on the hold-out prevalence
    m1 = json.load(open(REPO / "reports/stage3/M1_binary_metrics.json"))
    supp = {k: v["support"] for k, v in m1["per_class"].items()}
    n = sum(supp.values())
    p = {k: v / n for k, v in supp.items()}
    f1 = lambda pr, rc: 0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc)
    out["binary_prf_at_0.5"]["M0-random"] = {
        "macro_precision": round(sum(p.values()) / len(p), 4),
        "macro_recall": 0.5,
        "macro_f1": round(sum(f1(p[k], 0.5) for k in p) / len(p), 4),
        "n_eval": n,
    }

    tab = pd.read_csv(REPO / "data/processed/tabula_nl.csv")
    tab["key"] = tab["building_type"] + "|" + tab["period"]
    U = tab.set_index("key")[["u_wall", "u_roof", "u_floor", "u_window"]]
    geo = load_geometry()
    for name, path in MODELS.items():
        pr = pd.read_parquet(REPO / path)
        pr["pand_id"] = pr["pand_id"].astype(str).str.zfill(16)
        d = pr.merge(geo, on="pand_id", how="inner")
        d = d.dropna(subset=["true_bouwjaar", "pred_year", "true_type", "pred_type"])
        cell_gt = d["true_type"].astype(str) + "|" + pd.Series(
            [classify_period(int(y)) for y in d["true_bouwjaar"]], index=d.index)
        cell_pr = d["pred_type"].astype(str) + "|" + pd.Series(
            [classify_period(int(round(y))) for y in d["pred_year"]], index=d.index)
        keep = cell_gt.isin(U.index) & cell_pr.isin(U.index)
        d, cell_gt, cell_pr = d[keep], cell_gt[keep], cell_pr[keep]
        hg = h_tr(cell_gt.to_numpy(), d, U, WWR_MAIN)
        hp = h_tr(cell_pr.to_numpy(), d, U, WWR_MAIN)
        m = np.isfinite(hg) & np.isfinite(hp)
        hg, hp = hg[m], hp[m]
        r2 = 1 - float(np.sum((hp - hg) ** 2)) / float(np.sum((hg - hg.mean()) ** 2))
        out["htr_r2"][name] = {"n": int(m.sum()), "r2": round(r2, 4),
                               "mae_check": round(float(np.abs(hp - hg).mean()), 4)}

    print(json.dumps(out, indent=1))
    with open(REPO / "reports/stage3/binary_prf_htr_r2.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
