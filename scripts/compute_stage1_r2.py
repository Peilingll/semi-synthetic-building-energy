"""Compute holdout R^2 for construction-year and floor-count regression (Table 4.2 supplement).

R^2 = 1 - SS_res / SS_tot, with SS_tot taken against the mean of the true values
over the same evaluated subset used for the published MAE (rows with a valid
prediction). Bootstrap 95% CI uses 1,000 building-level resamples, seed 42,
matching the convention of the existing holdout_metrics.json files.

Usage: .venv/Scripts/python.exe scripts/compute_stage1_r2.py
Output: reports/stage1/r2_holdout.json
"""

import json

import numpy as np
import pandas as pd

SOURCES = {
    "dinov2_frozen": "reports/stage1/dinov2_frozen/holdout_preds.parquet",
    "resnet50_ft": "reports/stage1/resnet50_ft/holdout_preds.parquet",
    "vlm_internvl3_v3": "reports/stage1/vlm_internvl3/v3_holdout_per_pand_id.parquet",
}

TARGETS = {
    "year": ("pred_year", "true_bouwjaar"),
    "floors": ("pred_floors", "true_num_floors"),
}

N_BOOT = 1000
SEED = 42


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = {"n_boot": N_BOOT, "seed": SEED, "models": {}}
    for model, path in SOURCES.items():
        df = pd.read_parquet(path)
        entry = {}
        for target, (pred_col, true_col) in TARGETS.items():
            sub = df[[pred_col, true_col]].dropna()
            y_true = sub[true_col].to_numpy(dtype=float)
            y_pred = sub[pred_col].to_numpy(dtype=float)
            point = r2(y_true, y_pred)
            boots = []
            for _ in range(N_BOOT):
                idx = rng.integers(0, len(sub), len(sub))
                yt = y_true[idx]
                if np.ptp(yt) == 0:
                    continue
                boots.append(r2(yt, y_pred[idx]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            entry[target] = {
                "n_eval": int(len(sub)),
                "mae_check": round(float(np.abs(y_true - y_pred).mean()), 4),
                "r2": round(point, 4),
                "r2_ci95": [round(float(lo), 4), round(float(hi), 4)],
                "true_std": round(float(y_true.std()), 2),
            }
        out["models"][model] = entry
        print(model, entry)
    with open("reports/stage1/r2_holdout.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
