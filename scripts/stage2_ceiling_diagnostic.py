"""One-off diagnostic: registry-feature ceiling on the FULL GT (124k) vs the
SVI dev subset (8,068).

Answers: is Stage 2's low ceiling a property of the old-city SVI subsample, or
intrinsic to predicting EP label from registry attributes? Vision-free, so the
full GT can be used. NOT part of the official Stage 2 ablation.
"""

import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold

from src.stage2.features import (
    CATEGORICAL,
    ENERGY_LABELS,
    PROCESSED,
    RUN_FEATURES,
    U_COLS,
    merge_energy_class,
)
from src.stage2.metrics import evaluate
from src.stage2.train_eval import FIXED_PARAMS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RUNS = ["M0", "S_min", "S_lookup", "S_full"]
N_FOLDS = 5
SEED = 42


def build_full() -> pd.DataFrame:
    gt = pd.read_parquet(PROCESSED / "stage1_gt.parquet")
    gt["pand_id"] = gt["pand_id"].astype(str)
    tabula = pd.read_csv(PROCESSED / "tabula_nl.csv")
    df = gt.merge(tabula[["building_type", "period"] + U_COLS],
                  left_on=["building_type", "tabula_period"],
                  right_on=["building_type", "period"], how="left").drop(columns=["period"])
    df = df[df["u_wall"].notna()].copy()
    df["energy_class"] = merge_energy_class(df["Energieklasse"])
    df = df[df["energy_class"].isin(ENERGY_LABELS)].copy().reset_index(drop=True)
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


def assign_folds(df: pd.DataFrame) -> np.ndarray:
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = np.full(len(df), -1, dtype=int)
    for f, (_, va) in enumerate(sgkf.split(df, df["energy_class"], groups=df["pand_id"])):
        folds[va] = f
    return folds


def oof_for_run(df: pd.DataFrame, folds: np.ndarray, run: str) -> pd.DataFrame:
    y = df["energy_class"]
    pred = np.empty(len(df), dtype=object)
    if run == "M0":
        for f in range(N_FOLDS):
            tr = folds != f
            pred[folds == f] = y[tr].value_counts().idxmax()
    else:
        cols = RUN_FEATURES[run]
        X = df[cols]
        cat = [c for c in CATEGORICAL if c in cols]
        for f in range(N_FOLDS):
            tr, va = folds != f, folds == f
            clf = LGBMClassifier(**FIXED_PARAMS)
            clf.fit(X[tr], y[tr], categorical_feature=cat or "auto")
            pred[va] = clf.predict(X[va])
    return pd.DataFrame({"true": y.values, "pred": pred})


def main():
    df = build_full()
    folds = assign_folds(df)
    logger.info("FULL GT: %d buildings | classes=%s",
                len(df), df["energy_class"].value_counts().reindex(ENERGY_LABELS).to_dict())
    logger.info("median bouwjaar=%d (dev subset was 1925)", int(df["bouwjaar"].median()))
    print(f"\n{'run':10s} {'macro_f1':>9s} {'kappa':>7s} {'acc':>7s}")
    for run in RUNS:
        oof = oof_for_run(df, folds, run)
        r = evaluate(oof, with_ci=False)
        print(f"{run:10s} {r['macro_f1']:9.4f} {r['quadratic_kappa']:7.4f} {r['accuracy']:7.4f}")


if __name__ == "__main__":
    main()
