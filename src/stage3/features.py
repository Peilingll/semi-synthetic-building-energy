"""Stage 3 feature builders on the hold-out set (n=2,018).

Every route uses the SAME S_full feature schema; routes differ only in the
SOURCE of type / year / floor:
- M1: ground-truth registry attributes
- M3: vision-predicted attributes (Stage 1 hold-out predictions)

U-values are looked up from tabula_nl.csv on (building_type, period); for M3 the
period is derived from the PREDICTED year, so vision errors propagate into the
lookup — the realistic pipeline. city is GT metadata (always known).
"""

import logging
from pathlib import Path

import pandas as pd

from src.stage2.features import (
    ENERGY_LABELS,
    PROCESSED,
    S_FULL,
    U_COLS,
    merge_energy_class,
)
from src.tabula_matcher import classify_period

logger = logging.getLogger(__name__)

_TABULA = pd.read_csv(PROCESSED / "tabula_nl.csv")


def _add_uvalues(df: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(
        _TABULA[["building_type", "period"] + U_COLS],
        left_on=["building_type", "tabula_period"],
        right_on=["building_type", "period"],
        how="left",
    ).drop(columns=["period"])
    n_nan = out["u_wall"].isna().sum()
    if n_nan:
        logger.warning("%d rows missing U-values after lookup", n_nan)
    return out


def load_holdout_labels(holdout_path: str | Path | None = None) -> pd.DataFrame:
    """GT attributes + energy label for the hold-out buildings (from stage1_gt)."""
    gt = pd.read_parquet(PROCESSED / "stage1_gt.parquet")
    gt["pand_id"] = gt["pand_id"].astype(str)
    ho = pd.read_parquet(holdout_path or PROCESSED / "holdout_test_pand_ids.parquet")
    ho["pand_id"] = ho["pand_id"].astype(str)
    g = gt[gt["pand_id"].isin(set(ho["pand_id"]))].copy()
    g["energy_class"] = merge_energy_class(g["Energieklasse"])
    g = g[g["energy_class"].isin(ENERGY_LABELS)].reset_index(drop=True)
    return g


def build_m1_holdout(holdout_path: str | Path | None = None) -> pd.DataFrame:
    """M1: GT type/year/floor → S_full features + true label."""
    g = load_holdout_labels(holdout_path)
    df = pd.DataFrame({
        "pand_id": g["pand_id"].values,
        "building_type": g["building_type"].astype(str).values,
        "bouwjaar": g["bouwjaar"].astype(int).values,
        "num_floors": g["num_floors"].astype(int).values,
        "city": g["city"].astype(str).values,
        "tabula_period": g["tabula_period"].astype(str).values,
        "energy_class": g["energy_class"].values,
    })
    return _add_uvalues(df)


def build_m3_holdout(pred_path: str | Path,
                     holdout_path: str | Path | None = None) -> pd.DataFrame:
    """M3: vision-predicted type/year/floor → S_full features + true label."""
    pred = pd.read_parquet(pred_path)
    pred["pand_id"] = pred["pand_id"].astype(str)
    labels = load_holdout_labels(holdout_path)[["pand_id", "energy_class",
                                                "building_type", "bouwjaar"]].rename(
        columns={"building_type": "true_type", "bouwjaar": "true_bouwjaar"})

    df = pred[["pand_id", "city", "pred_type", "pred_year", "pred_floors"]].merge(
        labels, on="pand_id", how="inner")
    n0 = len(df)
    df = df.dropna(subset=["pred_type", "pred_year", "pred_floors"]).reset_index(drop=True)
    if len(df) < n0:
        logger.info("dropped %d rows with missing vision predictions", n0 - len(df))
    df["building_type"] = df["pred_type"].astype(str)
    df["bouwjaar"] = df["pred_year"].round().astype(int)
    df["tabula_period"] = df["bouwjaar"].apply(classify_period)
    df["num_floors"] = df["pred_floors"].round().clip(lower=1).astype(int)
    df["city"] = df["city"].astype(str)
    df = _add_uvalues(df)
    logger.info("M3 holdout from %s: %d buildings", Path(pred_path).name, len(df))
    return df


def to_X(df: pd.DataFrame, cat_dtypes: dict) -> pd.DataFrame:
    """Slice to S_full columns, applying the training set's category dtypes so
    LightGBM's categorical encoding matches between train and predict."""
    X = df[S_FULL].copy()
    for col, categories in cat_dtypes.items():
        X[col] = pd.Categorical(X[col], categories=categories)
        n_unknown = X[col].isna().sum()
        if n_unknown:
            logger.warning("%s: %d values outside training categories", col, n_unknown)
    return X
