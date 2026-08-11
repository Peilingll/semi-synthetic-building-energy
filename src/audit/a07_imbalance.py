"""0.9 — How much of the low score is imbalance, and how much is structural?

Two imbalances are in play:
  feature side  79.5% of the dev pool sits in one TABULA cell (AB|NL.01),
                a Mapillary sampling artefact
  label side    C 31% ... G 3.9%

Everything below holds the feature set fixed at S_full (type, bouwjaar, four
U-values, num_floors, city) and the hyper-parameters fixed, and varies only the
pool composition, the pool size and the class weighting. Comparing the runs
isolates each effect:

  dev vs dev+balanced          -> does class weighting help?
  dev vs fullstock@8k          -> does a less skewed pool help, at equal n?
  fullstock@8k vs fullstock    -> does 15x more data help?
  dev vs dev cell-capped       -> does breaking the AB|NL.01 monopoly help?

Output: reports/tables/audit/A07_imbalance.md
"""

import argparse
import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

from src.audit import ep_raw
from src.stage2.features import (
    BINARY_POSITIVE,
    ENERGY_LABELS,
    U_COLS,
    merge_energy_class,
    to_binary,
)
from src.stage2.metrics import evaluate
from src.stage2.train_eval import params_for, task_suffix

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROCESSED = ep_raw.REPO_ROOT / "data" / "processed"
TABLES = ep_raw.REPO_ROOT / "reports" / "tables" / "audit"
FEATS = ["building_type", "bouwjaar", *U_COLS, "num_floors", "city"]
CATS = ["building_type", "city"]
SEED = 0


def build_pool() -> pd.DataFrame:
    gt = pd.read_parquet(PROCESSED / "stage1_gt.parquet")
    gt["pand_id"] = gt["pand_id"].astype(str).str.zfill(16)
    tab = pd.read_csv(PROCESSED / "tabula_nl.csv")
    df = gt.merge(tab[["building_type", "period", *U_COLS]],
                  left_on=["building_type", "tabula_period"],
                  right_on=["building_type", "period"], how="left").drop(columns="period")
    df["energy_class"] = merge_energy_class(df["Energieklasse"])
    df = df[df["energy_class"].isin(ENERGY_LABELS) & df["u_wall"].notna()].copy()
    df["energy_binary"] = to_binary(df["energy_class"])
    df["cell"] = df["building_type"].astype(str) + "|" + df["tabula_period"].astype(str)
    return df.reset_index(drop=True)


def run(df: pd.DataFrame, balanced: bool, task: str,
        folds: np.ndarray | None = None) -> dict:
    """OOF over 5 folds. `folds` supplies the frozen dev fold ids; without it we
    fall back to a stratified shuffle split, which is the only option for the
    derived pools (down-sampled / full stock) that have no frozen assignment."""
    X = df[FEATS].copy()
    for c in CATS:
        X[c] = X[c].astype("category")
    y = df["energy_class" if task == "7class" else "energy_binary"].to_numpy()
    params = params_for(task)
    if balanced:
        params["class_weight"] = "balanced"
    oof = np.empty(len(df), dtype=object)
    oof_proba = np.full(len(df), np.nan)

    if folds is None:
        splits = StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)
    else:
        splits = ((np.where(folds != f)[0], np.where(folds == f)[0]) for f in range(5))

    for tr, va in splits:
        cl = LGBMClassifier(**params)
        cl.fit(X.iloc[tr], y[tr], categorical_feature=CATS)
        oof[va] = cl.predict(X.iloc[va])
        if task == "binary":
            pos = list(cl.classes_).index(BINARY_POSITIVE)
            oof_proba[va] = cl.predict_proba(X.iloc[va])[:, pos]
    out = pd.DataFrame({"true": y, "pred": oof})
    return evaluate(out, with_ci=False, task=task,
                    proba=oof_proba if task == "binary" else None)


def describe(df: pd.DataFrame, task: str) -> str:
    top = df["cell"].value_counts(normalize=True).iloc[0]
    lab = df["energy_class" if task == "7class" else "energy_binary"].value_counts(normalize=True)
    return (f"top cell {top:.1%}, largest class {lab.max():.1%}, "
            f"smallest class {lab.min():.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="7class", choices=["7class", "binary"])
    args = parser.parse_args()
    task = args.task

    full = build_pool()
    dev_folds = pd.read_parquet(PROCESSED / "dev_fold_indices.parquet")
    dev_folds["pand_id"] = dev_folds["pand_id"].astype(str).str.zfill(16)
    fold_map = dict(zip(dev_folds["pand_id"], dev_folds["fold"]))
    dev_ids = set(fold_map)
    dev = full[full["pand_id"].isin(dev_ids)].reset_index(drop=True)
    dev_frozen_folds = dev["pand_id"].map(fold_map).to_numpy()

    # equal-n random draw from the full stock
    sub = full.sample(n=len(dev), random_state=SEED).reset_index(drop=True)
    def cap_by(df: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
        """Down-sample every group of `col` to at most n rows."""
        idx = np.concatenate([
            g.sample(min(len(g), n), random_state=SEED).index.to_numpy()
            for _, g in df.groupby(col, observed=True)])
        return df.loc[np.sort(idx)].reset_index(drop=True)

    # cell-capped dev: no cell may exceed the 2nd-largest cell's size
    counts = dev["cell"].value_counts()
    cap = int(counts.iloc[1]) if len(counts) > 1 else len(dev)
    capped = cap_by(dev, "cell", cap)
    # class-capped dev: every label capped at the smallest class size
    ccap = int(dev["energy_class"].value_counts().min())
    clsbal = cap_by(dev, "energy_class", ccap)

    runs = [
        # The frozen-fold row is the one directly comparable to T2a's S_full;
        # every derived pool below has no frozen assignment, so the contrast
        # rows all use the stratified split and are compared among themselves.
        ("dev, frozen Stage-1 folds", dev, False, dev_frozen_folds),
        ("dev (reference)", dev, False, None),
        ("dev + class_weight=balanced", dev, True, None),
        (f"dev, cell-capped at {cap}", capped, False, None),
        (f"dev, every class capped at {ccap}", clsbal, False, None),
        ("full stock, same n as dev", sub, False, None),
        ("full stock, all", full, False, None),
        ("full stock + class_weight=balanced", full, True, None),
    ]

    task_name = "A-G 7-class" if task == "7class" else "binary A-C | D-G"
    L = [f"# A07 — is the low score caused by imbalance? ({task_name})", "",
         "Feature set fixed at S_full (type, bouwjaar, 4 U-values, num_floors, city), "
         "hyper-parameters fixed, 5-fold OOF. Only the pool and the class "
         "weighting change, so the differences are attributable.", "",
         "The first row uses the frozen Stage-1 dev folds and is therefore the row "
         "comparable to T2a; the rest use a stratified shuffle split, the only option "
         "for the down-sampled and full-stock pools, and are compared among themselves.",
         "",
         "| run | n | composition | macro-F1 | quad. kappa | acc |",
         "|---|---:|---|---:|---:|---:|"]
    res = {}
    for name, d, bal, fld in runs:
        r = run(d, bal, task, fld)
        res[name] = r
        L += [f"| {name} | {len(d):,} | {describe(d, task)} | **{r['macro_f1']:.4f}** | "
              f"{r['quadratic_kappa']:.4f} | {r['accuracy']:.4f} |"]
        logger.info("%-38s n=%7d  mF1=%.4f", name, len(d), r["macro_f1"])
    L += [""]

    ref = res["dev (reference)"]["macro_f1"]
    L += ["## What each comparison isolates", "",
          "| comparison | effect | d macro-F1 |", "|---|---|---:|"]
    for a, b, what in [
        ("dev (reference)", "dev + class_weight=balanced", "class weighting only"),
        ("dev (reference)", f"dev, cell-capped at {cap}", "breaking the AB|NL.01 monopoly"),
        ("dev (reference)", f"dev, every class capped at {ccap}", "perfect label balance"),
        ("dev (reference)", "full stock, same n as dev", "less skewed pool, same n"),
        ("full stock, same n as dev", "full stock, all", "15x more data"),
        ("dev (reference)", "full stock, all", "pool + size combined"),
    ]:
        L += [f"| {a} -> {b} | {what} | "
              f"{res[b]['macro_f1'] - res[a]['macro_f1']:+.4f} |"]
    L += ["",
          f"Reference macro-F1 on the dev pool: {ref:.4f} "
          f"(frozen folds: {res['dev, frozen Stage-1 folds']['macro_f1']:.4f}).", ""]

    out = TABLES / f"A07_imbalance{task_suffix(task)}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
