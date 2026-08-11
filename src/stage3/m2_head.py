"""M2 end-to-end head: frozen DINOv2 embedding -> energy label A-G, bypassing
type/year/floor and TABULA. Trained on dev embeddings, evaluated on the same
hold-out common set as M3.

Primary head = LightGBM with the SAME fixed HP as M1/M3, so M2 vs M3-DINOv2
isolates "end-to-end vs decomposed" (same learner, same backbone, only the input
differs: 768-d image embedding vs three extracted attributes). Logistic
regression (Mayer 2023 style) reported as a secondary reference.

Usage (after extract_embeddings):
    uv run python -m src.stage3.m2_head
"""

import argparse
import json
import logging

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.stage2.features import BINARY_POSITIVE, REPO_ROOT, build_master_table, to_binary
from src.stage2.metrics import evaluate
from src.stage2.train_eval import params_for, task_suffix
from src.stage3.features import load_holdout_labels

logger = logging.getLogger(__name__)

REPORTS = REPO_ROOT / "reports" / "stage3"
TABLES = REPO_ROOT / "reports" / "tables" / "stage3"
EMB_COLS = [f"e{i}" for i in range(768)]

# This module is the frozen-probe SECONDARY reading of M2-DINOv2; the paper's
# primary M2-DINOv2 comes from m2_aligned.py. It used to write plain
# "M2-DINOv2_*" filenames and so silently overwrote the aligned result on any
# re-run — hence the hand-renamed frozenprobe files already on disk. It now owns
# the frozenprobe name outright.
ROUTE = "M2-DINOv2-frozenprobe"


def _xy(emb_path, labels: pd.DataFrame):
    emb = pd.read_parquet(emb_path)
    emb["pand_id"] = emb["pand_id"].astype(str)
    df = emb.merge(labels[["pand_id", "energy_class"]], on="pand_id", how="inner")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="7class", choices=["7class", "binary"])
    args = parser.parse_args()
    task, sfx = args.task, task_suffix(args.task)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dev_labels = build_master_table()[["pand_id", "energy_class"]]
    dev_labels["pand_id"] = dev_labels["pand_id"].astype(str)
    ho_labels = load_holdout_labels()

    dev = _xy(REPORTS / "embeddings_dev.parquet", dev_labels)
    ho = _xy(REPORTS / "embeddings_holdout.parquet", ho_labels)
    if task == "binary":
        dev["energy_class"] = to_binary(dev["energy_class"])
        ho["energy_class"] = to_binary(ho["energy_class"])

    # Restrict hold-out to the SAME common set used by Stage 3 M1/M3.
    m1 = pd.read_parquet(REPORTS / f"M1{sfx}_holdout_preds.parquet")
    common = set(m1["pand_id"].astype(str))
    ho = ho[ho["pand_id"].isin(common)].reset_index(drop=True)
    logger.info("dev=%d  holdout(common)=%d", len(dev), len(ho))

    Xtr, ytr = dev[EMB_COLS], dev["energy_class"]
    Xte = ho[EMB_COLS]

    heads = {
        "lightgbm": LGBMClassifier(**params_for(task)),
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, class_weight=None),
        ),
    }

    results = {}
    for name, clf in heads.items():
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        df = pd.DataFrame({"pand_id": ho["pand_id"].values,
                           "true": ho["energy_class"].values, "pred": pred})
        proba = None
        if task == "binary":
            pos = list(clf.classes_).index(BINARY_POSITIVE)
            proba = clf.predict_proba(Xte)[:, pos]
            df["proba"] = proba
        rep = evaluate(df, with_ci=True, task=task, proba=proba)
        rep["head"] = name
        results[name] = (rep, df)
        logger.info("%s[%s] macroF1=%.4f kappa=%.4f acc=%.4f",
                    ROUTE, name, rep["macro_f1"], rep["quadratic_kappa"], rep["accuracy"])

    # Primary = the better head by macro-F1.
    best = max(results, key=lambda n: results[n][0]["macro_f1"])
    rep, df = results[best]
    rep["primary_head"] = best
    df.to_parquet(REPORTS / f"{ROUTE}{sfx}_holdout_preds.parquet", index=False)
    (REPORTS / f"{ROUTE}{sfx}_metrics.json").write_text(
        json.dumps({k: v for k, v in rep.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    (REPORTS / f"{ROUTE}{sfx}_both_heads.json").write_text(
        json.dumps({n: r for n, (r, _) in results.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    # Comparison vs M1 / M3-DINOv2 (from saved Stage 3 metrics).
    def load(route):
        return json.load(open(REPORTS / f"{route}{sfx}_metrics.json"))
    m1m, m3m = load("M1"), load("M3-DINOv2")
    print(f"\n{'route':14s} {'macroF1':>8s} {'kappa':>7s} {'acc':>7s}")
    for nm, r in [("M1", m1m), ("M3-DINOv2", m3m),
                  (f"{ROUTE}[{best}]", rep)]:
        print(f"{nm:14s} {r['macro_f1']:8.4f} {r['quadratic_kappa']:7.4f} {r['accuracy']:7.4f}")
    print(f"\nM2 - M3-DINOv2 : mF1 {rep['macro_f1']-m3m['macro_f1']:+.4f}  "
          f"kappa {rep['quadratic_kappa']-m3m['quadratic_kappa']:+.4f}")
    print(f"M2 - M1        : mF1 {rep['macro_f1']-m1m['macro_f1']:+.4f}  "
          f"kappa {rep['quadratic_kappa']-m1m['quadratic_kappa']:+.4f}")


if __name__ == "__main__":
    main()
