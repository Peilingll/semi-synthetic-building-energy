"""Aggregate M2-VLM per-image energy-label predictions to per-pand_id (majority
vote, tie broken by the more-efficient/earlier label) and evaluate on the same
hold-out common set as Stage 3. Adds M2-VLM to the comparison.

Usage: uv run python -m src.stage3.m2_vlm_eval
"""

import json
import logging
from collections import Counter

import pandas as pd

from src.stage2.features import ENERGY_LABELS, REPO_ROOT
from src.stage2.metrics import evaluate
from src.stage3.features import load_holdout_labels

logger = logging.getLogger(__name__)
REPORTS = REPO_ROOT / "reports" / "stage3"
ORDER = {lab: i for i, lab in enumerate(ENERGY_LABELS)}


def vote(labels: list[str]) -> str:
    c = Counter(labels)
    top = max(c.values())
    winners = [lab for lab, n in c.items() if n == top]
    return min(winners, key=lambda l: ORDER[l])  # tie -> more efficient (A side)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    per_img = pd.read_parquet(REPORTS / "m2vlm_holdout_per_image.parquet")
    per_img["pand_id"] = per_img["pand_id"].astype(str)
    ok = per_img[per_img["parse_ok"] & per_img["pred_label"].notna()]
    logger.info("per-image parse_ok: %d / %d (%.3f)", len(ok), len(per_img), len(ok) / len(per_img))

    agg = (ok.groupby("pand_id")["pred_label"].apply(lambda s: vote(list(s)))
           .rename("pred").reset_index())

    labels = load_holdout_labels()[["pand_id", "energy_class"]].rename(columns={"energy_class": "true"})
    labels["pand_id"] = labels["pand_id"].astype(str)
    df = agg.merge(labels, on="pand_id", how="inner")

    # same common set as classification Stage 3
    common = set(pd.read_parquet(REPORTS / "M1_holdout_preds.parquet")["pand_id"].astype(str))
    df = df[df["pand_id"].isin(common)].reset_index(drop=True)
    logger.info("evaluated buildings: %d", len(df))

    rep = evaluate(df[["true", "pred"]], with_ci=True)
    rep["route"] = "M2-VLMv1-zeroshot"
    rep["n"] = int(len(df))
    rep["pred_distribution"] = df["pred"].value_counts().reindex(ENERGY_LABELS).fillna(0).astype(int).to_dict()
    df.to_parquet(REPORTS / "M2-VLM_holdout_preds.parquet", index=False)
    (REPORTS / "M2-VLM_metrics.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    m3 = json.load(open(REPORTS / "M3-VLMv3_metrics.json"))
    m1 = json.load(open(REPORTS / "M1_metrics.json"))
    print(f"\n{'route':22s} {'macroF1':>8s} {'kappa':>7s} {'acc':>7s}")
    for nm, r in [("M1 (GT)", m1), ("M3-VLMv3 (decomp ZS)", m3), ("M2-VLM (direct ZS)", rep)]:
        print(f"{nm:22s} {r['macro_f1']:8.4f} {r['quadratic_kappa']:7.4f} {r['accuracy']:7.4f}")
    print(f"\nM2-VLM pred distribution: {rep['pred_distribution']}")
    print(f"M2-VLM - M3-VLM : mF1 {rep['macro_f1']-m3['macro_f1']:+.4f}  kappa {rep['quadratic_kappa']-m3['quadratic_kappa']:+.4f}")


if __name__ == "__main__":
    main()
