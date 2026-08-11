"""Aggregate M2-VLM per-image energy-label predictions to per-pand_id (majority
vote, tie broken by the more-efficient/earlier label) and evaluate on the same
hold-out common set as Stage 3. Adds M2-VLM to the comparison.

Unlike every other route this one is never retrained: the model was asked for a
letter, and there is nothing to fit. So the binary task has two distinct
readings, selected with --source:

  collapse   the SAME A-G zero-shot answers re-binned to A-C | D-G
  binprompt  a separate run that asked the binary question DIRECTLY
             (`m2_vlm_runner.py --task binary`, identical evidence wording,
             only the answer options differ)

Both are zero-shot, so neither carries a probability and neither gets an AUC.

Usage:
    uv run python -m src.stage3.m2_vlm_eval
    uv run python -m src.stage3.m2_vlm_eval --task binary
    uv run python -m src.stage3.m2_vlm_eval --task binary --source binprompt
"""

import argparse
import json
import logging
from collections import Counter

import pandas as pd

from src.stage2.features import ENERGY_LABELS, REPO_ROOT, labels_for, to_binary
from src.stage2.metrics import evaluate
from src.stage2.train_eval import task_suffix
from src.stage3.features import load_holdout_labels

logger = logging.getLogger(__name__)
REPORTS = REPO_ROOT / "reports" / "stage3"


def vote(labels: list[str], order: dict[str, int]) -> str:
    c = Counter(labels)
    top = max(c.values())
    winners = [lab for lab, n in c.items() if n == top]
    return min(winners, key=lambda l: order[l])  # tie -> more efficient (A side)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="7class", choices=["7class", "binary"])
    parser.add_argument("--source", default="collapse", choices=["collapse", "binprompt"],
                        help="binary only: re-bin the A-G answers, or read the "
                             "separate run that asked the binary question directly")
    args = parser.parse_args()
    task, sfx = args.task, task_suffix(args.task)
    if args.source == "binprompt" and task != "binary":
        parser.error("--source binprompt requires --task binary")
    binprompt = args.source == "binprompt"
    route = "M2-VLM-binprompt" if binprompt else "M2-VLM"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    src = ("m2vlm_holdout_binary_per_image.parquet" if binprompt
           else "m2vlm_holdout_per_image.parquet")
    per_img = pd.read_parquet(REPORTS / src)
    per_img["pand_id"] = per_img["pand_id"].astype(str)
    ok = per_img[per_img["parse_ok"] & per_img["pred_label"].notna()]
    logger.info("%s: per-image parse_ok %d / %d (%.3f)", src, len(ok), len(per_img),
                len(ok) / len(per_img))

    # tie-break toward the more efficient side in both label spaces
    order = ({lab: i for i, lab in enumerate(labels_for("binary"))} if binprompt
             else {lab: i for i, lab in enumerate(ENERGY_LABELS)})
    agg = (ok.groupby("pand_id")["pred_label"].apply(lambda s: vote(list(s), order))
           .rename("pred").reset_index())

    labels = load_holdout_labels()[["pand_id", "energy_class"]].rename(columns={"energy_class": "true"})
    labels["pand_id"] = labels["pand_id"].astype(str)
    df = agg.merge(labels, on="pand_id", how="inner")

    # same common set as classification Stage 3
    common = set(pd.read_parquet(REPORTS / f"M1{sfx}_holdout_preds.parquet")["pand_id"].astype(str))
    df = df[df["pand_id"].isin(common)].reset_index(drop=True)
    logger.info("evaluated buildings: %d", len(df))

    if task == "binary":
        df["true"] = to_binary(df["true"])
        if not binprompt:
            df["pred"] = to_binary(df["pred"])

    rep = evaluate(df[["true", "pred"]], with_ci=True, task=task)
    rep["route"] = f"{route}-zeroshot"
    rep["n"] = int(len(df))
    rep["pred_distribution"] = (df["pred"].value_counts().reindex(labels_for(task))
                                .fillna(0).astype(int).to_dict())
    if task == "binary":
        rep["note"] = (
            "asked the A-C | D-G question directly (separate zero-shot run, same "
            "evidence wording as the A-G prompt)" if binprompt else
            "A-G zero-shot answers collapsed to the Sun cut; not retrained and not "
            "re-prompted") + "; zero-shot, so no probability and no AUC"
    df.to_parquet(REPORTS / f"{route}{sfx}_holdout_preds.parquet", index=False)
    (REPORTS / f"{route}{sfx}_metrics.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    m3 = json.load(open(REPORTS / f"M3-VLMv3{sfx}_metrics.json"))
    m1 = json.load(open(REPORTS / f"M1{sfx}_metrics.json"))
    print(f"\n{'route':22s} {'macroF1':>8s} {'kappa':>7s} {'acc':>7s}")
    for nm, r in [("M1 (GT)", m1), ("M3-VLMv3 (decomp ZS)", m3), (f"{route} (direct ZS)", rep)]:
        print(f"{nm:22s} {r['macro_f1']:8.4f} {r['quadratic_kappa']:7.4f} {r['accuracy']:7.4f}")
    print(f"\nM2-VLM pred distribution: {rep['pred_distribution']}")
    print(f"M2-VLM - M3-VLM : mF1 {rep['macro_f1']-m3['macro_f1']:+.4f}  kappa {rep['quadratic_kappa']-m3['quadratic_kappa']:+.4f}")


if __name__ == "__main__":
    main()
