"""Binary task: the same predictions read at two operating points.

Why this table exists. The binary pool is ~70/30, and a probabilistic model
trained on log-loss puts almost every building below 0.5: the LightGBM routes
predict D-G for well under 10% of the hold-out, so macro-F1 at the default
threshold measures how eagerly a route guesses the minority class rather than
how well it separates the two. M3-VLMv3 tops macro-F1 at 0.5 while holding the
*worst* ROC-AUC of any route — an operating-point artefact, not skill.

So every route is scored twice:

- **0.5 (primary)** — the literature-comparable point. Mayer 2023 and Sun 2026
  both report a plain thresholded classifier, so this is the number that may sit
  next to theirs.
- **rate-matched (secondary)** — predict D-G for the top-k scoring buildings,
  where k is the *dev* prevalence of D-G. The base rate is a training-set
  quantity and hold-out scores are used only for ranking, so no hold-out label
  touches the threshold. This answers "if the model had to name the right number
  of poor performers, would it name the right ones?", which is the question a
  retrofit-targeting user actually asks.

Ranking metrics (ROC-AUC, PR-AUC) are threshold-free and identical under both.

Usage:
    uv run python -m src.stage3.binary_operating_point
    uv run python -m src.stage3.binary_operating_point --run-tag loco_amsterdam
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.stage2.features import BINARY_POSITIVE, REPO_ROOT, build_master_table
from src.stage2.metrics import evaluate

logger = logging.getLogger(__name__)

REPORTS = REPO_ROOT / "reports" / "stage3"
TABLES = REPO_ROOT / "reports" / "tables" / "stage3"

DEFAULT_ROUTES = ["M0", "M1", "M3-DINOv2", "M3-ResNet50", "M3-VLMv3",
                  "M2-DINOv2", "M2-ResNet50"]


def rate_matched_pred(proba: pd.Series, rate: float) -> pd.Series:
    """Label the top `rate` fraction of scores as the positive class."""
    cut = float(np.quantile(proba, 1.0 - rate))
    return pd.Series(np.where(proba > cut, BINARY_POSITIVE, "A-C"), index=proba.index), cut


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="pooled")
    parser.add_argument("--routes", default=",".join(DEFAULT_ROUTES))
    parser.add_argument("--dev-folds", default=None,
                        help="dev split whose base rate sets the threshold (LOCO override)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tag = "_binary" + ("" if args.run_tag == "pooled" else f"_{args.run_tag}")
    routes = [r.strip() for r in args.routes.split(",") if r.strip()]

    dev = build_master_table(dev_path=Path(args.dev_folds) if args.dev_folds else None)
    rate = float((dev["energy_binary"] == BINARY_POSITIVE).mean())
    logger.info("dev base rate P(%s) = %.4f", BINARY_POSITIVE, rate)

    rows, thresholds = [], {}
    for r in routes:
        path = REPORTS / f"{r}{tag}_holdout_preds.parquet"
        label = r
        if not path.exists():
            # An interrupted GPU run leaves a _provisional result (hold-out model
            # picked from fewer than 5 folds). Show it rather than drop the route,
            # but never let it pass as the protocol number.
            alt = REPORTS / f"{r}{tag}_provisional_holdout_preds.parquet"
            if alt.exists():
                path, label = alt, f"{r} *"
                logger.warning("%s: using PROVISIONAL predictions", r)
            else:
                logger.warning("%s: %s missing, skipping", r, path.name)
                continue
        df = pd.read_parquet(path)
        if "proba" not in df.columns:
            logger.warning("%s: no proba column, skipping", r)
            continue
        at_default = evaluate(df, with_ci=False, task="binary", proba=df["proba"])
        rm_pred, cut = rate_matched_pred(df["proba"], rate)
        thresholds[label] = cut
        at_matched = evaluate(df.assign(pred=rm_pred), with_ci=False,
                              task="binary", proba=df["proba"])
        rows.append((label, at_default, at_matched))

    if not rows:
        logger.error("no route predictions found for tag %s", tag)
        return

    n = rows[0][1]["n_eval"]
    split_name = "hold-out" if args.run_tag == "pooled" else args.run_tag
    L = [
        f"# T8 — Binary task at two operating points ({split_name}, n={n:,})",
        "",
        f"Positive class = `{BINARY_POSITIVE}`. Dev base rate P({BINARY_POSITIVE}) = "
        f"**{rate:.3f}**; the rate-matched point labels that same fraction of the "
        "hold-out (highest scores first). ROC-AUC is threshold-free and shared by "
        "both columns.",
        "",
        "| route | mF1 @0.5 | bal.acc @0.5 | D-G recall @0.5 | mF1 @rate | bal.acc @rate "
        "| D-G recall @rate | ROC-AUC | PR-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r, a, b in rows:
        L.append(
            f"| {r} | {a['macro_f1']:.4f} | {a['balanced_accuracy']:.4f} | "
            f"{a['per_class'][BINARY_POSITIVE]['recall']:.4f} | "
            f"{b['macro_f1']:.4f} | {b['balanced_accuracy']:.4f} | "
            f"{b['per_class'][BINARY_POSITIVE]['recall']:.4f} | "
            f"{a.get('roc_auc', float('nan')):.4f} | {a.get('pr_auc', float('nan')):.4f} |")

    L += ["", "Rate-matched thresholds on P(D-G): "
          + ", ".join(f"{r} {t:.3f}" for r, t in thresholds.items()) + ".", ""]
    if any(r.endswith(" *") for r, _, _ in rows):
        L += ["`*` = provisional: the hold-out model was picked from fewer than the "
              "five folds the protocol requires, because that run was interrupted. "
              "Not a final number.", ""]

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / f"T8_binary_operating_point{tag.replace('_binary', '', 1)}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    logger.info("wrote %s", out)
    for r, a, b in rows:
        print(f"{r:14s} mF1@0.5={a['macro_f1']:.4f} mF1@rate={b['macro_f1']:.4f} "
              f"AUC={a.get('roc_auc', float('nan')):.4f}")


if __name__ == "__main__":
    main()
