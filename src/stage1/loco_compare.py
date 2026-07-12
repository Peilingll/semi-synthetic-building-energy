"""LOCO Stage 1 paradigm comparison tables.

Produces reports/tables/stage1/T3_model_comparison_loco_<city>.md with two
sections:
1. Full LOCO hold-out (all imaged buildings of the held-out city) — trained
   paradigms only (their LOCO predictions cover the whole city).
2. Strictly comparable zero-shot subset (LOCO hold-out INTERSECT pooled
   hold-out) — all paradigms, including zero-shot ones whose predictions only
   exist on the pooled hold-out (split-independent, so reusable).
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.stage1.evaluate import evaluate_predictions

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]

MODELS = {
    "DINOv2 frozen": ("dinov2_frozen", True),
    "ResNet-50 ft": ("resnet50_ft", True),
    "InternVL3 (ZS)": ("vlm_internvl3", False),  # zero-shot: pooled preds reused
}
VLM_POOLED = REPO / "reports/stage1/vlm_internvl3/v3_holdout_per_pand_id.parquet"


def load_preds(name: str, run_tag: str) -> pd.DataFrame | None:
    subdir, trained = MODELS[name]
    if trained:
        path = REPO / "reports" / "stage1" / subdir / f"{run_tag}_holdout_preds.parquet"
        if not path.exists():
            logger.warning("%s: %s missing, skipping", name, path.name)
            return None
    else:
        path = VLM_POOLED
    df = pd.read_parquet(path)
    df["pand_id"] = df["pand_id"].astype(str)
    return df


def headline_row(name: str, rep: dict) -> str:
    return (f"| {name} | {rep['n_eval']} | {rep['type_acc']:.4f} | {rep['type_macro_f1']:.4f} "
            f"| {rep['year_mae']:.2f} | {rep['period_acc']:.4f} | {rep['floors_mae']:.3f} |")


HEADER = ["| model | n | type_acc | type_macro_f1 | year_mae | period_acc | floors_mae |",
          "|---|---:|---:|---:|---:|---:|---:|"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-city", default="amsterdam")
    parser.add_argument("--run-tag", default=None, help="default loco_<city>")
    args = parser.parse_args()
    run_tag = args.run_tag or f"loco_{args.holdout_city}"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    subset_ids = set(pd.read_parquet(
        REPO / "data" / "processed" / f"loco_{args.holdout_city}" / "holdout_vlm_subset.parquet"
    )["pand_id"].astype(str))

    full_rows, subset_rows = [], []
    for name in MODELS:
        df = load_preds(name, run_tag)
        if df is None:
            continue
        trained = MODELS[name][1]
        if trained:
            rep = evaluate_predictions(df, with_ci=False)
            full_rows.append(headline_row(name, rep))
        sub = df[df["pand_id"].isin(subset_ids)]
        rep_s = evaluate_predictions(sub, with_ci=False)
        subset_rows.append(headline_row(name, rep_s))
        logger.info("%s: full n=%s subset n=%d", name, len(df) if trained else "—", len(sub))

    lines = [
        f"# Table 3 — Stage 1 paradigm comparison, LOCO-{args.holdout_city}",
        "",
        f"Trained paradigms: trained on R+U+D, evaluated on {args.holdout_city}.",
        "Zero-shot paradigms: pooled predictions reused (split-independent).",
        "",
        f"## Full LOCO hold-out (all imaged {args.holdout_city} buildings)",
        "", *HEADER, *full_rows,
        "",
        f"## Strictly comparable subset (LOCO hold-out INTERSECT pooled hold-out, n={len(subset_ids)})",
        "", *HEADER, *subset_rows,
    ]
    out = REPO / "reports" / "tables" / "stage1" / f"T3_model_comparison_{run_tag}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
