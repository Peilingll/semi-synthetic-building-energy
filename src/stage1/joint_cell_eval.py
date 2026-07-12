"""Joint TABULA-cell assignment accuracy (#3).

A TABULA archetype is a cell in the (size_class x construction period) matrix.
Stage 1 reports type accuracy and period accuracy separately; archetype
assignment requires BOTH correct simultaneously. This script computes, per
paradigm, from existing hold-out predictions (no training):

  - type_acc, period_acc (reference)
  - joint_acc          P(pred type == true type AND pred period == true period)
  - majority_cell      baseline: share of the most frequent TRUE cell
  - macro_cell_recall  mean per-cell recall over true cells with support > 0
  - per-cell recall + top confusion pairs (honesty vs the 88%-AB dev bias)

Periods are derived from predicted continuous year via
tabula_matcher.classify_period for all paradigms (VLM's stored pred_period was
derived the same way at aggregate time).

Output: reports/tables/stage1/T4_joint_cell.md
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.tabula_matcher import classify_period

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports" / "tables" / "stage1" / "T4_joint_cell.md"

# (path, zero_shot) — zero-shot models have split-independent predictions, so
# their pooled files may be reused for other run tags; trained models MUST have
# a tagged prediction file or they are skipped (a pooled-trained model has seen
# the LOCO test city — falling back would contaminate the evaluation).
MODELS = {
    "DINOv2 frozen": (REPO / "reports/stage1/dinov2_frozen/holdout_preds.parquet", False),
    "ResNet-50 ft": (REPO / "reports/stage1/resnet50_ft/holdout_preds.parquet", False),
    "InternVL3 (ZS)": (REPO / "reports/stage1/vlm_internvl3/v3_holdout_per_pand_id.parquet", True),
}


def resolve_models(run_tag: str) -> dict[str, Path]:
    """Per-model prediction paths for a run tag (see MODELS zero-shot note)."""
    if run_tag == "pooled":
        return {name: path for name, (path, _) in MODELS.items()}
    resolved = {}
    for name, (path, zero_shot) in MODELS.items():
        tagged = path.with_name(f"{run_tag}_{path.name}")
        if tagged.exists():
            resolved[name] = tagged
        elif zero_shot:
            logger.warning("%s: zero-shot, reusing pooled %s", name, path.name)
            resolved[name] = path
        else:
            logger.warning("%s: no %s and model is trained — SKIPPED "
                           "(pooled fallback would leak the held-out city)", name, tagged.name)
    return resolved


def _load(name: str, path: Path, restrict_ids: set[str] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if restrict_ids is not None:
        before = len(df)
        df = df[df["pand_id"].astype(str).isin(restrict_ids)].copy()
        logger.info("%s: restricted %d -> %d buildings", name, before, len(df))
    df = df.dropna(subset=["pred_type", "pred_year", "true_type", "true_bouwjaar"]).copy()
    df["pred_period"] = df["pred_year"].round().astype(int).map(classify_period)
    df["true_period"] = df["true_bouwjaar"].round().astype(int).map(classify_period)
    df["true_cell"] = df["true_type"] + "|" + df["true_period"]
    df["pred_cell"] = df["pred_type"] + "|" + df["pred_period"]
    logger.info("%s: %d buildings with valid predictions", name, len(df))
    return df


def _metrics(df: pd.DataFrame) -> dict:
    type_ok = df["pred_type"] == df["true_type"]
    period_ok = df["pred_period"] == df["true_period"]
    joint_ok = type_ok & period_ok
    per_cell = joint_ok.groupby(df["true_cell"]).mean()
    support = df["true_cell"].value_counts()
    return {
        "n": len(df),
        "type_acc": type_ok.mean(),
        "period_acc": period_ok.mean(),
        "joint_acc": joint_ok.mean(),
        "majority_cell": support.max() / len(df),
        "macro_cell_recall": per_cell.mean(),
        "n_cells": len(support),
        "per_cell": pd.DataFrame({"support": support, "recall": per_cell}).sort_values(
            "support", ascending=False
        ),
        "confusions": (
            df.loc[~joint_ok]
            .groupby(["true_cell", "pred_cell"])
            .size()
            .sort_values(ascending=False)
            .head(8)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="pooled",
                        help="prediction file prefix (e.g. loco_amsterdam); pooled = original files")
    parser.add_argument("--restrict", type=Path, default=None,
                        help="parquet with pand_id column; evaluate only those buildings")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    restrict_ids = None
    if args.restrict is not None:
        restrict_ids = set(pd.read_parquet(args.restrict)["pand_id"].astype(str))
        logger.info("restricting to %d pand_ids from %s", len(restrict_ids), args.restrict)

    models = resolve_models(args.run_tag)
    results = {name: _metrics(_load(name, path, restrict_ids)) for name, path in models.items()}

    if args.out is not None:
        out_path = args.out
    elif args.run_tag == "pooled":
        out_path = OUT
    else:
        out_path = OUT.with_name(f"T4_joint_cell_{args.run_tag}.md")

    title_tag = "hold-out" if args.run_tag == "pooled" else args.run_tag
    lines = [
        f"# Table 4 — Joint TABULA-cell assignment ({title_tag})",
        "",
        "Cell = (size class x TABULA period). Joint = type AND period both correct.",
        "majority_cell = always predicting the most frequent TRUE cell (baseline).",
        "macro_cell_recall = unweighted mean recall over occupied cells.",
        "",
        "| model | n | type_acc | period_acc | joint_acc | majority-cell baseline | macro-cell recall | cells |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name} | {m['n']} | {m['type_acc']:.4f} | {m['period_acc']:.4f} "
            f"| {m['joint_acc']:.4f} | {m['majority_cell']:.4f} "
            f"| {m['macro_cell_recall']:.4f} | {m['n_cells']} |"
        )

    for name, m in results.items():
        lines += ["", f"## {name} — per-cell recall (true cells, by support)", "",
                  "| true cell | support | recall |", "|---|---:|---:|"]
        for cell, row in m["per_cell"].iterrows():
            lines.append(f"| {cell} | {int(row['support'])} | {row['recall']:.3f} |")
        lines += ["", f"### {name} — top confusion pairs (true -> pred, joint errors)", "",
                  "| true cell | pred cell | count |", "|---|---|---:|"]
        for (t, p), c in m["confusions"].items():
            lines.append(f"| {t} | {p} | {c} |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s", out_path)
    for name, m in results.items():
        logger.info(
            "%s: joint %.4f (majority-cell %.4f, macro-cell %.4f)",
            name, m["joint_acc"], m["majority_cell"], m["macro_cell_recall"],
        )


if __name__ == "__main__":
    main()
