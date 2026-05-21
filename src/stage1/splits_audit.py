"""Generate paper-appendix markdown documenting the Stage 1 Phase B splits.

Reads holdout_test_pand_ids.parquet + dev_fold_indices.parquet and writes a
markdown report covering the hold-out checksum, per-strata distribution
(city / Gebouwtype / Energieklasse / tabula_period), and the per-fold
composition of the dev 5-fold split. Output: reports/stage1/splits_audit_phase_b.md
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def crosstab_md(df: pd.DataFrame, key: str, holdout_pids: set, dev_pids: set) -> list[str]:
    df = df[df["pand_id"].astype(str).isin(holdout_pids | dev_pids)].copy()
    df["split"] = df["pand_id"].astype(str).map(
        lambda p: "holdout" if p in holdout_pids else "dev"
    )
    pivot = df.groupby([key, "split"]).size().unstack(fill_value=0).reindex(
        columns=["holdout", "dev"], fill_value=0,
    )
    pivot["holdout_pct"] = (pivot["holdout"] / (pivot["holdout"] + pivot["dev"]) * 100).round(1)
    pivot = pivot.sort_values("dev", ascending=False)

    lines = [f"| {key} | Holdout | Dev | Holdout % |", "|---|---:|---:|---:|"]
    for k, row in pivot.iterrows():
        lines.append(f"| {k} | {int(row['holdout'])} | {int(row['dev'])} | {row['holdout_pct']:.1f}% |")
    return lines


def fold_composition_md(dev_df: pd.DataFrame) -> list[str]:
    lines = [
        "| Fold | Train | Val | val_amsterdam | val_rotterdam | val_utrecht | val_delft | val_SFH | val_TH | val_MFH | val_AB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    n_total = len(dev_df)
    for f in sorted(dev_df["fold"].unique()):
        val = dev_df[dev_df["fold"] == f]
        n_val = len(val)
        n_train = n_total - n_val
        city_vc = val["city"].value_counts()
        type_vc = val["building_type"].value_counts()
        row = [
            str(f), str(n_train), str(n_val),
            str(int(city_vc.get("amsterdam", 0))),
            str(int(city_vc.get("rotterdam", 0))),
            str(int(city_vc.get("utrecht", 0))),
            str(int(city_vc.get("delft", 0))),
            str(int(type_vc.get("SFH", 0))),
            str(int(type_vc.get("TH", 0))),
            str(int(type_vc.get("MFH", 0))),
            str(int(type_vc.get("AB", 0))),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_audit(
    holdout_path: Path,
    dev_path: Path,
    checksum_path: Path,
    gt_path: Path,
) -> str:
    holdout = pd.read_parquet(holdout_path)
    dev = pd.read_parquet(dev_path)
    gt = pd.read_parquet(gt_path)
    checksum_meta = checksum_path.read_text(encoding="utf-8")

    holdout_pids = set(holdout["pand_id"].astype(str))
    dev_pids = set(dev["pand_id"].astype(str))

    assert holdout_pids.isdisjoint(dev_pids), "hold-out and dev pand_ids overlap"

    lines: list[str] = []
    lines.append("# Stage 1 Phase B Splits Audit\n")
    lines.append("Two-stage StratifiedGroupKFold: 20% hold-out test + 80% dev 5-fold CV.\n")
    lines.append("## Provenance\n")
    lines.append("```")
    lines.append(checksum_meta.strip())
    lines.append("```\n")

    lines.append("## Sizes\n")
    lines.append(f"- Training universe (manifest ∩ GT): **{len(holdout) + len(dev):,}** buildings")
    lines.append(f"- Hold-out test: **{len(holdout):,}** buildings ({100*len(holdout)/(len(holdout)+len(dev)):.1f}%)")
    lines.append(f"- Dev set: **{len(dev):,}** buildings ({100*len(dev)/(len(holdout)+len(dev)):.1f}%)")
    lines.append(f"- Dev folds: **5** (val ~{len(dev)//5:,} per fold)\n")

    lines.append("## Hold-out vs Dev distribution\n")
    lines.append("### By city\n")
    lines.extend(crosstab_md(gt, "city", holdout_pids, dev_pids))
    lines.append("")
    lines.append("### By Gebouwtype (Stage 1 target)\n")
    lines.extend(crosstab_md(gt, "Gebouwtype", holdout_pids, dev_pids))
    lines.append("")
    lines.append("### By building_type (4-class, derived)\n")
    lines.extend(crosstab_md(gt, "building_type", holdout_pids, dev_pids))
    lines.append("")
    lines.append("### By Energieklasse (Stage 2/3 target)\n")
    lines.extend(crosstab_md(gt, "Energieklasse", holdout_pids, dev_pids))
    lines.append("")
    lines.append("### By tabula_period\n")
    lines.extend(crosstab_md(gt, "tabula_period", holdout_pids, dev_pids))
    lines.append("")

    lines.append("## Dev 5-fold composition (val side)\n")
    lines.extend(fold_composition_md(dev))
    lines.append("")

    lines.append("## Reproducibility note\n")
    lines.append("All splits are reproducible by running `python -m src.stage1.splits` with")
    lines.append("the same `random_state` (42). The SHA256 prefix above is a stable digest of the")
    lines.append("sorted hold-out `pand_id` list and should match across re-runs.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", default=None)
    parser.add_argument("--dev", default=None)
    parser.add_argument("--checksum", default=None)
    parser.add_argument("--gt", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    repo = Path(__file__).resolve().parents[2]
    holdout = Path(args.holdout) if args.holdout else repo / "data/processed/holdout_test_pand_ids.parquet"
    dev = Path(args.dev) if args.dev else repo / "data/processed/dev_fold_indices.parquet"
    checksum = Path(args.checksum) if args.checksum else repo / "data/processed/holdout_test_pand_ids.checksum.txt"
    gt = Path(args.gt) if args.gt else repo / "data/processed/stage1_gt.parquet"
    out = Path(args.out) if args.out else repo / "reports/stage1/splits_audit_phase_b.md"

    md = build_audit(holdout, dev, checksum, gt)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    logger.info("wrote %s (%d chars)", out, len(md))


if __name__ == "__main__":
    main()
