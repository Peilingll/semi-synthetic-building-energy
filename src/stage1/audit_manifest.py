"""Audit the SVI manifest and write a markdown summary for the paper appendix.

Reports per-pand_id image count distribution (buckets 1 / 2-4 / 5-7 / 8-cap)
and per-city totals (buildings, images, mean/median per building), both for
the full manifest and after intersecting with the four-city Stage 1 GT.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def bucket_n_images(n: int) -> str:
    if n == 1:
        return "1"
    if n <= 4:
        return "2-4"
    if n <= 7:
        return "5-7"
    return "8 (cap)"


BUCKET_ORDER = ["1", "2-4", "5-7", "8 (cap)"]


def build_audit(manifest_path: Path, gt_path: Path | None) -> str:
    m = pd.read_parquet(manifest_path)
    logger.info("manifest: %d rows, %d unique pand_id", len(m), m["pand_id"].nunique())

    per_pand = m.groupby(["city", "pand_id"]).size().rename("n_images").reset_index()

    lines: list[str] = []
    lines.append("# SVI Manifest Audit (Stage 1 Phase B)\n")
    lines.append(f"- Manifest: `{manifest_path.name}`")
    lines.append(f"- Total images: {len(m):,}")
    lines.append(f"- Unique buildings (pand_id): {m['pand_id'].nunique():,}")
    lines.append("")

    lines.append("## Per-city totals (full manifest)")
    lines.append("")
    lines.append("| City | Buildings | Images | Mean/bldg | Median/bldg |")
    lines.append("|---|---:|---:|---:|---:|")
    for city, grp in per_pand.groupby("city"):
        n_b = len(grp)
        n_i = int(grp["n_images"].sum())
        mean = grp["n_images"].mean()
        med = grp["n_images"].median()
        lines.append(f"| {city} | {n_b:,} | {n_i:,} | {mean:.2f} | {med:.1f} |")
    lines.append(f"| **total** | **{per_pand.shape[0]:,}** | **{int(per_pand['n_images'].sum()):,}** | "
                 f"**{per_pand['n_images'].mean():.2f}** | **{per_pand['n_images'].median():.1f}** |")
    lines.append("")

    lines.append("## Image-count distribution per pand_id (full manifest)")
    lines.append("")
    per_pand["bucket"] = per_pand["n_images"].apply(bucket_n_images)
    counts = per_pand["bucket"].value_counts().reindex(BUCKET_ORDER).fillna(0).astype(int)
    total = counts.sum()
    lines.append("| n_images | Buildings | % |")
    lines.append("|---|---:|---:|")
    for bucket in BUCKET_ORDER:
        n = counts[bucket]
        pct = 100 * n / total if total else 0
        lines.append(f"| {bucket} | {n:,} | {pct:.1f}% |")
    lines.append(f"| **total** | **{total:,}** | 100.0% |")
    lines.append("")

    if gt_path is not None and gt_path.exists():
        gt = pd.read_parquet(gt_path)
        logger.info("GT: %d rows", len(gt))
        gt_pids = set(gt["pand_id"].astype(str))
        per_pand["in_gt"] = per_pand["pand_id"].astype(str).isin(gt_pids)
        intersect = per_pand[per_pand["in_gt"]]
        logger.info("manifest ∩ GT: %d buildings", len(intersect))

        lines.append("## Manifest ∩ GT (training universe)")
        lines.append("")
        lines.append(f"- Buildings in both manifest and Stage 1 GT: **{len(intersect):,}**")
        lines.append(f"- Manifest buildings without GT row (filtered out): {(~per_pand['in_gt']).sum():,}")
        lines.append("")

        lines.append("### Per-city training universe")
        lines.append("")
        lines.append("| City | Buildings | Images | Mean/bldg |")
        lines.append("|---|---:|---:|---:|")
        for city, grp in intersect.groupby("city"):
            n_b = len(grp)
            n_i = int(grp["n_images"].sum())
            mean = grp["n_images"].mean()
            lines.append(f"| {city} | {n_b:,} | {n_i:,} | {mean:.2f} |")
        lines.append(f"| **total** | **{len(intersect):,}** | **{int(intersect['n_images'].sum()):,}** | "
                     f"**{intersect['n_images'].mean():.2f}** |")
        lines.append("")

        lines.append("### building_type composition (training universe)")
        lines.append("")
        type_in_gt = gt[gt["pand_id"].astype(str).isin(set(intersect["pand_id"].astype(str)))]
        lines.append("| City | SFH | TH | MFH | AB | Total |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for city, grp in type_in_gt.groupby("city"):
            vc = grp["building_type"].value_counts()
            row = [city]
            t = 0
            for cls in ["SFH", "TH", "MFH", "AB"]:
                n = int(vc.get(cls, 0))
                row.append(str(n))
                t += n
            row.append(str(t))
            lines.append("| " + " | ".join(row) + " |")
        vc_total = type_in_gt["building_type"].value_counts()
        t_total = 0
        row = ["**total**"]
        for cls in ["SFH", "TH", "MFH", "AB"]:
            n = int(vc_total.get(cls, 0))
            row.append(f"**{n}**")
            t_total += n
        row.append(f"**{t_total}**")
        lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--gt", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = Path(args.manifest) if args.manifest else (
        repo_root / "data" / "processed" / "svi_manifest.parquet"
    )
    gt_path = Path(args.gt) if args.gt else (
        repo_root / "data" / "processed" / "stage1_gt.parquet"
    )
    out_path = Path(args.out) if args.out else (
        repo_root / "reports" / "stage1" / "manifest_audit_phase_b.md"
    )

    md = build_audit(manifest_path, gt_path if gt_path.exists() else None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    logger.info("wrote %s (%d chars)", out_path, len(md))


if __name__ == "__main__":
    main()
