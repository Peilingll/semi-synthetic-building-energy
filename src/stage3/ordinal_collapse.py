"""#4 — Ordinal metrics + literature-aligned label collapses (RQ, §3.6.2).

Re-bins the existing 7-class route predictions (no retraining):

- ordinal: ±1 accuracy, grade MAE (A=1..G=7)
- binary Mayer 2023:  A–D vs E–G   (their headline macro-F1 64.6%, UK)
- binary Sun 2026:    A–C vs D–G   (their two-city accuracies 0.64 / 0.69)
- 3-class (generic):  A–B / C–D / E–G

The thesis' primary task stays 7-class; these are comparability metrics.

Usage:
    python -m src.stage3.ordinal_collapse                  # pooled
    python -m src.stage3.ordinal_collapse --run-tag loco_amsterdam --routes M0,M1,M3-DINOv2,M3-ResNet50
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "reports" / "stage3"
TABLES = REPO / "reports" / "tables" / "stage3"

DEFAULT_ROUTES = ["M0", "M1", "M3-DINOv2", "M3-ResNet50", "M3-VLMv3",
                  "M2-DINOv2", "M2-ResNet50", "M2-VLM"]
RANK = {l: i + 1 for i, l in enumerate("ABCDEFG")}  # A=1 .. G=7

COLLAPSES = {
    "binary Mayer (A–D | E–G)": lambda l: "A–D" if RANK[l] <= 4 else "E–G",
    "binary Sun (A–C | D–G)": lambda l: "A–C" if RANK[l] <= 3 else "D–G",
    "3-class (A–B | C–D | E–G)": lambda l: ("A–B" if RANK[l] <= 2 else
                                            "C–D" if RANK[l] <= 4 else "E–G"),
}


def route_metrics(df: pd.DataFrame) -> dict:
    t, p = df["true"], df["pred"]
    rt, rp = t.map(RANK), p.map(RANK)
    out = {
        "n": len(df),
        "macro_f1_7": f1_score(t, p, average="macro", zero_division=0),
        "acc_7": accuracy_score(t, p),
        "pm1_acc": float(((rt - rp).abs() <= 1).mean()),
        "grade_mae": float((rt - rp).abs().mean()),
    }
    for name, fn in COLLAPSES.items():
        ct, cp = t.map(fn), p.map(fn)
        out[name] = {"macro_f1": f1_score(ct, cp, average="macro", zero_division=0),
                     "acc": accuracy_score(ct, cp)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default="pooled")
    parser.add_argument("--routes", default=",".join(DEFAULT_ROUTES))
    parser.add_argument("--out-suffix", default="")
    args = parser.parse_args()
    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    tag = "" if args.run_tag == "pooled" else f"_{args.run_tag}"
    tag += args.out_suffix

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    frames = {}
    for r in routes:
        path = REPORTS / f"{r}{tag}_holdout_preds.parquet"
        if not path.exists():
            logger.warning("%s: %s missing, skipping", r, path.name)
            continue
        df = pd.read_parquet(path)
        df["pand_id"] = df["pand_id"].astype(str)
        frames[r] = df

    common = set.intersection(*(set(df["pand_id"]) for df in frames.values()))
    logger.info("common buildings across %d routes: %d", len(frames), len(common))
    results = {r: route_metrics(df[df["pand_id"].isin(common)]) for r, df in frames.items()}

    split_name = "hold-out" if args.run_tag == "pooled" else args.run_tag
    lines = [
        f"# Table — Ordinal metrics + literature-aligned collapses ({split_name}, n={len(common)})",
        "",
        "Primary task stays 7-class A–G; collapses re-bin the SAME predictions to each",
        "reference paper's label granularity. External anchors: Mayer et al. 2023 binary",
        "macro-F1 0.646 (UK, multi-modal); Sun et al. 2026 binary acc 0.64/0.69 (Glasgow/Edinburgh).",
        "",
        "| route | 7c mF1 | 7c acc | ±1 acc | grade MAE | Mayer mF1 | Mayer acc | Sun mF1 | Sun acc | 3c mF1 | 3c acc |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in routes:
        if r not in results:
            continue
        m = results[r]
        may = m["binary Mayer (A–D | E–G)"]
        sun = m["binary Sun (A–C | D–G)"]
        c3 = m["3-class (A–B | C–D | E–G)"]
        lines.append(
            f"| {r} | {m['macro_f1_7']:.4f} | {m['acc_7']:.4f} | {m['pm1_acc']:.4f} "
            f"| {m['grade_mae']:.3f} | {may['macro_f1']:.4f} | {may['acc']:.4f} "
            f"| {sun['macro_f1']:.4f} | {sun['acc']:.4f} | {c3['macro_f1']:.4f} | {c3['acc']:.4f} |"
        )
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / f"T3_ordinal_collapse{tag}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s", out)
    for r in routes:
        if r in results:
            m = results[r]
            print(f"{r:13s} 7c_mF1={m['macro_f1_7']:.4f} pm1={m['pm1_acc']:.4f} "
                  f"mayer_mF1={m['binary Mayer (A–D | E–G)']['macro_f1']:.4f} "
                  f"sun_mF1={m['binary Sun (A–C | D–G)']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
