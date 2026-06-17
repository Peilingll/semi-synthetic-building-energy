"""Stage 2 full ablation: M0 + 3 cumulative + 5 leave-one-out runs.

Produces Table 2a (cumulative) and Table 2b (leave-one-out) as markdown, plus
a per-class table and confusion matrix for S_full.

Usage:
    uv run python -m src.stage2.run_ablation
"""

import json
import logging
from pathlib import Path

import pandas as pd

from src.stage2.features import ENERGY_LABELS, REPO_ROOT, build_master_table
from src.stage2.train_eval import run_single

logger = logging.getLogger(__name__)

TABLES_DIR = REPO_ROOT / "reports" / "tables" / "stage2"

CUMULATIVE = ["M0", "S_min", "S_lookup", "S_full"]
LEAVE_ONE_OUT = ["S_full-year", "S_full-type", "S_full-floor",
                 "S_full-u_values", "S_full-postcode"]
LOO_INTERP = {
    "S_full-year": "Cost of losing year",
    "S_full-type": "Cost of losing type",
    "S_full-floor": "Cost of losing floor",
    "S_full-u_values": "Value of TABULA U-value lookup",
    "S_full-postcode": "Geographic (city) effect",
}


def _ci(report: dict, metric: str) -> str:
    ci = report.get("bootstrap_95ci", {}).get(metric)
    if not ci:
        return ""
    return f"[{ci['lo']:.3f}, {ci['hi']:.3f}]"


def fmt(x: float) -> str:
    return f"{x:+.4f}"


def write_table_2a(reports: dict[str, dict]) -> str:
    base = reports["S_min"]["macro_f1"]
    lines = [
        "# Table 2a — Stage 2 Cumulative Ablation (pooled 5-fold OOF, n=8,068)",
        "",
        "| Run | Macro-F1 | 95% CI | Quadratic κ | Accuracy | Δ macro-F1 from S_min |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for run in CUMULATIVE:
        r = reports[run]
        delta = "(baseline)" if run == "S_min" else (
            "—" if run == "M0" else fmt(r["macro_f1"] - base))
        lines.append(
            f"| {run} | {r['macro_f1']:.4f} | {_ci(r, 'macro_f1')} | "
            f"{r['quadratic_kappa']:.4f} | {r['accuracy']:.4f} | {delta} |")
    return "\n".join(lines) + "\n"


def write_table_2b(reports: dict[str, dict]) -> str:
    full = reports["S_full"]["macro_f1"]
    lines = [
        "# Table 2b — Stage 2 Leave-One-Out from S_full (pooled 5-fold OOF, n=8,068)",
        "",
        "| Run | Macro-F1 | Quadratic κ | Δ macro-F1 from S_full | Interpretation |",
        "|---|---:|---:|---:|---|",
        f"| S_full | {reports['S_full']['macro_f1']:.4f} | "
        f"{reports['S_full']['quadratic_kappa']:.4f} | (ceiling) | |",
    ]
    for run in LEAVE_ONE_OUT:
        r = reports[run]
        lines.append(
            f"| {run} | {r['macro_f1']:.4f} | {r['quadratic_kappa']:.4f} | "
            f"{fmt(r['macro_f1'] - full)} | {LOO_INTERP[run]} |")
    return "\n".join(lines) + "\n"


def write_per_class_s_full(report: dict) -> str:
    lines = [
        "# Table 2 — S_full per-class precision/recall (pooled 5-fold OOF)",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for cls in ENERGY_LABELS:
        pc = report["per_class"][cls]
        lines.append(f"| {cls} | {pc['precision']:.3f} | {pc['recall']:.3f} | "
                     f"{pc['f1']:.3f} | {pc['support']} |")
    lines += ["", "## Confusion matrix (rows = true, cols = pred)", "",
              "| true\\pred | " + " | ".join(ENERGY_LABELS) + " |",
              "|---|" + "---:|" * len(ENERGY_LABELS)]
    for lab, row in zip(report["confusion_matrix"]["labels"],
                        report["confusion_matrix"]["matrix"]):
        lines.append(f"| {lab} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    master = build_master_table()

    reports: dict[str, dict] = {}
    for run in CUMULATIVE + LEAVE_ONE_OUT:
        reports[run] = run_single(run, master=master, with_ci=True, save=True)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "T2a_cumulative.md").write_text(write_table_2a(reports), encoding="utf-8")
    (TABLES_DIR / "T2b_leave_one_out.md").write_text(write_table_2b(reports), encoding="utf-8")
    (TABLES_DIR / "T2_per_class_s_full.md").write_text(
        write_per_class_s_full(reports["S_full"]), encoding="utf-8")
    logger.info("wrote tables to %s", TABLES_DIR)

    print("\n" + write_table_2a(reports))
    print(write_table_2b(reports))


if __name__ == "__main__":
    main()
