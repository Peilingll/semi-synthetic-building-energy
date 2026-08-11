"""Stage 2 full ablation: M0 + 3 cumulative + 5 leave-one-out runs.

Produces Table 2a (cumulative) and Table 2b (leave-one-out) as markdown, plus
a per-class table and confusion matrix for S_full.

Usage:
    uv run python -m src.stage2.run_ablation
"""

import argparse
import logging

from src.stage2.features import REPO_ROOT, build_master_table, labels_for
from src.stage2.train_eval import run_single, task_suffix

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


def _title(task: str, n: int) -> str:
    return (f"(pooled 5-fold OOF, n={n:,})" if task == "7class"
            else f"(binary A-C | D-G, pooled 5-fold OOF, n={n:,})")


def _extra_cols(task: str) -> tuple[str, str]:
    """Binary adds balanced acc / MCC / AUC; 7-class keeps the original columns."""
    if task != "binary":
        return "", ""
    return " Balanced acc | MCC | ROC-AUC |", "---:|---:|---:|"


def _extra_vals(r: dict, task: str) -> str:
    if task != "binary":
        return ""
    auc = r.get("roc_auc")
    return (f" {r.get('balanced_accuracy', float('nan')):.4f} | "
            f"{r.get('mcc', float('nan')):.4f} | "
            f"{'—' if auc is None else f'{auc:.4f}'} |")


def write_table_2a(reports: dict[str, dict], task: str, n: int) -> str:
    base = reports["S_min"]["macro_f1"]
    hx, hy = _extra_cols(task)
    lines = [
        f"# Table 2a — Stage 2 Cumulative Ablation {_title(task, n)}",
        "",
        "| Run | Macro-F1 | 95% CI | Quadratic κ | Accuracy |" + hx
        + " Δ macro-F1 from S_min |",
        "|---|---:|---|---:|---:|" + hy + "---:|",
    ]
    for run in CUMULATIVE:
        r = reports[run]
        delta = "(baseline)" if run == "S_min" else (
            "—" if run == "M0" else fmt(r["macro_f1"] - base))
        lines.append(
            f"| {run} | {r['macro_f1']:.4f} | {_ci(r, 'macro_f1')} | "
            f"{r['quadratic_kappa']:.4f} | {r['accuracy']:.4f} |"
            + _extra_vals(r, task) + f" {delta} |")
    if task == "binary":
        lines += ["", "Accuracy must be read against M0: the pool is ~70/30, so a "
                  "constant prediction already scores acc ≈ 0.70 at macro-F1 ≈ 0.41."]
    return "\n".join(lines) + "\n"


def write_table_2b(reports: dict[str, dict], task: str, n: int) -> str:
    full = reports["S_full"]["macro_f1"]
    lines = [
        f"# Table 2b — Stage 2 Leave-One-Out from S_full {_title(task, n)}",
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


def write_per_class_s_full(report: dict, task: str) -> str:
    labels = labels_for(task)
    lines = [
        f"# Table 2 — S_full per-class precision/recall {_title(task, report['n_eval'])}",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for cls in labels:
        pc = report["per_class"][cls]
        lines.append(f"| {cls} | {pc['precision']:.3f} | {pc['recall']:.3f} | "
                     f"{pc['f1']:.3f} | {pc['support']} |")
    lines += ["", "## Confusion matrix (rows = true, cols = pred)", "",
              "| true\\pred | " + " | ".join(labels) + " |",
              "|---|" + "---:|" * len(labels)]
    for lab, row in zip(report["confusion_matrix"]["labels"],
                        report["confusion_matrix"]["matrix"]):
        lines.append(f"| {lab} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="7class", choices=["7class", "binary"])
    args = parser.parse_args()
    task = args.task
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    master = build_master_table()

    reports: dict[str, dict] = {}
    for run in CUMULATIVE + LEAVE_ONE_OUT:
        reports[run] = run_single(run, master=master, with_ci=True, save=True, task=task)

    sfx, n = task_suffix(task), len(master)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / f"T2a_cumulative{sfx}.md").write_text(
        write_table_2a(reports, task, n), encoding="utf-8")
    (TABLES_DIR / f"T2b_leave_one_out{sfx}.md").write_text(
        write_table_2b(reports, task, n), encoding="utf-8")
    (TABLES_DIR / f"T2_per_class_s_full{sfx}.md").write_text(
        write_per_class_s_full(reports["S_full"], task), encoding="utf-8")
    logger.info("wrote tables to %s", TABLES_DIR)

    # ASCII-safe console summary (the .md files carry the unicode tables):
    # a Windows cp950 console cannot encode the kappa/delta/approx signs.
    print(f"\n{'run':18s} {'macroF1':>8s} {'kappa':>7s} {'acc':>7s}")
    for run in CUMULATIVE + LEAVE_ONE_OUT:
        r = reports[run]
        print(f"{run:18s} {r['macro_f1']:8.4f} {r['quadratic_kappa']:7.4f} {r['accuracy']:7.4f}")


if __name__ == "__main__":
    main()
