"""Cell-only oracle: how much EPC-label information does the TABULA cell carry?

For each TABULA-NL cell (building_type x tabula_period), take the modal EPC
label among development-set buildings, then predict that label for every
hold-out building using its REFERENCE (registry-derived) cell. Because the
true cell is used, this is the ceiling of any predictor that consumes only
the archetype cell — upstream vision quality is irrelevant here.

Reported for the 7-class task (A+..A++++ merged into A) and the binary task
(A-C vs D-G). Anchors: M0 (development majority) below, M1 (eight-feature
LightGBM on reference attributes) above.

Usage: .venv/Scripts/python.exe scripts/compute_cell_oracle.py
Output: reports/stage2/cell_oracle.json
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

REPO = Path(__file__).resolve().parents[1]

EPC_ORDER = ["A", "B", "C", "D", "E", "F", "G"]


def merge_epc(label: str) -> str:
    return "A" if str(label).startswith("A") else str(label)


def evaluate(dev: pd.DataFrame, hold: pd.DataFrame, target: str) -> dict:
    modal = dev.groupby("cell")[target].agg(lambda s: s.mode().iloc[0])
    fallback = dev[target].mode().iloc[0]
    pred = hold["cell"].map(modal).fillna(fallback)
    out = {
        "accuracy": round(accuracy_score(hold[target], pred), 4),
        "macro_f1": round(f1_score(hold[target], pred, average="macro"), 4),
        "n_cells_in_dev": int(modal.size),
        "cell_modal_labels": modal.to_dict(),
    }
    if target == "epc7":
        idx = {c: i for i, c in enumerate(EPC_ORDER)}
        dist = (hold[target].map(idx) - pred.map(idx)).abs()
        out["within_1_accuracy"] = round(float((dist <= 1).mean()), 4)
    return out


def main() -> None:
    pool = pd.read_parquet(REPO / "data/processed/stage1_gt.parquet")
    manifest = pd.read_parquet(REPO / "data/processed/svi_manifest.parquet")
    hold_ids = pd.read_parquet(REPO / "data/processed/holdout_test_pand_ids.parquet")

    pool["pand_id"] = pool["pand_id"].astype(str)
    sample = pool[pool["pand_id"].isin(set(manifest["pand_id"].astype(str)))].copy()
    sample["cell"] = sample["building_type"] + "|" + sample["tabula_period"]
    sample["epc7"] = sample["Energieklasse"].map(merge_epc)
    sample["epc_bin"] = sample["epc7"].map(lambda c: "A-C" if c in ("A", "B", "C") else "D-G")

    hold_set = set(hold_ids["pand_id"].astype(str))
    hold = sample[sample["pand_id"].isin(hold_set)]
    dev = sample[~sample["pand_id"].isin(hold_set)]
    print(f"dev {len(dev)}, holdout {len(hold)} (expected 8,068 / 2,018)")

    result = {
        "n_dev": int(len(dev)),
        "n_holdout": int(len(hold)),
        "seven_class": evaluate(dev, hold, "epc7"),
        "binary": evaluate(dev, hold, "epc_bin"),
    }
    for task in ("seven_class", "binary"):
        print(task, {k: v for k, v in result[task].items() if k != "cell_modal_labels"})

    with open(REPO / "reports/stage2/cell_oracle.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
