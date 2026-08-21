"""Expected metrics of a uniform random-guessing baseline (new M0).

For each task the guess is drawn uniformly over the label space, independent of
the building. All values are analytic expectations (no seed needed):
  accuracy      = 1/K
  precision_c   = p_c   (share of true c among random predictions)
  recall_c      = 1/K
  F1_c          = 2 * p_c * (1/K) / (p_c + 1/K)
  +-1 accuracy  = E[ #classes within +-1 of true / K ]
Rate-matched operating point (binary): a random score ranking marks a random
29.7% subset as D-G, so recall_DG = k, precision_DG = p_DG.

Evaluated on the pooled hold-out (n=2,018) and the LOCO Amsterdam test pool
(n=8,011).

Usage: .venv/Scripts/python.exe scripts/compute_random_baseline.py
Output: reports/stage2/random_baseline.json
"""

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EPC_ORDER = ["A", "B", "C", "D", "E", "F", "G"]


def merge_epc(label: str) -> str:
    return "A" if str(label).startswith("A") else str(label)


def f1(prec: float, rec: float) -> float:
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def seven_class(labels: pd.Series) -> dict:
    k = len(EPC_ORDER)
    p = labels.value_counts(normalize=True)
    macro = sum(f1(p.get(c, 0.0), 1 / k) for c in EPC_ORDER) / k
    within1 = sum(
        p.get(c, 0.0) * len([d for d in range(7) if abs(d - i) <= 1]) / k
        for i, c in enumerate(EPC_ORDER)
    )
    return {"accuracy": round(1 / k, 4), "macro_f1": round(macro, 4),
            "within_1_accuracy": round(within1, 4)}


def binary(labels: pd.Series, rate: float) -> dict:
    p_pos = float((labels == "D-G").mean())
    macro_05 = (f1(p_pos, 0.5) + f1(1 - p_pos, 0.5)) / 2
    macro_rate = (f1(p_pos, rate) + f1(1 - p_pos, 1 - rate)) / 2
    return {"accuracy_at_0.5": 0.5, "macro_f1_at_0.5": round(macro_05, 4),
            "macro_f1_at_rate": round(macro_rate, 4), "roc_auc": 0.5,
            "positive_prevalence": round(p_pos, 4)}


def main() -> None:
    pool = pd.read_parquet(REPO / "data/processed/stage1_gt.parquet")
    manifest = pd.read_parquet(REPO / "data/processed/svi_manifest.parquet")
    hold_ids = pd.read_parquet(REPO / "data/processed/holdout_test_pand_ids.parquet")

    pool["pand_id"] = pool["pand_id"].astype(str)
    sample = pool[pool["pand_id"].isin(set(manifest["pand_id"].astype(str)))].copy()
    sample["epc7"] = sample["Energieklasse"].map(merge_epc)
    sample["epc_bin"] = sample["epc7"].map(lambda c: "A-C" if c in ("A", "B", "C") else "D-G")

    hold = sample[sample["pand_id"].isin(set(hold_ids["pand_id"].astype(str)))]
    ams = sample[sample["city"] == "amsterdam"]
    dev = sample[~sample["pand_id"].isin(set(hold_ids["pand_id"].astype(str)))]
    rate = float((dev["epc_bin"] == "D-G").mean())

    out = {
        "pooled_holdout": {"n": len(hold),
                           "seven_class": seven_class(hold["epc7"]),
                           "binary": binary(hold["epc_bin"], rate)},
        "loco_amsterdam": {"n": len(ams),
                           "seven_class": seven_class(ams["epc7"]),
                           "binary": binary(ams["epc_bin"], rate)},
        "rate_from_dev": round(rate, 4),
    }
    print(json.dumps(out, indent=2))
    with open(REPO / "reports/stage2/random_baseline.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
