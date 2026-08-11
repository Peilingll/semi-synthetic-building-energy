"""Stage 2 feature builder.

Builds the GT-feature matrix for the downstream Energieklasse classifier from
`stage1_gt.parquet`, restricted to the frozen dev set (8,068 buildings) with
the same 5-fold indices Stage 1 used (`dev_fold_indices.parquet`).

U-values are looked up deterministically from `tabula_nl.csv` on
(building_type, tabula_period) — the same join `tabula_matcher.match_tabula`
uses — so no per-city `residential_tabula_matched.parquet` is needed.

Label: `Energieklasse` with A+/A++/A+++/A++++ merged into A → 7 ordinal classes
A < B < C < D < E < F < G.

Feature sets (per ablation run):
- Cumulative: S_min ⊂ S_lookup ⊂ S_full
- Leave-one-out from S_full: drops year / type / floor / u_values / postcode
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"

# Ordinal energy classes (A best ... G worst) — order matters for quadratic kappa.
ENERGY_LABELS = ["A", "B", "C", "D", "E", "F", "G"]

# Binary task (decision 2026-08-10): the Sun et al. 2026 cut, A-C vs D-G, which
# falls on the NTA 8800 boundary C <= 250 kWh/m2.yr. Positive class = D-G, the
# side that needs retrofit, so precision/recall/AUC read the way policy does.
# ASCII labels on purpose — these reach the Windows console via run_stage3.
BINARY_LABELS = ["A-C", "D-G"]
BINARY_POSITIVE = "D-G"
BINARY_CUT_KWH = 250.0  # EP2 upper bound of class C

TASKS = ("7class", "binary")
TARGET_COL = {"7class": "energy_class", "binary": "energy_binary"}


def labels_for(task: str) -> list[str]:
    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}; valid: {list(TASKS)}")
    return ENERGY_LABELS if task == "7class" else BINARY_LABELS


def target_col(task: str) -> str:
    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}; valid: {list(TASKS)}")
    return TARGET_COL[task]


def to_binary(energy_class: pd.Series) -> pd.Series:
    """Collapse 7-class A..G to the Sun cut: A/B/C -> 'A-C', D/E/F/G -> 'D-G'."""
    return pd.Series(
        [BINARY_LABELS[0] if str(c) in ("A", "B", "C") else BINARY_LABELS[1]
         for c in energy_class],
        index=energy_class.index, dtype=object,
    )

U_COLS = ["u_wall", "u_roof", "u_floor", "u_window"]
CATEGORICAL = ["building_type", "city"]

# Each run -> ordered list of feature columns.
S_MIN = ["building_type", "bouwjaar"]
S_LOOKUP = S_MIN + U_COLS
S_FULL = S_LOOKUP + ["num_floors", "city"]

RUN_FEATURES: dict[str, list[str]] = {
    "S_min": S_MIN,
    "S_lookup": S_LOOKUP,
    "S_full": S_FULL,
    "S_full-year": [c for c in S_FULL if c != "bouwjaar"],
    "S_full-type": [c for c in S_FULL if c != "building_type"],
    "S_full-floor": [c for c in S_FULL if c != "num_floors"],
    "S_full-u_values": [c for c in S_FULL if c not in U_COLS],
    "S_full-postcode": [c for c in S_FULL if c != "city"],
}


def merge_energy_class(energieklasse: pd.Series) -> pd.Series:
    """Merge A+ / A++ / A+++ / A++++ into A; leave B..G unchanged."""
    s = energieklasse.astype(str).str.strip()
    merged = s.where(~s.str.startswith("A"), "A")
    return merged


def build_master_table(
    gt_path: Path | None = None,
    dev_path: Path | None = None,
    tabula_path: Path | None = None,
) -> pd.DataFrame:
    """Return one row per dev building with all candidate features, label, fold.

    Columns: pand_id, fold, city, building_type, bouwjaar, num_floors,
             u_wall, u_roof, u_floor, u_window, energy_class
    """
    gt_path = gt_path or PROCESSED / "stage1_gt.parquet"
    dev_path = dev_path or PROCESSED / "dev_fold_indices.parquet"
    tabula_path = tabula_path or PROCESSED / "tabula_nl.csv"

    gt = pd.read_parquet(gt_path)
    gt["pand_id"] = gt["pand_id"].astype(str)
    dev = pd.read_parquet(dev_path)
    dev["pand_id"] = dev["pand_id"].astype(str)

    # Restrict to dev set, attach fold.
    df = dev[["pand_id", "fold"]].merge(gt, on="pand_id", how="left")
    n_missing = df["building_type"].isna().sum()
    if n_missing:
        raise ValueError(f"{n_missing} dev pand_ids not found in stage1_gt")

    # U-value lookup (deterministic on building_type x tabula_period).
    tabula = pd.read_csv(tabula_path)
    df = df.merge(
        tabula[["building_type", "period"] + U_COLS],
        left_on=["building_type", "tabula_period"],
        right_on=["building_type", "period"],
        how="left",
    ).drop(columns=["period"])
    n_no_u = df["u_wall"].isna().sum()
    if n_no_u:
        raise ValueError(f"{n_no_u} rows missing U-values after TABULA join")

    df["energy_class"] = merge_energy_class(df["Energieklasse"])
    bad = ~df["energy_class"].isin(ENERGY_LABELS)
    if bad.any():
        raise ValueError(f"{bad.sum()} rows have unexpected energy_class: "
                         f"{df.loc[bad, 'energy_class'].unique().tolist()}")

    df["energy_binary"] = to_binary(df["energy_class"])

    out = df[[
        "pand_id", "fold", "city", "building_type", "bouwjaar",
        "num_floors", *U_COLS, "energy_class", "energy_binary",
    ]].copy()

    # Cast categoricals so LightGBM picks them up natively.
    for c in CATEGORICAL:
        out[c] = out[c].astype("category")

    logger.info("master table: %d buildings, %d folds, classes=%s",
                len(out), out["fold"].nunique(),
                out["energy_class"].value_counts().reindex(ENERGY_LABELS).to_dict())
    return out


def feature_matrix(master: pd.DataFrame, run: str) -> tuple[pd.DataFrame, list[str]]:
    """Slice the master table to one run's feature columns.

    Returns (X, categorical_features_present).
    """
    if run not in RUN_FEATURES:
        raise KeyError(f"unknown run {run!r}; valid: {list(RUN_FEATURES)}")
    cols = RUN_FEATURES[run]
    X = master[cols].copy()
    cat_present = [c for c in CATEGORICAL if c in cols]
    return X, cat_present
