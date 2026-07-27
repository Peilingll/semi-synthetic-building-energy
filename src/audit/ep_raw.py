"""Shared raw EP-Online loader for the 0.x audits.

Streams the 1.56 GB export once, keeps every certificate whose first BAGPandID
belongs to one of the four study cities, and caches the slice to
`data/interim/ep_four_cities.parquet`.

Certificate-level (NOT deduplicated) — the audits need the per-unit rows.
`pand_id` follows the repo convention: first entry of BAGPandIDs, zfill(16).
"""

import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_EP = REPO_ROOT / "data" / "raw" / "v20260401_v4_csv" / "v20260401_v4_csv.csv"
CACHE = REPO_ROOT / "data" / "interim" / "ep_four_cities.parquet"

# BAG pand_id starts with the gemeentecode of the municipality at registration.
CITY_CODES = {"0363": "amsterdam", "0599": "rotterdam", "0344": "utrecht", "0503": "delft"}

# Raw CSV column -> output column. Numeric ones are parsed with comma decimals.
STR_COLS = {
    "Registratiedatum": "reg_date",
    "Opnamedatum": "survey_date",
    "Berekeningstype": "calc_type",
    "Gebouwklasse": "gebouwklasse",
    "Gebouwtype": "gebouwtype",
    "Gebouwsubtype": "gebouwsubtype",
    "SoortOpname": "soort_opname",
    "Status": "status",
    "Postcode": "postcode",
    "BAGVerblijfsobjectID": "vbo_id",
    "Energieklasse": "energieklasse",
}
NUM_COLS = {
    "Bouwjaar": "bouwjaar",
    "GebruiksoppervlakteThermischeZone": "opp_thermische_zone",
    "Compactheid": "compactheid",
    "Energiebehoefte": "energiebehoefte",
    "PrimaireFossieleEnergie": "primaire_fossiele_energie",
    "AandeelHernieuwbareEnergie": "aandeel_hernieuwbare_energie",
    "Warmtebehoefte": "warmtebehoefte",
    "EnergieIndex": "energie_index",
    # EMG forfaitair variants: used when external heat delivery (EMG) applies —
    # the registered label can be based on these instead of the plain columns.
    "PrimaireFossieleEnergieEMGForfaitair": "pf_emg_forfaitair",
    "AandeelHernieuwbareEnergieEMGForfaitair": "aandeel_emg_forfaitair",
    "Temperatuuroverschrijding": "temperatuuroverschrijding",
}


def _num(x: str) -> float:
    x = (x or "").strip().replace(",", ".")
    try:
        return float(x)
    except ValueError:
        return np.nan


def build_cache(force: bool = False) -> Path:
    if CACHE.exists() and not force:
        return CACHE
    CACHE.parent.mkdir(parents=True, exist_ok=True)

    with open(RAW_EP, encoding="utf-8-sig", newline="") as f:
        f.readline()  # PublicatieDatum
        f.readline()  # LaatstVerwerkteMutatievolgnummer
        header = f.readline().rstrip("\n").split(";")
        idx = {c: i for i, c in enumerate(header)}
        i_pand = idx["BAGPandIDs"]
        str_idx = [(idx[c], out) for c, out in STR_COLS.items()]
        num_idx = [(idx[c], out) for c, out in NUM_COLS.items()]
        need = max([i_pand] + [i for i, _ in str_idx] + [i for i, _ in num_idx])

        rows, n_total = [], 0
        for row in csv.reader(f, delimiter=";"):
            n_total += 1
            if len(row) <= need:
                continue
            raw_pands = str(row[i_pand])
            pid = raw_pands.split(",")[0].strip().zfill(16)
            city = CITY_CODES.get(pid[:4])
            if city is None:
                continue
            rec = {"pand_id": pid, "city": city,
                   "n_pand_ids": raw_pands.count(",") + 1 if raw_pands else 0}
            rec.update({out: row[i].strip() for i, out in str_idx})
            rec.update({out: _num(row[i]) for i, out in num_idx})
            rows.append(rec)

    df = pd.DataFrame(rows)
    logger.info("raw EP rows scanned: %d; four-city certificates kept: %d", n_total, len(df))
    df.to_parquet(CACHE, index=False)
    return CACHE


def load(residential_only: bool = True, nta_only: bool = False) -> pd.DataFrame:
    """Four-city certificate table.

    residential_only: Gebouwklasse == 'W' (woningbouw).
    nta_only:         Berekeningstype starts with 'NTA 8800'.
    """
    build_cache()
    df = pd.read_parquet(CACHE)
    if residential_only:
        df = df[df["gebouwklasse"].eq("W")]
    if nta_only:
        df = df[df["calc_type"].str.startswith("NTA 8800", na=False)]
    return df.reset_index(drop=True)


def merge_a_classes(s: pd.Series) -> pd.Series:
    """A+ / A++ / A+++ / A++++ -> A (repo convention, see stage2.features)."""
    s = s.astype(str).str.strip()
    return s.where(~s.str.startswith("A"), "A")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = build_cache(force=True)
    d = pd.read_parquet(p)
    logger.info("cached %s  rows=%d", p, len(d))
    logger.info("by city:\n%s", d["city"].value_counts().to_string())
    logger.info("by gebouwklasse:\n%s", d["gebouwklasse"].value_counts().to_string())
    logger.info("calc_type top:\n%s", d["calc_type"].value_counts().head(8).to_string())
