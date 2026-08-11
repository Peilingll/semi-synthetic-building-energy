"""Extract real continuous energy use (primary fossil energy, kWh/m2.yr) from
the raw EP-Online CSV, keyed by pand_id, for the regression experiments.

Which column (audit A04, fixed 2026-08-10): the register bins
`PrimaireFossieleEnergieEMGForfaitair` to produce `Energieklasse` (99.985%
agreement), NOT the plain `PrimaireFossieleEnergie` (78.4%). Regressing the
plain column made the implied label disagree with the registered one for 3.89%
of certificates, concentrated in the district-heating stock that dominates
Amsterdam and Rotterdam, with a median gap of 45.5 kWh/m2.yr — the same order as
the reported model MAE. So: take the forfaitair column where it is filled and
fall back to the plain one otherwise. This matters twice over for the binary
task, where a single threshold at 250 kWh/m2.yr decides the class.

The EP-Online export has two metadata rows before the real header (row index 2).
We keep residential (Gebouwklasse == 'W'), valid PF in [0, 1000], and the most
recent record per pand_id (by Registratiedatum).

Output: data/processed/ep_kwh.parquet
        [pand_id, pf_kwh, pf_plain, pf_source, energieklasse_raw]
"""

import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw" / "v20260401_v4_csv" / "v20260401_v4_csv.csv"
OUT = REPO / "data" / "processed" / "ep_kwh.parquet"

PF_MIN, PF_MAX = 0.0, 1000.0


def _num(x: str) -> float:
    x = (x or "").strip().replace(",", ".")
    try:
        return float(x)
    except ValueError:
        return np.nan


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = []
    with open(RAW, encoding="utf-8-sig") as f:
        f.readline()  # PublicatieDatum
        f.readline()  # LaatstVerwerkteMutatievolgnummer
        header = f.readline().rstrip("\n").split(";")
        idx = {h: i for i, h in enumerate(header)}
        iP, iPF, iEMG, iEK, iGk, iReg = (
            idx["BAGPandIDs"], idx["PrimaireFossieleEnergie"],
            idx["PrimaireFossieleEnergieEMGForfaitair"],
            idx["Energieklasse"], idx["Gebouwklasse"], idx["Registratiedatum"])
        need = max(iP, iPF, iEMG, iEK, iGk, iReg)
        r = csv.reader(f, delimiter=";")
        for row in r:
            if len(row) <= need:
                continue
            if row[iGk].strip() != "W":  # residential only
                continue
            plain = _num(row[iPF])
            emg = _num(row[iEMG])
            # the column the register itself bins, with fallback (A04)
            pf = emg if np.isfinite(emg) else plain
            if not (PF_MIN <= pf <= PF_MAX):
                continue
            pid = str(row[iP]).split(",")[0].zfill(16)
            rows.append((pid, pf, plain, "emg_forfaitair" if np.isfinite(emg) else "plain",
                         row[iEK].strip(), row[iReg].strip()))

    ep = pd.DataFrame(rows, columns=["pand_id", "pf_kwh", "pf_plain", "pf_source",
                                     "energieklasse_raw", "reg"])
    logger.info("residential PF rows in [%g,%g]: %d", PF_MIN, PF_MAX, len(ep))
    # keep most recent registration per pand_id
    ep = ep.sort_values("reg").drop_duplicates("pand_id", keep="last").drop(columns="reg")
    ep.to_parquet(OUT, index=False)
    logger.info("wrote %s: %d unique pand_id", OUT, len(ep))
    logger.info("pf_source: %s", ep["pf_source"].value_counts().to_dict())
    logger.info("pf_kwh describe: %s",
                ep["pf_kwh"].describe()[["min", "25%", "50%", "75%", "max"]].round(1).to_dict())
    gap = (ep["pf_kwh"] - ep["pf_plain"]).abs()
    moved = (gap > 1e-9).sum()
    logger.info("rows where the corrected column differs from the plain one: %d (%.2f%%), "
                "median gap %.1f kWh/m2.yr", moved, 100 * moved / len(ep),
                float(gap[gap > 1e-9].median()) if moved else 0.0)


if __name__ == "__main__":
    main()
