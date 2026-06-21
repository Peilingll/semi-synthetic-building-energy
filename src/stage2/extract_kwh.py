"""Extract real continuous energy use (PrimaireFossieleEnergie, kWh/m2.yr) from
the raw EP-Online CSV, keyed by pand_id, for the regression experiments.

The EP-Online export has two metadata rows before the real header (row index 2).
We keep residential (Gebouwklasse == 'W'), valid PF in [0, 1000], and the most
recent record per pand_id (by Registratiedatum).

Output: data/processed/ep_kwh.parquet  [pand_id, pf_kwh, energieklasse_raw]
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
        iP, iPF, iEK, iGk, iReg = (idx["BAGPandIDs"], idx["PrimaireFossieleEnergie"],
                                   idx["Energieklasse"], idx["Gebouwklasse"], idx["Registratiedatum"])
        r = csv.reader(f, delimiter=";")
        for row in r:
            if len(row) <= max(iP, iPF, iEK, iGk, iReg):
                continue
            if row[iGk].strip() != "W":  # residential only
                continue
            pf = _num(row[iPF])
            if not (PF_MIN <= pf <= PF_MAX):
                continue
            pid = str(row[iP]).split(",")[0].zfill(16)
            rows.append((pid, pf, row[iEK].strip(), row[iReg].strip()))

    ep = pd.DataFrame(rows, columns=["pand_id", "pf_kwh", "energieklasse_raw", "reg"])
    logger.info("residential PF rows in [%g,%g]: %d", PF_MIN, PF_MAX, len(ep))
    # keep most recent registration per pand_id
    ep = ep.sort_values("reg").drop_duplicates("pand_id", keep="last").drop(columns="reg")
    ep.to_parquet(OUT, index=False)
    logger.info("wrote %s: %d unique pand_id", OUT, len(ep))
    logger.info("pf_kwh describe: %s",
                ep["pf_kwh"].describe()[["min", "25%", "50%", "75%", "max"]].round(1).to_dict())


if __name__ == "__main__":
    main()
