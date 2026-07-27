"""Intra-pand EPC label entropy audit (#6).

Dutch EPCs are issued per dwelling unit (verblijfsobject); our unit of analysis
is the building (pand). In an 88%-AB dev pool one pand holds many units whose
labels may disagree (renovated flat A next to original D). If intra-pand
disagreement is high, a single pand-level label is ill-defined and part of the
low downstream ceiling is irreducible label noise ("observation-unit
mismatch"), not missing features.

From the raw EP-Online CSV (ALL residential certificates, no dedup), restricted
to manifest pand_ids, per pand:
  n_certs, n_unique labels, modal (majority) share, entropy(bits),
  latest label (the pipeline's y convention), latest == modal, PF spread.

Reported for two universes: (a) all certificates, (b) NTA 8800 era only
(Registratiedatum >= 2021, matching the pipeline's EPC filter).

Oracle ceiling: mean modal share over pands = accuracy of a perfect model that
predicts each pand's modal label, scored against a randomly drawn unit label.
Against the pipeline's latest-label y, the analogous bound is P(latest==modal).

Output: reports/tables/stage2/T2d_label_entropy.md
"""

import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw" / "v20260401_v4_csv" / "v20260401_v4_csv.csv"
MANIFEST = REPO / "data" / "processed" / "svi_manifest.parquet"
GT = REPO / "data" / "processed" / "stage1_gt.parquet"
OUT = REPO / "reports" / "tables" / "stage2" / "T2d_label_entropy.md"

VALID = {"A", "B", "C", "D", "E", "F", "G"}


def _norm_label(x: str) -> str | None:
    x = (x or "").strip().upper()
    if x.startswith("A"):
        x = "A"  # merge A+..A++++ into A (same rule as the pipeline)
    return x if x in VALID else None


def _num(x: str) -> float:
    try:
        return float((x or "").strip().replace(",", "."))
    except ValueError:
        return np.nan


def load_certificates(pand_ids: set) -> pd.DataFrame:
    rows = []
    with open(RAW, encoding="utf-8-sig") as f:
        f.readline()  # PublicatieDatum
        f.readline()  # LaatstVerwerkteMutatievolgnummer
        header = f.readline().rstrip("\n").split(";")
        idx = {h: i for i, h in enumerate(header)}
        iP, iEK, iGk, iReg, iPF = (idx["BAGPandIDs"], idx["Energieklasse"],
                                   idx["Gebouwklasse"], idx["Registratiedatum"],
                                   idx["PrimaireFossieleEnergie"])
        for row in csv.reader(f, delimiter=";"):
            if len(row) <= max(iP, iEK, iGk, iReg, iPF):
                continue
            if row[iGk].strip() != "W":
                continue
            label = _norm_label(row[iEK])
            if label is None:
                continue
            pid = str(row[iP]).split(",")[0].zfill(16)
            if pid not in pand_ids:
                continue
            reg = row[iReg].strip()
            try:
                reg_year = int(reg[:4])
            except ValueError:
                continue
            rows.append((pid, label, reg, reg_year, _num(row[iPF])))
    df = pd.DataFrame(rows, columns=["pand_id", "label", "reg", "reg_year", "pf"])
    logger.info("certificates on manifest pands: %d rows / %d pands",
                len(df), df["pand_id"].nunique())
    return df


def per_pand(df: pd.DataFrame) -> pd.DataFrame:
    def _agg(g: pd.DataFrame) -> pd.Series:
        counts = g["label"].value_counts()
        p = counts / counts.sum()
        modal = counts.index[0]
        latest = g.sort_values("reg")["label"].iloc[-1]
        pf = g["pf"].dropna()
        return pd.Series({
            "n_certs": len(g),
            "n_unique": len(counts),
            "modal_share": counts.iloc[0] / len(g),
            "entropy_bits": float(-(p * np.log2(p)).sum()),
            "modal": modal,
            "latest": latest,
            "latest_eq_modal": latest == modal,
            "pf_std": pf.std() if len(pf) > 1 else np.nan,
        })

    return df.groupby("pand_id").apply(_agg, include_groups=False)


def summarize(pp: pd.DataFrame, gt_type: pd.Series, title: str) -> list[str]:
    multi = pp[pp["n_certs"] >= 2]
    disag = multi[multi["n_unique"] > 1]
    lines = [
        f"## {title}",
        "",
        f"- pands with >=1 certificate: **{len(pp)}**",
        f"- pands with >=2 certificates: **{len(multi)}** ({len(multi)/len(pp):.1%})",
        f"- certificates per pand (all): median {pp['n_certs'].median():.0f}, "
        f"mean {pp['n_certs'].mean():.1f}, max {pp['n_certs'].max():.0f}",
        f"- among multi-cert pands, >=2 distinct labels: **{len(disag)}** "
        f"({len(disag)/len(multi):.1%} of multi)" if len(multi) else "",
        "",
        "| quantity | all pands | multi-cert pands only |",
        "|---|---:|---:|",
        f"| mean modal share (oracle acc ceiling) | **{pp['modal_share'].mean():.4f}** "
        f"| {multi['modal_share'].mean():.4f} |",
        f"| median modal share | {pp['modal_share'].median():.4f} "
        f"| {multi['modal_share'].median():.4f} |",
        f"| p25 modal share | {pp['modal_share'].quantile(.25):.4f} "
        f"| {multi['modal_share'].quantile(.25):.4f} |",
        f"| mean entropy (bits) | {pp['entropy_bits'].mean():.4f} "
        f"| {multi['entropy_bits'].mean():.4f} |",
        f"| P(latest == modal) | {pp['latest_eq_modal'].mean():.4f} "
        f"| {multi['latest_eq_modal'].mean():.4f} |",
        f"| median intra-pand PF std (kWh/m2.yr) | — "
        f"| {multi['pf_std'].median():.1f} |",
        "",
        "### by building type",
        "",
        "| type | pands | % multi-cert | median n_certs | mean modal share | mean entropy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    bt = gt_type.reindex(pp.index)
    for t in ["AB", "TH", "SFH", "MFH"]:
        sub = pp[bt == t]
        if not len(sub):
            continue
        lines.append(
            f"| {t} | {len(sub)} | {(sub['n_certs'] >= 2).mean():.1%} "
            f"| {sub['n_certs'].median():.0f} | {sub['modal_share'].mean():.4f} "
            f"| {sub['entropy_bits'].mean():.4f} |"
        )
    lines.append("")
    return lines


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pand_ids = set(pd.read_parquet(MANIFEST)["pand_id"].unique())
    logger.info("manifest pands: %d", len(pand_ids))
    gt_type = pd.read_parquet(GT).set_index("pand_id")["building_type"]

    certs = load_certificates(pand_ids)
    lines = [
        "# Table 2d — Intra-pand EPC label entropy audit (manifest pands)",
        "",
        "Labels normalised A+..A++++ -> A. modal share = fraction of a pand's",
        "certificates carrying its most common label. Oracle ceiling = accuracy of",
        "predicting each pand's modal label scored against a random unit's label.",
        "",
    ]
    lines += summarize(per_pand(certs), gt_type, "All certificates (any year)")
    nta = certs[certs["reg_year"] >= 2021]
    lines += summarize(per_pand(nta), gt_type,
                       "NTA 8800 era only (reg >= 2021, matches pipeline filter)")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
