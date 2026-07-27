"""Regenerate the TABULA NL U-value lookup from the official source workbook.

Why this exists: `data/processed/tabula_nl.csv` had no generator — it was typed in
by hand from `data/raw/tabula/tabula-values.xlsx` and every row carries the same
conversion error, `U_csv = 1 / (1/U_TABULA - 0.26)`, i.e. a uniform thermal
resistance offset of -0.26 m2K/W. Verified on 2026-07-27 against both the
workbook (`Tab.Building.Constr`, NL `*.ReEx.*` rows) and the TABULA WebTool
datasheets for `NL.N.TH.01.Gen` / `NL.N.AB.01.Gen`.

The row -> construction mapping in the old CSV was correct and is preserved:
terraced houses take the solid-wall constructions, apartment blocks / multi-family
/ single-family take the cavity-wall ones, matching TABULA's exemplary buildings.

Writes the canonical `data/processed/tabula_nl.csv` plus a provenance file naming
the exact TABULA construction behind every value. The superseded hand-made table
is kept at `data/processed/legacy/tabula_nl_handmade.csv` so pre-2026-07-27
results stay reproducible; `src/tabula/impact_check.py` diffs the two.

Validation against the TABULA WebTool datasheets (2026-07-27):
  NL.N.TH.01.Gen  wall 2.22 / roof 2.08 / floor 2.44  -> reproduced exactly
  NL.N.AB.01.Gen  wall 1.61 / roof 1.54               -> reproduced exactly
  NL.N.AB.01.Gen  floor 1.14                          -> KNOWN DEVIATION, this
    script yields 1.7241 (`NL.Floor.ReEx.01.02`, cavity walls, uninsulated).
    1.14 does not appear anywhere in the NL ReEx construction library, so the
    exemplary building must use an assembly documented only in the calculator
    workbook. 1.7241 is the traceable library value and the conservative one;
    walls dominate H_tr, so the difference does not move any downstream number.

Run: .venv/Scripts/python.exe -m src.tabula.build_lookup
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
XLSX = REPO_ROOT / "data" / "raw" / "tabula" / "tabula-values.xlsx"
OLD = REPO_ROOT / "data" / "processed" / "legacy" / "tabula_nl_handmade.csv"
OUT = REPO_ROOT / "data" / "processed" / "tabula_nl.csv"

SHEET = "Tab.Building.Constr"
R_OFFSET_BUG = 0.26  # m2K/W, the offset found in the old hand-made CSV

PERIODS = [("NL.01", 0, 1964), ("NL.02", 1965, 1974), ("NL.03", 1975, 1991),
           ("NL.04", 1992, 2005), ("NL.05", 2006, 2014), ("NL.06", 2015, 9999)]

# Which construction variant each TABULA size class uses, from the exemplary
# buildings (NL.N.TH.01.Gen -> solid walls; NL.N.AB.01.Gen -> cavity walls).
WALL_KIND = {"TH": "solid", "SFH": "cavity", "AB": "cavity", "MFH": "cavity"}
# Roof shape of the exemplary building: pitched for houses, flat for blocks.
ROOF_KIND = {"TH": "pitched", "SFH": "pitched", "AB": "flat", "MFH": "flat"}

# Window / door constructions are given directly as U (no R convention), so the
# old CSV's window column was already correct. Kept explicit for traceability.
WINDOW_U = {"NL.01": 5.2, "NL.02": 5.2, "NL.03": 5.2,
            "NL.04": 2.9, "NL.05": 1.8, "NL.06": 1.8}


def load_nl_constructions() -> pd.DataFrame:
    d = pd.read_excel(XLSX, sheet_name=SHEET)
    nl = d[(d["Code_Country"].astype(str).str.upper() == "NL")
           & d["Code_Construction"].astype(str).str.contains(r"\.ReEx\.", regex=True)]
    keep = ["Code_Construction", "Code_ElementType",
            "Code_Construction_ConstructionYearClass", "Type_Construction", "U"]
    nl = nl[keep].rename(columns={
        "Code_Construction_ConstructionYearClass": "period",
        "Code_ElementType": "element", "Type_Construction": "desc"})
    nl["desc"] = nl["desc"].fillna("").astype(str)
    return nl.reset_index(drop=True)


def pick(nl: pd.DataFrame, element: str, period: str, wants: list[str],
         avoid: str = "afterwards insulated") -> tuple[float, str]:
    """Pick the existing-state (un-refurbished) construction for one cell.

    `wants` are substrings applied as successive filters, most important first
    (e.g. ['cavity walls', 'flat'] for an apartment-block roof — NL.01 roofs and
    floors come in solid-wall and cavity-wall flavours, so the wall kind has to
    be matched on those elements too, not only on the wall itself). A filter that
    would empty the candidate set is skipped. Refurbished variants are excluded.
    When several sub-period variants exist (NL.03 splits 1975-82 / 1983-87 /
    1988-91) the worst U is taken, i.e. the earliest sub-period, matching how a
    single period-level archetype must behave conservatively.
    """
    c = nl[(nl["element"] == element) & (nl["period"] == period)].copy()
    if c.empty:
        raise KeyError(f"no {element} construction for {period}")
    not_refurb = ~c["desc"].str.contains(avoid, case=False, na=False)
    if not_refurb.any():
        c = c[not_refurb]
    for want in wants:
        w = c[c["desc"].str.contains(want, case=False, na=False)]
        if not w.empty:
            c = w
    r = c.loc[c["U"].idxmax()]
    return float(r["U"]), f"{r['Code_Construction']} ({r['desc'] or 'n/a'})"


def main():
    nl = load_nl_constructions()
    rows, prov = [], []
    for bt in ["AB", "MFH", "SFH", "TH"]:
        for period, y1, y2 in PERIODS:
            wall_want = f"{WALL_KIND[bt]} walls"
            u_wall, s_wall = pick(nl, "Wall", period, [wall_want])
            u_roof, s_roof = pick(nl, "Roof", period, [wall_want, ROOF_KIND[bt]])
            u_floor, s_floor = pick(nl, "Floor", period, [wall_want])
            rows.append(dict(building_type=bt, period=period, year_start=y1, year_end=y2,
                             u_wall=round(u_wall, 4), u_roof=round(u_roof, 4),
                             u_floor=round(u_floor, 4), u_window=WINDOW_U[period]))
            prov.append(dict(building_type=bt, period=period, src_wall=s_wall,
                             src_roof=s_roof, src_floor=s_floor))
    new = pd.DataFrame(rows)
    new.to_csv(OUT, index=False)
    pd.DataFrame(prov).to_csv(OUT.with_name("tabula_nl_provenance.csv"), index=False)
    logger.info("wrote %s (%d rows) + provenance", OUT, len(new))

    old = pd.read_csv(OLD)
    m = old.merge(new, on=["building_type", "period"], suffixes=("_old", "_new"))
    logger.info("\n%-6s %-7s %9s %9s %6s   %9s %9s   %9s %9s",
                "type", "period", "wall_old", "wall_new", "ratio", "roof_old",
                "roof_new", "floor_old", "floor_new")
    for _, r in m.iterrows():
        logger.info("%-6s %-7s %9.4f %9.4f %6.2f   %9.4f %9.4f   %9.4f %9.4f",
                    r["building_type"], r["period"], r["u_wall_old"], r["u_wall_new"],
                    r["u_wall_old"] / r["u_wall_new"], r["u_roof_old"], r["u_roof_new"],
                    r["u_floor_old"], r["u_floor_new"])
    # confirm the -0.26 R offset explains the old file
    dr = (1 / m["u_wall_new"] - 1 / m["u_wall_old"])
    logger.info("\nR offset old->new, wall: median %.4f  min %.4f  max %.4f "
                "(expected %.2f)", dr.median(), dr.min(), dr.max(), R_OFFSET_BUG)


if __name__ == "__main__":
    main()
