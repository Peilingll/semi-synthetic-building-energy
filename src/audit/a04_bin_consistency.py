"""0.4 — Is `Energieklasse` exactly a binning of `PrimaireFossieleEnergie`?

Under NTA 8800 the residential label ladder is a fixed table on EP2
(kWh/m2.yr, primary fossil energy). If the register is internally consistent,
bin(EP2) == Energieklasse for every NTA 8800 residential certificate. Any
mismatch is either (a) a different quantity being binned (EMG forfaitair
variant), (b) a boundary we have wrong, or (c) register noise — each of which is
worth knowing before the label is treated as ground truth.

Also verifies the 7-class boundaries used in `src/stage3/regression_kwh.py`.

Outputs: reports/tables/audit/A04_bin_consistency.md
"""

import logging

import numpy as np
import pandas as pd

from src.audit import ep_raw

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUT = ep_raw.REPO_ROOT / "reports" / "tables" / "audit" / "A04_bin_consistency.md"

# Official NTA 8800 residential (woningbouw) label ladder on EP2, upper bounds
# in kWh/m2.yr; last class is unbounded.
LADDER11 = [("A++++", 0.0), ("A+++", 50.0), ("A++", 75.0), ("A+", 105.0),
            ("A", 160.0), ("B", 190.0), ("C", 250.0), ("D", 290.0),
            ("E", 335.0), ("F", 380.0), ("G", np.inf)]
# The 7-class set the repo's regression route uses (A merges A+..A++++).
LADDER7 = [("A", 160.0), ("B", 190.0), ("C", 250.0), ("D", 290.0),
           ("E", 335.0), ("F", 380.0), ("G", np.inf)]


def bin_ep2(v: pd.Series, ladder) -> pd.Series:
    edges = [b for _, b in ladder[:-1]]
    names = [n for n, _ in ladder]
    return pd.Series(np.take(names, np.digitize(v.to_numpy(), edges, right=True)),
                     index=v.index)


def main():
    ep = ep_raw.load(residential_only=True, nta_only=True)
    ep = ep[ep["energieklasse"].isin([n for n, _ in LADDER11])].copy()
    ep = ep.dropna(subset=["primaire_fossiele_energie"])
    n0 = len(ep)
    ep = ep[ep["primaire_fossiele_energie"].between(-300, 1500)]

    L = ["# A04 — `bin(PrimaireFossieleEnergie) == Energieklasse`?", "",
         f"Scope: four-city NTA 8800 residential certificates with a non-null EP2, "
         f"n={len(ep):,} (dropped {n0 - len(ep)} implausible EP2 values).", "",
         "Boundaries tested (official NTA 8800 woningbouw, upper bound in kWh/m2.yr): "
         "A++++ <= 0, A+++ <= 50, A++ <= 75, A+ <= 105, A <= 160, B <= 190, C <= 250, "
         "D <= 290, E <= 335, F <= 380, G > 380.", ""]

    ep["pred11"] = bin_ep2(ep["primaire_fossiele_energie"], LADDER11)
    ep["match11"] = ep["pred11"] == ep["energieklasse"]
    ep["label7"] = ep_raw.merge_a_classes(ep["energieklasse"])
    ep["pred7"] = bin_ep2(ep["primaire_fossiele_energie"], LADDER7)
    ep["match7"] = ep["pred7"] == ep["label7"]

    # EMG forfaitair alternative: the registered class may follow that column instead.
    has_emg = ep["pf_emg_forfaitair"].notna()
    ep["pred11_emg"] = bin_ep2(ep["pf_emg_forfaitair"].fillna(
        ep["primaire_fossiele_energie"]), LADDER11)
    ep["match11_emg"] = ep["pred11_emg"] == ep["energieklasse"]

    L += ["## Overall agreement", "",
          "| test | n | agreement |", "|---|---:|---:|",
          f"| 11-class ladder on `PrimaireFossieleEnergie` | {len(ep):,} | "
          f"{ep['match11'].mean():.4%} |",
          f"| 7-class ladder (repo `regression_kwh.PF_BINS`) | {len(ep):,} | "
          f"{ep['match7'].mean():.4%} |",
          f"| 11-class, using `...EMGForfaitair` where present | {len(ep):,} | "
          f"{ep['match11_emg'].mean():.4%} |",
          f"| — subset that has an EMG forfaitair value | {int(has_emg.sum()):,} | "
          f"plain {ep.loc[has_emg, 'match11'].mean():.4%} -> "
          f"forfaitair {ep.loc[has_emg, 'match11_emg'].mean():.4%} |", ""]

    # per-year / per-calc-type breakdown
    ep["reg_year"] = ep["reg_date"].str[:4]
    L += ["## Agreement by registration year", "",
          "| year | n | 11-class | 7-class |", "|---|---:|---:|---:|"]
    for y, s in ep.groupby("reg_year"):
        if len(s) < 200:
            continue
        L += [f"| {y} | {len(s):,} | {s['match11'].mean():.3%} | {s['match7'].mean():.3%} |"]
    L += [""]

    L += ["## Agreement by Berekeningstype", "",
          "| calc type | n | 11-class |", "|---|---:|---:|"]
    for t, s in ep.groupby("calc_type"):
        if len(s) < 200:
            continue
        L += [f"| {t} | {len(s):,} | {s['match11'].mean():.3%} |"]
    L += [""]

    # where does it break?
    bad = ep[~ep["match11"]]
    L += [f"## The {len(bad):,} mismatches ({len(bad) / len(ep):.3%})", ""]
    if len(bad):
        idx = {n: i for i, (n, _) in enumerate(LADDER11)}
        off = bad["energieklasse"].map(idx) - bad["pred11"].map(idx)
        L += ["| registered - binned (class steps) | n |", "|---:|---:|"]
        for k, v in off.value_counts().sort_index().items():
            L += [f"| {k:+d} | {v:,} |"]
        L += ["",
              "| registered | binned | n | median EP2 | median EP1 | median renewables % |",
              "|---|---|---:|---:|---:|---:|"]
        top = bad.groupby(["energieklasse", "pred11"]).size().nlargest(10)
        for (a, b), n in top.items():
            s = bad[(bad["energieklasse"] == a) & (bad["pred11"] == b)]
            L += [f"| {a} | {b} | {n:,} | {s['primaire_fossiele_energie'].median():.1f} | "
                  f"{s['energiebehoefte'].median():.1f} | "
                  f"{s['aandeel_hernieuwbare_energie'].median():.1f} |"]
        L += ["",
              f"Share of mismatches that carry an EMG forfaitair value: "
              f"{bad['pf_emg_forfaitair'].notna().mean():.1%}; share within 2 kWh/m2.yr of "
              f"a class boundary (rounding): "
              f"{_near_boundary(bad['primaire_fossiele_energie']).mean():.1%}.", ""]

    # consequence for the kWh regression route
    e = ep[has_emg].copy()
    d = e["pf_emg_forfaitair"] - e["primaire_fossiele_energie"]
    ep["pred7_emg"] = bin_ep2(ep["pf_emg_forfaitair"].fillna(
        ep["primaire_fossiele_energie"]), LADDER7)
    L += ["## Consequence for the kWh regression route", "",
          "`src/stage2/extract_kwh.py` takes `PrimaireFossieleEnergie` as the regression "
          "target `pf_kwh`, but the registered label is the binning of "
          "`PrimaireFossieleEnergieEMGForfaitair` wherever that column is filled. For "
          "those certificates the regression target and the classification label are two "
          "different quantities.", "",
          "| quantity | value |", "|---|---:|",
          f"| certificates with an EMG forfaitair value | {int(has_emg.sum()):,} "
          f"({has_emg.mean():.1%}) |",
          f"| of those, share where the two columns actually differ (>1 kWh/m2.yr) | "
          f"{(d.abs() > 1).mean():.1%} |",
          f"| median absolute difference where they differ | "
          f"{d[d.abs() > 1].abs().median():.1f} kWh/m2.yr |",
          f"| p90 absolute difference where they differ | "
          f"{d[d.abs() > 1].abs().quantile(0.9):.1f} kWh/m2.yr |",
          f"| mean absolute difference over all EMG rows | {d.abs().mean():.1f} kWh/m2.yr |",
          f"| 7-class label mismatch using plain PF | {(~ep['match7']).mean():.2%} |",
          f"| 7-class label mismatch using EMG forfaitair | "
          f"{(ep['pred7_emg'] != ep['label7']).mean():.2%} |", "",
          f"So the two definitions coincide for most certificates and diverge sharply for "
          f"a minority ({(d.abs() > 1).mean():.0%} of the EMG rows, median gap "
          f"{d[d.abs() > 1].abs().median():.0f} kWh/m2.yr). On the 7-class ladder the "
          f"resulting label error is {(~ep['match7']).mean():.2%} — small, but it is "
          f"concentrated in exactly the district-heating stock that dominates Amsterdam "
          f"and Rotterdam, and it is removable at zero cost by reading the column the "
          f"register itself bins.", ""]

    # empirical boundary recovery — do the data imply exactly these edges?
    L += ["## Empirical boundaries implied by the data", "",
          "For each class, the observed EP2 range. Under an exact binning the max of "
          "class k equals the min of class k+1 up to rounding.", "",
          "| class | n | min EP2 | max EP2 | official upper bound |",
          "|---|---:|---:|---:|---:|"]
    for name, ub in LADDER11:
        s = ep.loc[ep["energieklasse"] == name, "primaire_fossiele_energie"]
        if s.empty:
            continue
        L += [f"| {name} | {len(s):,} | {s.min():.2f} | {s.max():.2f} | "
              f"{'—' if not np.isfinite(ub) else f'{ub:.0f}'} |"]
    L += [""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    logger.info("wrote %s", OUT)


def _near_boundary(v: pd.Series, tol: float = 2.0) -> pd.Series:
    edges = np.array([b for _, b in LADDER11[:-1]])
    d = np.abs(v.to_numpy()[:, None] - edges[None, :]).min(axis=1)
    return pd.Series(d <= tol, index=v.index)


if __name__ == "__main__":
    main()
