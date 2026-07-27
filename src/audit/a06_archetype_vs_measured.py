"""0.6 — Does the archetype chain track *measured* demand? (design "B")

Distinction being tested:
  A  = |H_tr(SVI cell) - H_tr(registry cell)|   -> internal consistency, no reality
  B  = |demand(cell) - measured Energiebehoefte| -> external validity

B is NOT "train a model to predict EP1". It is a FORWARD physical calculation:
cell -> TABULA U-values -> transmission loss over 3DBAG areas -> demand, then
compared against the register's own EP1. Nothing is fitted except a single
global 2-parameter scaling (shared by all branches) so that the result is not
driven by assumed HDD / air-change / gains values.

Target is EP1 (`Energiebehoefte`, net demand), NOT EP2 (`PrimaireFossieleEnergie`)
and NOT the label: EP2 adds the installation side, which no envelope model can
reach (see A02).

Decomposition reported:
  B_gt   registry cell -> demand vs measured   (the archetype's own error)
  B_pred SVI cell      -> demand vs measured   (whole chain)
  B_pred - B_gt                                 (damage attributable to Stage 1)
  A                                             (same predictions, internal metric)

Output: reports/tables/audit/A06_archetype_vs_measured.md
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.audit import ep_raw

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO = ep_raw.REPO_ROOT
PROCESSED = REPO / "data" / "processed"
OUT = REPO / "reports" / "tables" / "audit" / "A06_archetype_vs_measured.md"
CITIES = ["amsterdam", "rotterdam", "utrecht", "delft"]

# Free parameters of the *uncalibrated* physical variant (magnitude only).
HDD = 2900.0        # Kd, base 18 C, NL
ACH = 0.5           # h^-1 air change rate
GAINS = 35.0        # kWh/m2.yr internal + solar gains
WWR = 0.25          # window-to-wall ratio

MODELS = {
    "DINOv2 frozen": "reports/stage1/dinov2_frozen/holdout_preds.parquet",
    "ResNet-50 ft": "reports/stage1/resnet50_ft/holdout_preds.parquet",
    "InternVL3 ZS": "reports/stage1/vlm_internvl3/v3_holdout_per_pand_id.parquet",
}
PERIODS = [(0, 1964, "NL.01"), (1965, 1974, "NL.02"), (1975, 1991, "NL.03"),
           (1992, 2005, "NL.04"), (2006, 2014, "NL.05"), (2015, 9999, "NL.06")]


def classify_period(years) -> np.ndarray:
    y = np.asarray(years, dtype=float)
    out = np.full(len(y), None, dtype=object)
    for lo, hi, lab in PERIODS:
        out[(y >= lo) & (y <= hi)] = lab
    return out


def load_geometry() -> pd.DataFrame:
    g = []
    for c in CITIES:
        g.append(pd.read_parquet(PROCESSED / c / "bag_3dbag_ep_joined.parquet",
                                 columns=["pand_id", "b3_opp_buitenmuur", "b3_opp_dak_plat",
                                          "b3_opp_dak_schuin", "b3_opp_grond",
                                          "b3_volume_lod22"]))
    g = pd.concat(g).drop_duplicates("pand_id")
    f = []
    for c in CITIES:
        f.append(pd.read_parquet(PROCESSED / c / "residential_with_3d_features.parquet",
                                 columns=["pand_id", "floor_area_estimated"]))
    f = pd.concat(f).drop_duplicates("pand_id")
    for d in (g, f):
        d["pand_id"] = d["pand_id"].astype(str).str.zfill(16)
    g = g.merge(f, on="pand_id", how="inner")
    ok = (g["floor_area_estimated"].between(20, 1e5)
          & g["b3_opp_buitenmuur"].between(1, 5e4)
          & g["b3_opp_grond"].between(5, 2e4)
          & g["b3_volume_lod22"].between(50, 1e7))
    return g[ok].reset_index(drop=True)


U = None


def h_tr(cells, g) -> np.ndarray:
    """Specific transmission loss coefficient, W/(K.m2 floor)."""
    u = U.reindex(pd.Index(cells))
    a_wall = g["b3_opp_buitenmuur"].to_numpy()
    a_roof = (g["b3_opp_dak_plat"] + g["b3_opp_dak_schuin"]).to_numpy()
    a_gnd = g["b3_opp_grond"].to_numpy()
    q = (u["u_wall"].to_numpy() * a_wall * (1 - WWR)
         + u["u_window"].to_numpy() * a_wall * WWR
         + u["u_roof"].to_numpy() * a_roof
         + u["u_floor"].to_numpy() * a_gnd)
    return q / g["floor_area_estimated"].to_numpy()


def demand_physical(cells, g) -> np.ndarray:
    """Uncalibrated seasonal balance, kWh/(m2.yr). Assumption-laden."""
    h_ve = 0.34 * ACH * 0.8 * g["b3_volume_lod22"].to_numpy() / g["floor_area_estimated"].to_numpy()
    return (h_tr(cells, g) + h_ve) * HDD * 24 / 1000 - GAINS


def mae(a, b) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.abs(a[m] - b[m]).mean())


def main():
    global U
    tab = pd.read_csv(PROCESSED / "tabula_nl.csv")
    tab["key"] = tab["building_type"] + "|" + tab["period"]
    U = tab.set_index("key")[["u_wall", "u_roof", "u_floor", "u_window"]]

    geo = load_geometry()
    gt = pd.read_parquet(PROCESSED / "stage1_gt.parquet")
    gt["pand_id"] = gt["pand_id"].astype(str).str.zfill(16)
    gt["cell_gt"] = gt["building_type"].astype(str) + "|" + gt["tabula_period"].astype(str)

    # measured EP1 per pand: median over that pand's NTA 8800 residential certificates
    ep = ep_raw.load(residential_only=True, nta_only=True)
    ep = ep[ep["energiebehoefte"].between(20, 800)]
    m1 = (ep.groupby("pand_id")["energiebehoefte"].agg(["median", "size", "std"])
          .rename(columns={"median": "ep1", "size": "n_cert", "std": "ep1_std"}))

    d = (gt[["pand_id", "city", "building_type", "tabula_period", "cell_gt"]]
         .merge(geo, on="pand_id", how="inner").merge(m1, on="pand_id", how="inner"))
    d["h_gt"] = h_tr(d["cell_gt"].to_numpy(), d)
    d["q_gt"] = demand_physical(d["cell_gt"].to_numpy(), d)
    d = d[np.isfinite(d["h_gt"])].reset_index(drop=True)

    L = ["# A06 — Does the archetype chain track measured demand? (design B)", "",
         "`cell -> TABULA U -> transmission loss over 3DBAG areas -> demand`, compared "
         "against the register's own **EP1 `Energiebehoefte`** (net demand). Not EP2, not "
         "the label — EP2 adds the installation side that no envelope model reaches (A02).", "",
         f"Four-city pands with GT cell + valid 3DBAG geometry + measured EP1: n={len(d):,}.", "",
         "## B-1 — Does `H_tr` from the registry cell correlate with measured EP1 at all?", "",
         "| subset | n | Pearson r | R2 | Spearman rho |", "|---|---:|---:|---:|---:|"]

    def corr_row(name, sub):
        if len(sub) < 100:
            return
        a, b = sub["h_gt"].to_numpy(), sub["ep1"].to_numpy()
        r = pearsonr(a, b)[0]
        L.append(f"| {name} | {len(sub):,} | {r:.3f} | {r ** 2:.3f} | "
                 f"{spearmanr(a, b)[0]:.3f} |")

    corr_row("all four cities", d)
    for t, s in d.groupby("building_type"):
        corr_row(f"type {t}", s)
    for c, s in d.groupby("city"):
        corr_row(f"city {c}", s)
    L += [""]

    # per-cell: implied demand vs measured — exposes bad TABULA rows
    L += ["## B-2 — Implied vs measured demand per cell "
          f"(uncalibrated: HDD {HDD:.0f}, ACH {ACH}, gains {GAINS:.0f}, WWR {WWR})", "",
          "| cell | n | median H_tr | implied kWh/m2.yr | measured EP1 median | ratio |",
          "|---|---:|---:|---:|---:|---:|"]
    cs = d.groupby("cell_gt").agg(n=("ep1", "size"), h=("h_gt", "median"),
                                  q=("q_gt", "median"), ep1=("ep1", "median"))
    for c, r in cs.sort_values("n", ascending=False).head(10).iterrows():
        L += [f"| {c} | {r['n']:,.0f} | {r['h']:.2f} | {r['q']:.0f} | {r['ep1']:.0f} | "
              f"{r['q'] / r['ep1']:.2f} |"]
    L += ["", "A ratio far from 1 with everything else near 1 indicates a bad TABULA row "
          "rather than a modelling error.", ""]

    # ---- calibrated map: one global 2-parameter fit, shared by all branches ----
    dev = pd.read_parquet(PROCESSED / "dev_fold_indices.parquet")
    dev["pand_id"] = dev["pand_id"].astype(str).str.zfill(16)
    fit_set = d[d["pand_id"].isin(set(dev["pand_id"]))]
    A_ = np.c_[np.ones(len(fit_set)), fit_set["h_gt"]]
    beta, *_ = np.linalg.lstsq(A_, fit_set["ep1"].to_numpy(), rcond=None)
    L += ["## B-3 — Calibrated comparison on the hold-out", "",
          f"To avoid the assumed HDD/ACH/gains driving the result, EP1 is regressed on "
          f"`H_tr` **once** on the dev pool with the registry cell "
          f"(EP1 = {beta[0]:.1f} + {beta[1]:.2f}·H_tr, n={len(fit_set):,}); the same two "
          f"coefficients are then applied to every branch. Nothing else is fitted.", "",
          "| model | joint cell acc | A: MAE(H_tr) | B_gt MAE | B_pred MAE | B_pred − B_gt |",
          "|---|---:|---:|---:|---:|---:|"]

    for name, path in MODELS.items():
        p = pd.read_parquet(REPO / path)
        p["pand_id"] = p["pand_id"].astype(str).str.zfill(16)
        s = p.merge(d, on="pand_id", how="inner")
        cell_pred = pd.Series(p["pred_type"].to_numpy()).astype(str).to_numpy()
        s = s.assign(cell_pred=(s["pred_type"].astype(str) + "|"
                                + classify_period(s["pred_year"])))
        s = s[s["cell_pred"].isin(U.index)]
        hp = h_tr(s["cell_pred"].to_numpy(), s)
        hg = s["h_gt"].to_numpy()
        ep1 = s["ep1"].to_numpy()
        q_gt = beta[0] + beta[1] * hg
        q_pr = beta[0] + beta[1] * hp
        joint = (s["cell_pred"] == s["cell_gt"]).mean()
        L += [f"| {name} | {joint:.3f} | {mae(hp, hg):.3f} | {mae(q_gt, ep1):.1f} | "
              f"{mae(q_pr, ep1):.1f} | {mae(q_pr, ep1) - mae(q_gt, ep1):+.1f} |"]
    L += ["", "Units: A in W/(K·m²), B in kWh/(m²·yr).", ""]

    # reference: how large is B_gt relative to the natural scale of EP1?
    L += ["## B-4 — Is B_gt large?", "",
          "| reference quantity | value |", "|---|---:|",
          f"| measured EP1 std (four cities) | {d['ep1'].std():.1f} kWh/m2.yr |",
          f"| measured EP1 mean | {d['ep1'].mean():.1f} |",
          f"| MAE of predicting the global mean EP1 | "
          f"{mae(np.full(len(d), d['ep1'].mean()), d['ep1'].to_numpy()):.1f} |",
          f"| MAE of the calibrated registry-cell model (dev) | "
          f"{mae(beta[0] + beta[1] * fit_set['h_gt'].to_numpy(), fit_set['ep1'].to_numpy()):.1f} |",
          f"| median within-pand EP1 std (multi-cert pands) | "
          f"{d.loc[d['n_cert'] >= 2, 'ep1_std'].median():.1f} |", "",
          "If the calibrated registry-cell model does not beat 'predict the mean', the "
          "archetype chain carries no usable information about measured demand, and any "
          "Stage-1 improvement is invisible in B by construction.", ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
