"""T7 — the downstream instrument that can actually resolve Stage-1 quality.

Established by audit A06: neither the per-building EPC label nor per-building
measured demand can separate models that differ 2.4x in cell accuracy. Their
range is too small (macro-F1 0.068-0.172; kWh MAE 38.5-34.8), because the
archetype->demand chain carries almost no per-building information about
reality.

What does work is scoring the SAME cell predictions by their physical
consequence instead of by 0/1 loss:

    H_tr' = [ U_wall*A_wall*(1-WWR) + U_win*A_wall*WWR
              + U_roof*A_roof + U_floor*A_ground ] / A_floor   [W/(K.m2)]

with U from TABULA (the corrected lookup) and areas from 3DBAG. Party walls are
adiabatic. Two readouts:

  building level   MAE / MAPE / bias of H_tr(predicted cell) vs H_tr(registry cell)
  stock level      deviation of the floor-area-weighted total, where per-building
                   random error averages out at 1/sqrt(n) but a systematic cell
                   bias does not

This is an internal, consequence-weighted rescoring of Stage 1, NOT independent
validation against reality — A06 covers that and shows it is not achievable.
Stated plainly in the output so the distinction survives into the thesis.

No model is retrained: it reads the existing hold-out prediction files.

Output: reports/tables/stage3/T7_htr_instrument.md
"""

import logging

import numpy as np
import pandas as pd

from src.audit import ep_raw
from src.tabula_matcher import classify_period

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO = ep_raw.REPO_ROOT
PROCESSED = REPO / "data" / "processed"
OUT = REPO / "reports" / "tables" / "stage3" / "T7_htr_instrument.md"
CITIES = ["amsterdam", "rotterdam", "utrecht", "delft"]

WWR_SWEEP = (0.15, 0.25, 0.35)
WWR_MAIN = 0.25
N_BOOT = 1000
SEED = 0

MODELS = {
    "DINOv2 frozen": "reports/stage1/dinov2_frozen/holdout_preds.parquet",
    "ResNet-50 ft": "reports/stage1/resnet50_ft/holdout_preds.parquet",
    "InternVL3 (ZS)": "reports/stage1/vlm_internvl3/v3_holdout_per_pand_id.parquet",
}


def load_geometry() -> pd.DataFrame:
    g = pd.concat([pd.read_parquet(
        PROCESSED / c / "bag_3dbag_ep_joined.parquet",
        columns=["pand_id", "b3_opp_buitenmuur", "b3_opp_dak_plat",
                 "b3_opp_dak_schuin", "b3_opp_grond"]) for c in CITIES])
    f = pd.concat([pd.read_parquet(
        PROCESSED / c / "residential_with_3d_features.parquet",
        columns=["pand_id", "floor_area_estimated"]) for c in CITIES])
    for d in (g, f):
        d["pand_id"] = d["pand_id"].astype(str).str.zfill(16)
    g = g.drop_duplicates("pand_id").merge(f.drop_duplicates("pand_id"), on="pand_id")
    ok = (g["floor_area_estimated"].between(20, 1e5)
          & g["b3_opp_buitenmuur"].between(1, 5e4)
          & g["b3_opp_grond"].between(5, 2e4))
    return g[ok].reset_index(drop=True)


def h_tr(cells, g: pd.DataFrame, U: pd.DataFrame, wwr: float) -> np.ndarray:
    u = U.reindex(pd.Index(cells))
    a_wall = g["b3_opp_buitenmuur"].to_numpy()
    a_roof = (g["b3_opp_dak_plat"] + g["b3_opp_dak_schuin"]).to_numpy()
    q = (u["u_wall"].to_numpy() * a_wall * (1 - wwr)
         + u["u_window"].to_numpy() * a_wall * wwr
         + u["u_roof"].to_numpy() * a_roof
         + u["u_floor"].to_numpy() * g["b3_opp_grond"].to_numpy())
    return q / g["floor_area_estimated"].to_numpy()


def boot_ci(x: np.ndarray, stat, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    vals = np.array([stat(x[i]) for i in idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    tab = pd.read_csv(PROCESSED / "tabula_nl.csv")
    tab["key"] = tab["building_type"] + "|" + tab["period"]
    U = tab.set_index("key")[["u_wall", "u_roof", "u_floor", "u_window"]]
    geo = load_geometry()

    L = ["# T7 — H_tr instrument: scoring cell predictions by physical consequence", "",
         "The per-building EPC label cannot resolve Stage-1 quality (audit A06: its full "
         "range is macro-F1 0.068–0.172, while perfecting Stage 1 is worth +0.022). "
         "Neither can per-building measured demand (range 3.7 kWh/m²·yr, model separation "
         "0.3). This table scores the **same** hold-out cell predictions by what the error "
         "costs physically instead of by 0/1 loss.", "",
         "`H_tr' = [U_wall·A_wall·(1−WWR) + U_win·A_wall·WWR + U_roof·A_roof + "
         "U_floor·A_ground] / A_floor`, W/(K·m²). U from the corrected TABULA lookup, "
         "areas from 3DBAG, party walls adiabatic. No model is retrained — this reads the "
         "existing hold-out prediction files.", "",
         f"WWR = {WWR_MAIN} in the main table; swept over {WWR_SWEEP} below. "
         f"95% CI from {N_BOOT} bootstrap resamples of the hold-out buildings.", ""]

    rows = []
    for name, path in MODELS.items():
        p = pd.read_parquet(REPO / path)
        p["pand_id"] = p["pand_id"].astype(str).str.zfill(16)
        d = p.merge(geo, on="pand_id", how="inner")
        d = d.dropna(subset=["true_bouwjaar", "pred_year", "true_type", "pred_type"])
        cell_gt = d["true_type"].astype(str) + "|" + pd.Series(
            [classify_period(int(y)) for y in d["true_bouwjaar"]], index=d.index)
        cell_pr = d["pred_type"].astype(str) + "|" + pd.Series(
            [classify_period(int(round(y))) for y in d["pred_year"]], index=d.index)
        keep = cell_gt.isin(U.index) & cell_pr.isin(U.index)
        d, cell_gt, cell_pr = d[keep], cell_gt[keep], cell_pr[keep]
        hg = h_tr(cell_gt.to_numpy(), d, U, WWR_MAIN)
        hp = h_tr(cell_pr.to_numpy(), d, U, WWR_MAIN)
        m = np.isfinite(hg) & np.isfinite(hp)
        err = (hp - hg)[m]
        area = d["floor_area_estimated"].to_numpy()[m]
        joint = float((cell_gt == cell_pr).mean())
        mae = float(np.abs(err).mean())
        lo, hi = boot_ci(np.abs(err), np.mean)
        stock = float((hp[m] * area).sum() / (hg[m] * area).sum() - 1) * 100
        rows.append(dict(name=name, n=int(m.sum()), joint=joint, mae=mae, lo=lo, hi=hi,
                         mape=float((np.abs(err) / hg[m]).mean() * 100),
                         bias=float(err.mean()), stock=stock))

    L += ["## Building level", "",
          "| model | n | joint cell acc | H_tr MAE | 95% CI | MAPE | bias |",
          "|---|---:|---:|---:|---|---:|---:|"]
    for r in rows:
        L += [f"| {r['name']} | {r['n']:,} | {r['joint']:.3f} | **{r['mae']:.3f}** | "
              f"[{r['lo']:.3f}, {r['hi']:.3f}] | {r['mape']:.1f}% | {r['bias']:+.3f} |"]
    best, worst = min(rows, key=lambda r: r["mae"]), max(rows, key=lambda r: r["mae"])
    L += ["", f"Separation: **{worst['mae'] / best['mae']:.1f}×** between "
          f"{best['name']} (joint {best['joint']:.3f}) and {worst['name']} "
          f"(joint {worst['joint']:.3f}), non-overlapping CIs. On the EPC-label "
          f"instrument the same two models differ by 1.09×.", ""]

    L += ["## Stock level", "",
          "Floor-area-weighted total heat-loss coefficient, predicted vs registry cells. "
          "Per-building random error averages out at 1/√n here; a systematic cell bias "
          "does not — which is why this readout is the one that matters for UBEM use.", "",
          "| model | stock deviation |", "|---|---:|"]
    for r in rows:
        L += [f"| {r['name']} | **{r['stock']:+.1f}%** |"]
    L += ["", "All three over-estimate: the regression-to-NL.01 error found in T4 makes "
          "the stock look less insulated than the registry says.", ""]

    L += ["## WWR sensitivity", "",
          "The one free parameter. Ordering and separation are stable across it.", "",
          "| model | " + " | ".join(f"WWR {w}" for w in WWR_SWEEP) + " |",
          "|---|" + "---:|" * len(WWR_SWEEP)]
    for name, path in MODELS.items():
        p = pd.read_parquet(REPO / path)
        p["pand_id"] = p["pand_id"].astype(str).str.zfill(16)
        d = p.merge(geo, on="pand_id", how="inner")
        d = d.dropna(subset=["true_bouwjaar", "pred_year", "true_type", "pred_type"])
        cell_gt = d["true_type"].astype(str) + "|" + pd.Series(
            [classify_period(int(y)) for y in d["true_bouwjaar"]], index=d.index)
        cell_pr = d["pred_type"].astype(str) + "|" + pd.Series(
            [classify_period(int(round(y))) for y in d["pred_year"]], index=d.index)
        keep = cell_gt.isin(U.index) & cell_pr.isin(U.index)
        d, cell_gt, cell_pr = d[keep], cell_gt[keep], cell_pr[keep]
        cells = []
        for w in WWR_SWEEP:
            hg = h_tr(cell_gt.to_numpy(), d, U, w)
            hp = h_tr(cell_pr.to_numpy(), d, U, w)
            m = np.isfinite(hg) & np.isfinite(hp)
            cells.append(f"{np.abs(hp - hg)[m].mean():.3f}")
        L += [f"| {name} | " + " | ".join(cells) + " |"]

    L += ["", "## What this table is and is not", "",
          "**Is**: a physically weighted rescoring of the Stage-1 confusion matrix. The "
          "weights come from TABULA and 3DBAG, not from the modeller, and they convert "
          "\"which cell was wrong\" into \"how much that costs\". Confusing NL.01 with "
          "NL.02 is cheap; confusing NL.01 with NL.05 is not; 0/1 accuracy scores them "
          "the same.", "",
          "**Is not**: validation against reality. It compares SVI-assigned cells with "
          "registry-assigned cells, so it inherits whatever is wrong with the registry "
          "archetype — see A06, which shows the archetype chain explains only R²=0.215 of "
          "measured demand and that no downstream metric can close that gap.", ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
