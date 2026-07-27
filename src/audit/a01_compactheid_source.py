"""0.1 — Where does `compactheid` come from, and is it leakage?

Claim under test (2026.06.26 log): "compactheid = BAG surface/volume ratio".
Counter-hypothesis: it is the EP-Online certificate field `Compactheid`, i.e. an
input to the NTA 8800 calculation that produces the label -> target leakage and
unavailable at inference time for uncertified buildings.

Tests
  T1  provenance in code (asserted here, verified by reading the loaders)
  T2  is EP compactheid a *building* property? within-pand spread across the
      unit-level certificates of the same pand
  T3  agreement with the 3DBAG geometric surface/volume ratio (`shape_factor`
      from src/lod2_features.py): Pearson / Spearman / identity R2, overall,
      per Gebouwtype and per unit count
  T4  same for floor_area (GebruiksoppervlakteThermischeZone) vs BAG areas
  T5  how tightly compactheid predicts the label's own inputs (EP1/EP2) inside
      one archetype cell -- the size of the leak

Output: reports/tables/audit/A01_compactheid_source.md
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.audit import ep_raw

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = ep_raw.REPO_ROOT
PROCESSED = REPO_ROOT / "data" / "processed"
OUT = REPO_ROOT / "reports" / "tables" / "audit" / "A01_compactheid_source.md"
CITIES = ["amsterdam", "rotterdam", "utrecht", "delft"]


def load_bag_geometry() -> pd.DataFrame:
    """Per-pand 3DBAG geometry for the four cities (shape_factor = env/vol)."""
    frames = []
    for c in CITIES:
        f = pd.read_parquet(PROCESSED / c / "residential_with_3d_features.parquet")
        f["city"] = c
        frames.append(f)
    g = pd.concat(frames, ignore_index=True)
    g["pand_id"] = g["pand_id"].astype(str).str.zfill(16)
    g = g.drop_duplicates("pand_id")
    # BAG unit count / VBO areas come from the join table.
    frames = []
    for c in CITIES:
        j = pd.read_parquet(PROCESSED / c / "bag_3dbag_ep_joined.parquet",
                            columns=["pand_id", "aantal_verblijfsobjecten",
                                     "oppervlakte_min", "oppervlakte_max",
                                     "b3_opp_grond", "b3_opp_scheidingsmuur",
                                     "b3_opp_buitenmuur", "b3_volume_lod22"])
        frames.append(j)
    j = pd.concat(frames, ignore_index=True)
    j["pand_id"] = j["pand_id"].astype(str).str.zfill(16)
    j = j.drop_duplicates("pand_id")
    g = g.merge(j, on="pand_id", how="left")
    # 3DBAG/BAG plausibility: volume reaches 4.3e21 m3 and envelope_area goes
    # negative in the LOD2 output; BAG oppervlakte has 5-digit sentinels.
    g.attrs["n_bad_volume"] = int((~g["volume"].between(10, 1e7)).sum())
    g.attrs["n_bad_envelope"] = int((g["envelope_area"] <= 0).sum())
    g.loc[~g["volume"].between(10, 1e7), "volume"] = np.nan
    g.loc[g["envelope_area"] <= 0, "envelope_area"] = np.nan
    g.loc[~g["shape_factor"].between(0.05, 5), "shape_factor"] = np.nan
    g.loc[~g["floor_area_estimated"].between(10, 1e5), "floor_area_estimated"] = np.nan
    for c in ["oppervlakte_min", "oppervlakte_max", "b3_opp_grond"]:
        g.loc[~g[c].between(5, 2e4), c] = np.nan
    # party-wall-inclusive variant, for completeness
    tot = (g["envelope_area"] + g["b3_opp_scheidingsmuur"].fillna(0))
    g["shape_factor_incl_party"] = np.where(g["volume"] > 0, tot / g["volume"], np.nan)
    g["shared_ratio"] = g["b3_opp_scheidingsmuur"] / (
        g["b3_opp_scheidingsmuur"] + g["b3_opp_buitenmuur"])
    return g


def agree(x: pd.Series, y: pd.Series) -> dict:
    m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 30:
        return dict(n=len(x), pearson=np.nan, spearman=np.nan, r2_identity=np.nan,
                    med_x=np.nan, med_y=np.nan, ratio=np.nan)
    ss_res = float(((x - y) ** 2).sum())
    ss_tot = float(((x - x.mean()) ** 2).sum())
    return dict(n=int(len(x)),
                pearson=float(pearsonr(x, y)[0]),
                spearman=float(spearmanr(x, y)[0]),
                r2_identity=float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
                med_x=float(x.median()), med_y=float(y.median()),
                ratio=float((x / y).median()))


def main():
    L: list[str] = ["# A01 — `compactheid` provenance and leakage audit", ""]

    ep = ep_raw.load(residential_only=True)
    ep["label"] = ep_raw.merge_a_classes(ep["energieklasse"])
    ep_nta = ep[ep["calc_type"].str.startswith("NTA 8800", na=False)].copy()
    bag = load_bag_geometry()

    # ---------------- T1 provenance ----------------
    L += ["## T1 — Provenance in code", "",
          "| consumer | column read | source file |",
          "|---|---|---|",
          "| `src/stage2/svi_compactheid.py:46` | `Compactheid` | raw EP-Online CSV |",
          "| `src/stage2/m1_plus_fullstock.py:178` | `Compactheid` | raw EP-Online CSV |",
          "| `src/stage2/m1_plus_fullstock.py:179` | `GebruiksoppervlakteThermischeZone` | raw EP-Online CSV |",
          "",
          "No code path derives `compactheid` from 3DBAG. The 3DBAG geometric ratio "
          "exists separately as `shape_factor` (`src/lod2_features.py:49`, "
          "`envelope_area / volume_lod22`) and is **not** what the ablation used.", ""]

    # ---------------- T2 unit-level? ----------------
    ep_nta.loc[~ep_nta["compactheid"].between(0.1, 10), "compactheid"] = np.nan
    g = ep_nta.groupby("pand_id")["compactheid"]
    per_pand = pd.DataFrame({"n_cert": g.size(), "std": g.std(),
                             "min": g.min(), "max": g.max(), "median": g.median()})
    multi = per_pand[per_pand["n_cert"] >= 2]
    exact = (multi["std"].fillna(0) < 1e-9).mean()
    L += ["## T2 — Is EP `Compactheid` a building-level quantity?", "",
          f"Four-city NTA 8800 residential certificates: n={len(ep_nta):,} over "
          f"{ep_nta['pand_id'].nunique():,} pands; {len(multi):,} pands carry >= 2 certificates.", "",
          "| quantity | value |", "|---|---:|",
          f"| pands with >= 2 certs where all `Compactheid` identical | {exact:.1%} |",
          f"| median within-pand std (multi-cert pands) | {multi['std'].median():.3f} |",
          f"| median within-pand range (max-min) | {(multi['max'] - multi['min']).median():.3f} |",
          f"| 90th pct within-pand range | {(multi['max'] - multi['min']).quantile(0.9):.3f} |",
          f"| overall value range (p5-p95) | {ep_nta['compactheid'].quantile(0.05):.2f}"
          f"-{ep_nta['compactheid'].quantile(0.95):.2f} |", ""]

    # ---------------- T3 agreement with 3DBAG ----------------
    latest = (ep_nta.sort_values("reg_date").drop_duplicates("pand_id", keep="last")
              [["pand_id", "city", "gebouwtype", "compactheid", "opp_thermische_zone",
                "label", "energiebehoefte", "primaire_fossiele_energie", "bouwjaar"]]
              .rename(columns={"compactheid": "ep_compactheid",
                               "opp_thermische_zone": "ep_floor_area"}))
    med = g.median().rename("ep_compactheid_med")
    m = latest.merge(med, on="pand_id").merge(
        bag[["pand_id", "shape_factor", "shape_factor_incl_party", "volume",
             "envelope_area", "aantal_verblijfsobjecten", "oppervlakte_min",
             "oppervlakte_max", "floor_area_estimated", "b3_opp_grond",
             "shared_ratio", "num_floors_estimated"]], on="pand_id", how="inner")
    # Apply the same clip the model code uses, and plausibility bounds on the rest
    # (the register holds entries like Compactheid=1.4e5, EP2=3.7e8 that would flip
    # any Pearson r single-handedly).
    m.loc[~m["ep_compactheid"].between(0.3, 5), "ep_compactheid"] = np.nan
    m.loc[~m["ep_compactheid_med"].between(0.3, 5), "ep_compactheid_med"] = np.nan
    m.loc[~m["ep_floor_area"].between(10, 5000), "ep_floor_area"] = np.nan
    for c in ["energiebehoefte", "primaire_fossiele_energie"]:
        m.loc[~m[c].between(-300, 1500), c] = np.nan

    rows = [("EP latest vs 3DBAG shape_factor", agree(m["ep_compactheid"], m["shape_factor"])),
            ("EP median vs 3DBAG shape_factor", agree(m["ep_compactheid_med"], m["shape_factor"])),
            ("EP latest vs shape_factor incl. party walls",
             agree(m["ep_compactheid"], m["shape_factor_incl_party"]))]
    L += ["## T3 — Agreement with the 3DBAG geometric surface/volume ratio", "",
          f"Side finding: `residential_with_3d_features.parquet` itself carries "
          f"{bag.attrs['n_bad_volume']} pands with implausible `volume` (max 4.3e21 m3) and "
          f"{bag.attrs['n_bad_envelope']} with `envelope_area` <= 0 across the four cities; "
          f"they are excluded here.", "",
          "| comparison | n | Pearson r | Spearman | R2 (identity) | median EP | median BAG | median ratio |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, a in rows:
        L += [f"| {name} | {a['n']:,} | {a['pearson']:.3f} | {a['spearman']:.3f} | "
              f"{a['r2_identity']:.3f} | {a['med_x']:.2f} | {a['med_y']:.2f} | {a['ratio']:.2f} |"]
    L += [""]

    L += ["### By Gebouwtype (EP latest vs shape_factor)", "",
          "| Gebouwtype | n | Pearson r | Spearman | median EP | median BAG |",
          "|---|---:|---:|---:|---:|---:|"]
    for t, sub in m.groupby("gebouwtype"):
        if len(sub) < 100:
            continue
        a = agree(sub["ep_compactheid"], sub["shape_factor"])
        L += [f"| {t} | {a['n']:,} | {a['pearson']:.3f} | {a['spearman']:.3f} | "
              f"{a['med_x']:.2f} | {a['med_y']:.2f} |"]
    L += [""]

    m["unit_bin"] = pd.cut(m["aantal_verblijfsobjecten"], [0, 1, 2, 4, 10, 1e9],
                           labels=["1", "2", "3-4", "5-10", ">10"])
    L += ["### By BAG unit count per pand", "",
          "| units in pand | n | Pearson r | median EP | median BAG |",
          "|---|---:|---:|---:|---:|"]
    for t, sub in m.groupby("unit_bin", observed=True):
        a = agree(sub["ep_compactheid"], sub["shape_factor"])
        L += [f"| {t} | {a['n']:,} | {a['pearson']:.3f} | {a['med_x']:.2f} | {a['med_y']:.2f} |"]
    L += [""]

    # ---------------- T4 floor_area ----------------
    L += ["## T4 — `floor_area` provenance", "",
          "`m1_plus_fullstock.load_ep_geometry` reads `GebruiksoppervlakteThermischeZone`: "
          "the **thermal-zone floor area of the certified unit**, not the pand.", "",
          "| comparison | n | Pearson r | Spearman | median EP | median BAG | median ratio |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for name, x, y in [
        ("EP thermal zone vs BAG floor_area_estimated (grond x floors)",
         m["ep_floor_area"], m["floor_area_estimated"]),
        ("EP thermal zone vs BAG oppervlakte_max (largest VBO)",
         m["ep_floor_area"], m["oppervlakte_max"]),
        ("EP thermal zone vs BAG b3_opp_grond (footprint)",
         m["ep_floor_area"], m["b3_opp_grond"]),
    ]:
        a = agree(x, y)
        L += [f"| {name} | {a['n']:,} | {a['pearson']:.3f} | {a['spearman']:.3f} | "
              f"{a['med_x']:.1f} | {a['med_y']:.1f} | {a['ratio']:.2f} |"]
    ga = ep_nta.groupby("pand_id")["opp_thermische_zone"]
    pa = pd.DataFrame({"n": ga.size(), "std": ga.std(), "med": ga.median()})
    pam = pa[pa["n"] >= 2]
    L += ["", f"Within-pand spread of `GebruiksoppervlakteThermischeZone` among multi-cert "
          f"pands: median std {pam['std'].median():.1f} m2 "
          f"(median pand area {pam['med'].median():.1f} m2) -> per-unit, as expected.", ""]

    # ---------------- T5 size of the leak ----------------
    dev = pd.read_parquet(PROCESSED / "dev_fold_indices.parquet")
    dev["pand_id"] = dev["pand_id"].astype(str).str.zfill(16)
    gt = pd.read_parquet(PROCESSED / "stage1_gt.parquet")
    gt["pand_id"] = gt["pand_id"].astype(str).str.zfill(16)
    d = dev[["pand_id", "fold"]].merge(
        gt[["pand_id", "building_type", "tabula_period"]], on="pand_id", how="left").merge(
        m, on="pand_id", how="left")
    d["cell"] = d["building_type"].astype(str) + "|" + d["tabula_period"].astype(str)

    L += ["## T5 — How much of the label does `compactheid` explain?", "",
          "Spearman against the two NTA 8800 calculation outputs that define the label, "
          "on the dev pool (n=8,068), overall and inside the dominant archetype cell.", "",
          "| subset | n | rho(compactheid, EP1 energiebehoefte) | rho(compactheid, EP2 PF) | rho(shape_factor, EP1) |",
          "|---|---:|---:|---:|---:|"]
    for name, sub in [("dev, all cells", d)] + [
            (f"dev, cell {c}", s) for c, s in d.groupby("cell") if len(s) >= 200]:
        a1 = agree(sub["ep_compactheid"], sub["energiebehoefte"])
        a2 = agree(sub["ep_compactheid"], sub["primaire_fossiele_energie"])
        a3 = agree(sub["shape_factor"], sub["energiebehoefte"])
        L += [f"| {name} | {a1['n']:,} | {a1['spearman']:.3f} | {a2['spearman']:.3f} | "
              f"{a3['spearman']:.3f} |"]
    L += [""]

    # ---------------- T6 re-run the headline ablation with real geometry ----------------
    from lightgbm import LGBMClassifier

    from src.stage2.features import build_master_table
    from src.stage2.metrics import evaluate
    from src.stage2.train_eval import FIXED_PARAMS

    mt = build_master_table()
    mt["pand_id"] = mt["pand_id"].astype(str).str.zfill(16)
    extra = m[["pand_id", "ep_compactheid", "shape_factor", "ep_floor_area",
               "floor_area_estimated", "volume", "shared_ratio"]]
    mt = mt.merge(extra, on="pand_id", how="left")
    base = ["building_type", "bouwjaar", "u_wall", "u_roof", "u_floor", "u_window",
            "num_floors", "city"]
    cats = ["building_type", "city"]
    folds = mt["fold"].to_numpy()
    yv = mt["energy_class"]

    def lgbm_oof(cols):
        Xd = mt[cols].copy()
        for c in cats:
            if c in cols:
                Xd[c] = Xd[c].astype("category")
        oof = np.empty(len(mt), dtype=object)
        for f in range(5):
            tr, va = folds != f, folds == f
            cl = LGBMClassifier(**FIXED_PARAMS)
            cl.fit(Xd[tr], yv[tr], categorical_feature=[c for c in cats if c in cols])
            oof[va] = cl.predict(Xd[va])
        return evaluate(pd.DataFrame({"true": yv.values, "pred": oof}), with_ci=False)

    L += ["## T6 — Re-running the headline ablation with real geometry", "",
          "Same protocol as the 2026.06.26 ablation (dev pool, 5-fold OOF LightGBM, "
          "fixed HP): each feature added singly to S_full.", "",
          "| added feature | provenance | macro-F1 | quad. kappa | acc | d macro-F1 |",
          "|---|---|---:|---:|---:|---:|"]
    r0 = lgbm_oof(base)
    L += [f"| — (S_full base) | — | {r0['macro_f1']:.4f} | {r0['quadratic_kappa']:.4f} | "
          f"{r0['accuracy']:.4f} | — |"]
    for col, prov in [("ep_compactheid", "**EP-Online certificate** (leak)"),
                      ("shape_factor", "3DBAG envelope/volume (clean)"),
                      ("ep_floor_area", "**EP-Online certificate** (leak)"),
                      ("floor_area_estimated", "3DBAG grond x floors (clean)"),
                      ("volume", "3DBAG lod22 (clean)"),
                      ("shared_ratio", "3DBAG party-wall frac (clean)")]:
        r = lgbm_oof(base + [col])
        L += [f"| `{col}` | {prov} | {r['macro_f1']:.4f} | {r['quadratic_kappa']:.4f} | "
              f"{r['accuracy']:.4f} | {r['macro_f1'] - r0['macro_f1']:+.4f} |"]
    r_both = lgbm_oof(base + ["shape_factor", "floor_area_estimated", "shared_ratio", "volume"])
    L += [f"| all four clean 3DBAG features | 3DBAG only | {r_both['macro_f1']:.4f} | "
          f"{r_both['quadratic_kappa']:.4f} | {r_both['accuracy']:.4f} | "
          f"{r_both['macro_f1'] - r0['macro_f1']:+.4f} |", ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
