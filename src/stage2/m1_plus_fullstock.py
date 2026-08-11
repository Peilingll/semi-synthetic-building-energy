"""M1+ upper-envelope arm on the full-stock pool (Tier C feature-gap decomposition).

Rebuilds the E6 full-stock master table (14 GT features, 124k buildings, log
2026.06.27) and adds two CBS buurt-level context features (Hettinga 2023's
socio-economic pair):

  - buurt_woz       = gemiddeldeWoningwaarde  (avg dwelling value, EUR x1000)
  - buurt_koop_pct  = percentageKoopwoningen  (owner-occupied share, %)

Buurt polygons + attributes come from the PDOK CBS wijkenbuurten 2023 WFS and
are cached to data/raw/cbs_buurten_2023.parquet. Buildings are joined to their
buurt by footprint centroid (EPSG:28992).

Runs (same fixed-HP LightGBM + 5-fold protocol as E6, class_weight=None):
  E6_base   : 14 features  -> sanity check against E6 (mF1 0.349 / k 0.560 / acc 0.510)
  +woz      : base + buurt_woz
  +koop     : base + buurt_koop_pct
  M1_plus   : base + both

Usage:
    uv run python -m src.stage2.m1_plus_fullstock
    uv run python -m src.stage2.m1_plus_fullstock --skip-fetch   # reuse cached buurten
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMClassifier
from shapely import wkb
from sklearn.model_selection import StratifiedKFold

from src.stage2.features import REPO_ROOT, merge_energy_class, target_col, to_binary
from src.stage2.metrics import evaluate
from src.stage2.train_eval import N_FOLDS, params_for, task_suffix

logger = logging.getLogger(__name__)

PROCESSED = REPO_ROOT / "data" / "processed"
RAW_EP_CSV = REPO_ROOT / "data" / "raw" / "v20260401_v4_csv" / "v20260401_v4_csv.csv"
BUURT_CACHE = REPO_ROOT / "data" / "raw" / "cbs_buurten_2023.parquet"
REPORTS_DIR = REPO_ROOT / "reports" / "stage2"
TABLES_DIR = REPO_ROOT / "reports" / "tables" / "stage2"

CITIES = ["amsterdam", "rotterdam", "utrecht", "delft"]
GEMEENTEN = ["Amsterdam", "Rotterdam", "Utrecht", "Delft"]

WFS_URL = "https://service.pdok.nl/cbs/wijkenbuurten/2023/wfs/v1_0"
BUURT_ATTRS = ["buurtcode", "gemeentenaam",
               "gemiddeldeWoningwaarde", "percentageKoopwoningen"]

BASE_FEATURES = [
    "building_type", "city", "bouwjaar", "num_floors",
    "u_wall", "u_roof", "u_floor", "u_window",
    "compactheid", "floor_area", "shared_ratio",
    "b3_volume_lod22", "b3_h_max", "aantal_verblijfsobjecten",
]

# --clean: audit A01 showed `compactheid` and `floor_area` are EP-Online
# CERTIFICATE columns, per dwelling unit, and NTA 8800 calculation inputs — so
# they correlate with the label by construction and do not exist for any
# building without a certificate, which is the whole target population. Their
# genuine 3DBAG counterparts are `shape_factor` (envelope/volume) and
# `floor_area_estimated` (footprint x floors). The other 12 features were always
# clean. Use this variant for any number quoted as a registry-only reference.
LEAKED = ["compactheid", "floor_area"]
CLEAN_SUBSTITUTES = ["shape_factor", "floor_area_estimated"]
CLEAN_FEATURES = [c for c in BASE_FEATURES if c not in LEAKED] + CLEAN_SUBSTITUTES

TIER_C = ["buurt_woz", "buurt_koop_pct"]
CAT_FEATURES = ["building_type", "city"]


# ---------------------------------------------------------------------------
# CBS buurten via PDOK WFS
# ---------------------------------------------------------------------------

def _wfs_page(params: dict, attempts: int = 3) -> dict:
    for i in range(attempts):
        try:
            r = requests.get(WFS_URL, params=params, timeout=180)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException:
            if i == attempts - 1:
                raise
            logger.warning("WFS request failed (attempt %d), retrying", i + 1)


def fetch_buurten(bboxes: dict[str, tuple]) -> gpd.GeoDataFrame:
    """Fetch buurt polygons + stats per city bbox (spatially indexed, fast)."""
    frames = []
    for city, (minx, miny, maxx, maxy) in bboxes.items():
        start = 0
        while True:
            params = {
                "service": "WFS", "version": "2.0.0", "request": "GetFeature",
                "typeName": "wijkenbuurten:buurten",
                "outputFormat": "application/json",
                "srsName": "urn:ogc:def:crs:EPSG::28992",
                "count": 1000, "startIndex": start,
                "bbox": f"{minx},{miny},{maxx},{maxy},urn:ogc:def:crs:EPSG::28992",
            }
            data = _wfs_page(params)
            if not data["features"]:
                break
            page = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:28992")
            frames.append(page)
            logger.info("buurten %s: +%d (start=%d)", city, len(page), start)
            if len(page) < 1000:
                break
            start += 1000
    buurten = pd.concat(frames, ignore_index=True)
    buurten = buurten.drop_duplicates("buurtcode", keep="first")
    buurten = gpd.GeoDataFrame(buurten, geometry="geometry", crs="EPSG:28992")
    keep = BUURT_ATTRS + ["geometry"]
    buurten = buurten[keep]
    # CBS sentinel codes (-99995 / -99997 / ...) mean "no data"
    for col in ["gemiddeldeWoningwaarde", "percentageKoopwoningen"]:
        vals = pd.to_numeric(buurten[col], errors="coerce")
        buurten[col] = vals.where(vals > -90000, np.nan)
    return buurten


def load_buurten(skip_fetch: bool, bboxes: dict[str, tuple]) -> gpd.GeoDataFrame:
    if skip_fetch and BUURT_CACHE.exists():
        logger.info("buurten: using cache %s", BUURT_CACHE)
        return gpd.read_parquet(BUURT_CACHE)
    buurten = fetch_buurten(bboxes)
    BUURT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    buurten.to_parquet(BUURT_CACHE)
    logger.info("buurten: fetched %d, cached to %s", len(buurten), BUURT_CACHE)
    return buurten


# ---------------------------------------------------------------------------
# Building-level features
# ---------------------------------------------------------------------------

def load_city_geometry() -> gpd.GeoDataFrame:
    """Per-pand 3DBAG/BAG columns + footprint centroid from the joined parquets."""
    cols = ["pand_id", "aantal_verblijfsobjecten", "b3_volume_lod22", "b3_h_max",
            "b3_opp_buitenmuur", "b3_opp_scheidingsmuur", "geometry"]
    frames = []
    for city in CITIES:
        path = PROCESSED / city / "bag_3dbag_ep_joined.parquet"
        df = pd.read_parquet(path, columns=cols)
        df = df.drop_duplicates("pand_id", keep="first")
        df["geo_city"] = city
        frames.append(df)
        logger.info("joined %s: %d pands", city, len(df))
    df = pd.concat(frames, ignore_index=True).drop_duplicates("pand_id", keep="first")

    geom = df["geometry"].apply(lambda b: wkb.loads(bytes(b)) if b is not None else None)
    gdf = gpd.GeoDataFrame(df.drop(columns=["geometry"]), geometry=geom, crs="EPSG:28992")

    denom = gdf["b3_opp_buitenmuur"] + gdf["b3_opp_scheidingsmuur"]
    gdf["shared_ratio"] = np.where(denom > 0, gdf["b3_opp_scheidingsmuur"] / denom, np.nan)
    gdf["centroid"] = gdf.geometry.centroid
    return gdf


def load_clean_geometry() -> pd.DataFrame:
    """3DBAG shape_factor / floor_area_estimated, the honest counterparts of the
    two leaked EP certificate columns. Plausibility bounds and their rationale
    are the same as `src/audit/a01_compactheid_source.py`: the LOD2 output holds
    a volume of 4.3e21 m3 and non-positive envelope areas."""
    frames = []
    for city in CITIES:
        f = pd.read_parquet(PROCESSED / city / "residential_with_3d_features.parquet",
                            columns=["pand_id", "shape_factor", "floor_area_estimated"])
        frames.append(f)
    g = pd.concat(frames, ignore_index=True)
    g["pand_id"] = g["pand_id"].astype(str).str.zfill(16)
    g = g.drop_duplicates("pand_id")
    g.loc[~g["shape_factor"].between(0.05, 5), "shape_factor"] = np.nan
    g.loc[~g["floor_area_estimated"].between(10, 1e5), "floor_area_estimated"] = np.nan
    logger.info("clean geometry: %d pands | shape_factor %.1f%% | floor_area_est %.1f%%",
                len(g), 100 * g["shape_factor"].notna().mean(),
                100 * g["floor_area_estimated"].notna().mean())
    return g


def _num(x):
    """Parse a Dutch decimal-comma number ('2,53') to float; NaN on failure."""
    x = (x or "").strip().replace(",", ".")
    try:
        return float(x)
    except ValueError:
        return np.nan


def load_ep_features() -> pd.DataFrame:
    """Compactheid + thermal-zone floor area per pand, latest certificate wins.

    BAGPandIDs can hold multiple comma-separated ids: take the first, zero-fill
    to 16 chars (same convention as svi_compactheid.load_compactheid).
    """
    rows = []
    with open(RAW_EP_CSV, encoding="utf-8-sig", errors="replace") as f:
        f.readline(); f.readline()  # two metadata lines before the header
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        idx = {c: i for i, c in enumerate(header)}
        iP, iC, iA, iReg, iGk = (idx["BAGPandIDs"], idx["Compactheid"],
                                 idx["GebruiksoppervlakteThermischeZone"],
                                 idx["Registratiedatum"], idx["Gebouwklasse"])
        for row in reader:
            if len(row) <= max(iP, iC, iA, iReg, iGk) or row[iGk] != "W":
                continue
            rows.append((str(row[iP]).split(",")[0].zfill(16),
                         _num(row[iC]), _num(row[iA]), row[iReg]))
    ep = pd.DataFrame(rows, columns=["pand_id", "compactheid", "floor_area", "reg"])
    ep = ep.sort_values("reg").drop_duplicates("pand_id", keep="last")
    ep.loc[~ep["compactheid"].between(0.3, 5), "compactheid"] = np.nan
    logger.info("EP raw: %d pands, compactheid valid %.1f%%, floor_area valid %.1f%%",
                len(ep), 100 * ep["compactheid"].notna().mean(),
                100 * ep["floor_area"].notna().mean())
    return ep[["pand_id", "compactheid", "floor_area"]]


def build_master(skip_fetch: bool) -> pd.DataFrame:
    gt = pd.read_parquet(PROCESSED / "stage1_gt.parquet")
    logger.info("stage1_gt: %d pands", len(gt))

    tabula = pd.read_csv(PROCESSED / "tabula_nl.csv")
    gt = gt.merge(
        tabula[["building_type", "period", "u_wall", "u_roof", "u_floor", "u_window"]],
        left_on=["building_type", "tabula_period"],
        right_on=["building_type", "period"], how="left",
    ).drop(columns=["period"])

    geo = load_city_geometry()
    bboxes = {}
    for city, grp in geo.groupby("geo_city"):
        b = grp["centroid"].total_bounds
        bboxes[city] = (b[0] - 500, b[1] - 500, b[2] + 500, b[3] + 500)
    gt = gt.merge(geo.drop(columns=["geometry", "geo_city"]), on="pand_id", how="left")
    gt = gt.merge(load_ep_features(), on="pand_id", how="left")
    gt = gt.merge(load_clean_geometry(), on="pand_id", how="left")

    buurten = load_buurten(skip_fetch, bboxes)
    pts = gpd.GeoDataFrame(gt[["pand_id"]], geometry=gt["centroid"], crs="EPSG:28992")
    joined = gpd.sjoin(pts, buurten, how="left", predicate="within")
    joined = joined.drop_duplicates("pand_id", keep="first")
    gt = gt.drop(columns=["centroid"]).merge(
        joined[["pand_id", "buurtcode", "gemiddeldeWoningwaarde", "percentageKoopwoningen"]],
        on="pand_id", how="left",
    ).rename(columns={"gemiddeldeWoningwaarde": "buurt_woz",
                      "percentageKoopwoningen": "buurt_koop_pct"})

    gt["energy_class"] = merge_energy_class(gt["Energieklasse"])
    gt = gt.dropna(subset=["energy_class", "u_wall"]).reset_index(drop=True)
    gt["energy_binary"] = to_binary(gt["energy_class"])
    logger.info("master: %d rows | buurt match %.1f%% | woz coverage %.1f%%",
                len(gt), 100 * gt["buurtcode"].notna().mean(),
                100 * gt["buurt_woz"].notna().mean())
    return gt


# ---------------------------------------------------------------------------
# OOF evaluation
# ---------------------------------------------------------------------------

def run_oof(master: pd.DataFrame, features: list[str], run: str,
            task: str = "7class") -> dict:
    X = master[features].copy()
    for c in CAT_FEATURES:
        X[c] = X[c].astype("category")
    y = master[target_col(task)]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_pred = np.empty(len(master), dtype=object)
    for f, (tr, va) in enumerate(skf.split(X, y)):
        clf = LGBMClassifier(**params_for(task))
        clf.fit(X.iloc[tr], y.iloc[tr])
        oof_pred[va] = clf.predict(X.iloc[va])
        logger.info("[%s] fold %d done (train=%d val=%d)", run, f, len(tr), len(va))

    oof = pd.DataFrame({"pand_id": master["pand_id"], "true": y, "pred": oof_pred})
    report = evaluate(oof, with_ci=False, task=task)
    report["run"] = run
    report["n_features"] = len(features)
    return report


def main(skip_fetch: bool, clean: bool = False, task: str = "7class") -> None:
    master = build_master(skip_fetch)

    base = CLEAN_FEATURES if clean else BASE_FEATURES
    runs = {
        "E6_base": base,
        "+woz": base + ["buurt_woz"],
        "+koop": base + ["buurt_koop_pct"],
        "M1_plus": base + TIER_C,
    }
    reports = {name: run_oof(master, feats, name, task)
               for name, feats in runs.items()}

    sfx = ("_clean" if clean else "") + task_suffix(task)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / f"m1_plus_fullstock{sfx}.json", "w") as f:
        json.dump(reports, f, indent=2)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    task_name = "A-G 7-class" if task == "7class" else "binary A-C | D-G"
    lines = [
        f"# Table 2e — M1+ upper-envelope arm on the full-stock pool "
        f"({'clean 3DBAG geometry' if clean else 'as originally run'}, {task_name})",
        "",
        f"n = {len(master):,} buildings (4 cities, registry pool, no image requirement).",
        "Protocol identical to E6 (fixed-HP LightGBM, 5-fold OOF, class_weight=None).",
        "Tier C = CBS buurt 2023: avg WOZ value + owner-occupied share (PDOK wijkenbuurten).",
        "",
    ]
    if clean:
        lines += [
            "**Clean variant.** The two EP-Online certificate columns `compactheid` and "
            "`floor_area` (audit A01: per dwelling unit, NTA 8800 calculation inputs, "
            "absent for any building without a certificate) are replaced by their 3DBAG "
            "counterparts `shape_factor` and `floor_area_estimated`. **This is the "
            "variant to quote as a registry-only reference**; the original run below is "
            "an 'if you already hold the certificate' figure and is not reachable for "
            "the population the method targets.",
            "",
            "Original (leaked) run for comparison: macro-F1 0.3513 / kappa 0.5636 / "
            "acc 0.5101 at 14 features (7-class).",
            "",
        ]
    else:
        lines += ["E6 reference (log 2026.06.27): macro-F1 0.349 / kappa 0.560 / acc 0.510.",
                  "",
                  "**Warning**: `compactheid` and `floor_area` here are EP-Online "
                  "certificate columns, not geometry — see audit A01 and the `_clean` "
                  "variant of this table.",
                  ""]
    lines += ["| run | features | macro-F1 | quadratic kappa | accuracy |",
              "|---|---:|---:|---:|---:|"]
    for name, rep in reports.items():
        lines.append(f"| {name} | {rep['n_features']} | {rep['macro_f1']:.4f} "
                     f"| {rep['quadratic_kappa']:.4f} | {rep['accuracy']:.4f} |")
    out = TABLES_DIR / f"T2e_m1plus_fullstock{sfx}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s", out)
    for name, rep in reports.items():
        logger.info("%-8s mF1=%.4f k=%.4f acc=%.4f",
                    name, rep["macro_f1"], rep["quadratic_kappa"], rep["accuracy"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-fetch", action="store_true",
                        help="reuse cached CBS buurten parquet")
    parser.add_argument("--clean", action="store_true",
                        help="swap the two leaked EP columns for their 3DBAG "
                             "counterparts (audit A01); use for any registry-only figure")
    parser.add_argument("--task", default="7class", choices=["7class", "binary"])
    args = parser.parse_args()
    main(args.skip_fetch, args.clean, args.task)
