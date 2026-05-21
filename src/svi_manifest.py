"""Build multi-city SVI manifest for Stage 1 CV training.

For each of the four cities (Amsterdam, Utrecht, Rotterdam, Delft):

1. Scan OpenFACADES output `phase_c_<city>_grid/cells/*/output/02_img/individual_building/`
   for cropped per-building images named `pid_<panorama_id>_bdid_<bdid>.png`.
2. Resolve `bdid` (OpenFACADES building_id) to BAG `pand_id` via
   `merged/bag_openfacades_id_mapping.csv` if present, otherwise compute it
   on the fly with `spatial_join_footprints_to_bag()` from vlm_pipeline.py and
   cache the result.
3. Attach `aov_geo` (degrees of building width in view) and `distance` (m to
   building) from `merged/aov.csv` as image-quality proxies.
4. Cap to 8 images per pand_id: sort by `aov_geo DESC, distance ASC`, take top 8.
5. Drop images with no BAG match (non-residential OSM footprints).

Output: `data/processed/svi_manifest.parquet`

  | column      | type | meaning                                       |
  |-------------|------|-----------------------------------------------|
  | pand_id     | str  | BAG 16-digit zero-padded id                   |
  | panorama_id | str  | Mapillary panorama id (from `pid_` in name)   |
  | bdid        | str  | OpenFACADES OSM building id (from `bdid_`)    |
  | file_path   | str  | absolute path to png                          |
  | city        | str  | amsterdam / utrecht / rotterdam / delft       |
  | aov_geo     | f64  | width of building in view, degrees (quality)  |
  | distance    | f64  | metres from camera to building (quality)      |
  | image_idx   | i64  | 0..N-1 within pand_id after cap-sort          |
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

from src.vlm_pipeline import spatial_join_footprints_to_bag

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
CITIES = ["amsterdam", "utrecht", "rotterdam", "delft"]
FILENAME_RE = re.compile(r"^pid_(?P<pid>\d+)_bdid_(?P<bdid>\d+)\.png$")


def _bag_geometry_for(city: str) -> tuple[Path, Path]:
    """Return (bag_geometry_parquet, residential_parquet) for a city.

    Delft kept only the 3-way joined parquet with geometry; the residential
    file dropped geometry during 3D feature extraction. The other three cities
    have a 2-way joined parquet AND residential parquet that both keep geometry.
    """
    city_dir = REPO_ROOT / "data" / "processed" / city
    residential = city_dir / "residential_tabula_matched.parquet"
    if city == "delft":
        bag_geom = city_dir / "bag_3dbag_ep_joined.parquet"
    else:
        bag_geom = city_dir / "bag_ep_joined.parquet"
    return bag_geom, residential


def _get_mapping(city: str, cache_path: Path) -> pd.DataFrame:
    """Load bdid→pand_id mapping for a city, computing + caching if missing."""
    if cache_path.exists():
        logger.info("[%s] using cached mapping at %s", city, cache_path)
        return pd.read_csv(cache_path, dtype={"building_id": str, "pand_id": str})

    bag_geom, residential = _bag_geometry_for(city)
    footprint = (
        REPO_ROOT / "data" / "openfacades_output" / f"phase_c_{city}_grid"
        / "merged" / "footprint.geojson"
    )
    if not footprint.exists():
        raise FileNotFoundError(f"footprint.geojson missing for {city}: {footprint}")

    logger.info("[%s] computing spatial join (this may take a minute)...", city)
    mapping = spatial_join_footprints_to_bag(footprint, bag_geom, residential)
    mapping["building_id"] = mapping["building_id"].astype(str)
    mapping["pand_id"] = mapping["pand_id"].astype(str).where(mapping["pand_id"].notna())

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(cache_path, index=False)
    n_match = mapping["pand_id"].notna().sum()
    logger.info(
        "[%s] cached %d rows to %s; matched %d/%d (%.0f%%)",
        city, len(mapping), cache_path, n_match, len(mapping),
        100 * n_match / max(len(mapping), 1),
    )
    return mapping


def _scan_pngs(city: str) -> pd.DataFrame:
    """Walk individual_building/ and parse filenames into rows."""
    base = (
        REPO_ROOT / "data" / "openfacades_output" / f"phase_c_{city}_grid" / "cells"
    )
    rows: list[dict] = []
    for png in base.glob("cell_*/output/02_img/individual_building/pid_*_bdid_*.png"):
        m = FILENAME_RE.match(png.name)
        if not m:
            continue
        rows.append({
            "panorama_id": m.group("pid"),
            "bdid": m.group("bdid"),
            "file_path": str(png),
            "city": city,
        })
    df = pd.DataFrame(rows)
    logger.info("[%s] scanned %d png files", city, len(df))
    return df


def _load_aov(city: str) -> pd.DataFrame:
    """Load aov.csv with quality proxies, normalising id dtype to str."""
    aov_csv = (
        REPO_ROOT / "data" / "openfacades_output" / f"phase_c_{city}_grid"
        / "merged" / "aov.csv"
    )
    aov = pd.read_csv(aov_csv, dtype={"pid": str, "building_id": str})
    return aov[["pid", "building_id", "aov_geo", "distance"]].rename(
        columns={"pid": "panorama_id", "building_id": "bdid"}
    )


def build_city_manifest(city: str, cap: int) -> pd.DataFrame:
    cache_path = (
        REPO_ROOT / "data" / "openfacades_output" / f"phase_c_{city}_grid"
        / "merged" / "bag_openfacades_id_mapping.csv"
    )
    mapping = _get_mapping(city, cache_path)[["building_id", "pand_id"]]
    mapping = mapping.rename(columns={"building_id": "bdid"})
    mapping = mapping.dropna(subset=["pand_id"]).drop_duplicates(subset=["bdid"])

    images = _scan_pngs(city)
    if images.empty:
        return images

    aov = _load_aov(city)

    df = images.merge(aov, on=["panorama_id", "bdid"], how="left")
    df = df.merge(mapping, on="bdid", how="left")

    n_unmatched = df["pand_id"].isna().sum()
    if n_unmatched:
        logger.warning(
            "[%s] dropping %d images with no residential BAG pand_id",
            city, n_unmatched,
        )
    df = df.dropna(subset=["pand_id"])

    df = df.sort_values(
        ["pand_id", "aov_geo", "distance"],
        ascending=[True, False, True],
        na_position="last",
    )
    df["image_idx"] = df.groupby("pand_id").cumcount()
    df = df[df["image_idx"] < cap].reset_index(drop=True)

    logger.info(
        "[%s] manifest: %d images across %d pand_ids (cap=%d)",
        city, len(df), df["pand_id"].nunique(), cap,
    )
    return df[[
        "pand_id", "panorama_id", "bdid", "file_path", "city",
        "aov_geo", "distance", "image_idx",
    ]]


def build_manifest(cities: list[str], cap: int) -> pd.DataFrame:
    parts = [build_city_manifest(c, cap) for c in cities]
    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cities", nargs="+", default=CITIES, choices=CITIES + ["all"],
        help="cities to include (default: all four)",
    )
    parser.add_argument(
        "--cap", type=int, default=8,
        help="max images per pand_id after sort by aov_geo desc, distance asc",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "data" / "processed" / "svi_manifest.parquet",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cities = CITIES if "all" in args.cities else args.cities
    manifest = build_manifest(cities, args.cap)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(args.output, index=False)

    print(f"\nwrote {len(manifest):,} rows to {args.output}")
    print(manifest.groupby("city").agg(
        buildings=("pand_id", "nunique"),
        images=("file_path", "count"),
    ))
    print("\nimages-per-building distribution:")
    print(manifest.groupby("pand_id").size().describe().round(2))


if __name__ == "__main__":
    main()
