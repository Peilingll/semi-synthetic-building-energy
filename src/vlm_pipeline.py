"""Wrapper around OpenFACADES for the VLM pipeline.

Phase B scope: only `spatial_join_footprints_to_bag` is implemented — validates
that OpenFACADES `building_id` maps to BAG `pand_id` via centroid containment.
Remaining functions (VLM inference, aggregation) will be added in Phase C.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


def spatial_join_footprints_to_bag(
    footprint_geojson: Path,
    bag_geometry_parquet: Path,
    residential_parquet: Path,
) -> pd.DataFrame:
    """Map OpenFACADES footprints to BAG pand_id by centroid containment.

    OpenFACADES `footprint.geojson` is WGS84 (EPSG:4326). BAG geometry is RD New
    (EPSG:28992). Both are reprojected to EPSG:28992 for meter-accurate
    containment checks.

    Returns columns [building_id, source, pand_id]. Unmatched footprints have
    pand_id=NaN.
    """
    of = gpd.read_file(footprint_geojson)

    bag_all = gpd.read_parquet(bag_geometry_parquet)[["pand_id", "geometry"]]
    residential_ids = set(pd.read_parquet(residential_parquet)["pand_id"])
    bag = bag_all[bag_all["pand_id"].isin(residential_ids)].copy()

    of = of.to_crs(bag.crs)
    centroids = of.copy()
    centroids["geometry"] = of.geometry.centroid

    joined = gpd.sjoin(
        centroids[["building_id", "source", "geometry"]],
        bag,
        predicate="within",
        how="left",
    )

    return joined[["building_id", "source", "pand_id"]].reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--footprint",
        required=True,
        help="Path to OpenFACADES footprint.geojson",
    )
    parser.add_argument(
        "--bag-geom",
        default="data/processed/bag_3dbag_ep_joined.parquet",
        help="BAG parquet with pand_id + geometry (EPSG:28992)",
    )
    parser.add_argument(
        "--residential",
        default="data/processed/residential_tabula_matched.parquet",
        help="BAG parquet restricted to residential buildings",
    )
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    mapping = spatial_join_footprints_to_bag(
        Path(args.footprint), Path(args.bag_geom), Path(args.residential)
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # pand_id is a zero-padded 16-digit string; downstream readers must use
    # dtype={"pand_id": str} or pandas will infer float and strip leading zeros.
    mapping.to_csv(args.out, index=False)
    n_match = mapping["pand_id"].notna().sum()
    print(f"wrote {len(mapping)} rows to {args.out}")
    print(f"  matched to BAG: {n_match}/{len(mapping)} ({n_match/len(mapping):.0%})")
    print(f"  unique BAG pand_ids: {mapping['pand_id'].nunique(dropna=True)}")
