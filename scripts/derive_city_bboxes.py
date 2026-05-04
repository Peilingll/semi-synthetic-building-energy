"""One-off helper: derive RD + WGS84 bounding boxes for cities from PDOK WFS.

Queries the Kadaster `bestuurlijkegebieden:Gemeentegebied` layer and prints
each gemeente's `total_bounds` in EPSG:28992 (RD) and EPSG:4326 (WGS84).
Paste the values into the per-city YAMLs under `configs/`.
"""

import logging
import sys
from io import BytesIO

import geopandas as gpd
import requests


logger = logging.getLogger(__name__)

WFS_URL = "https://service.pdok.nl/kadaster/bestuurlijkegebieden/wfs/v1_0"
LAYER = "bestuurlijkegebieden:Gemeentegebied"
DEFAULT_CITIES = ("Amsterdam", "Utrecht", "Rotterdam", "Delft")


def fetch_all_gemeenten(page_size: int = 1000) -> gpd.GeoDataFrame:
    """Fetch all gemeente polygons (PDOK doesn't honour CQL_FILTER on this layer)."""
    pages: list[gpd.GeoDataFrame] = []
    start_index = 0
    while True:
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeName": LAYER,
            "outputFormat": "application/json",
            "srsName": "EPSG:28992",
            "count": page_size,
            "startIndex": start_index,
        }
        resp = requests.get(WFS_URL, params=params, timeout=60)
        resp.raise_for_status()
        page = gpd.read_file(BytesIO(resp.content))
        if page.empty:
            break
        pages.append(page)
        if len(page) < page_size:
            break
        start_index += page_size
    return gpd.GeoDataFrame(
        __import__("pandas").concat(pages, ignore_index=True),
        geometry="geometry",
        crs="EPSG:28992",
    )


def report(gdf_all: gpd.GeoDataFrame, city: str) -> dict | None:
    rows = gdf_all[gdf_all["naam"] == city]
    if rows.empty:
        logger.warning("  no result for %s", city)
        return None

    bbox_rd = [round(x) for x in rows.total_bounds]
    bbox_wgs = [round(x, 4) for x in rows.to_crs("EPSG:4326").total_bounds]

    logger.info("=== %s ===", city)
    logger.info("  bbox_rd:  %s", bbox_rd)
    logger.info("  bbox_wgs: %s", bbox_wgs)
    return {"city": city, "bbox_rd": bbox_rd, "bbox_wgs": bbox_wgs}


def main(cities: tuple[str, ...] = DEFAULT_CITIES) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Fetching all gemeenten from PDOK...")
    gdf_all = fetch_all_gemeenten()
    logger.info("Fetched %d gemeenten", len(gdf_all))
    for city in cities:
        report(gdf_all, city)
    return 0


if __name__ == "__main__":
    sys.exit(main())
