"""Step 1: Data integration — BAG + 3D BAG + EP-Online join pipeline."""

import logging
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from src.config import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 3.1  Bounding box helpers
# ---------------------------------------------------------------------------

def bbox_to_wfs_param(bbox: list[float]) -> str:
    """Convert [min_x, min_y, max_x, max_y] to a comma-separated WFS string."""
    return ",".join(str(c) for c in bbox)


# ---------------------------------------------------------------------------
# Shared WFS pagination helper
# ---------------------------------------------------------------------------

def _paginated_wfs_fetch(
    wfs_url: str,
    params: dict,
    page_size: int,
    label: str,
    crs: str,
) -> gpd.GeoDataFrame:
    """Fetch all pages from a WFS endpoint, handling server-side limits."""
    all_pages: list[gpd.GeoDataFrame] = []
    start_index = 0

    while True:
        params["startIndex"] = start_index
        logger.info("%s WFS request  startIndex=%d", label, start_index)
        try:
            resp = requests.get(wfs_url, params=params, timeout=120)
            resp.raise_for_status()
        except requests.HTTPError:
            logger.warning(
                "%s WFS returned error at startIndex=%d — stopping pagination",
                label, start_index,
            )
            break
        gdf = gpd.read_file(BytesIO(resp.content))
        if gdf.empty:
            break
        all_pages.append(gdf)
        if len(gdf) < page_size:
            break
        start_index += page_size

    if not all_pages:
        return gpd.GeoDataFrame()

    result = pd.concat(all_pages, ignore_index=True)
    return gpd.GeoDataFrame(result, geometry="geometry", crs=crs)


# ---------------------------------------------------------------------------
# 3.2  BAG (PDOK WFS)
# ---------------------------------------------------------------------------

def fetch_bag_pand(
    bbox: list[float],
    wfs_url: str = "https://service.pdok.nl/lv/bag/wfs/v2_0",
    layer: str = "bag:pand",
    crs: str = "EPSG:28992",
    page_size: int = 10000,
) -> gpd.GeoDataFrame:
    """Fetch BAG pand data from PDOK WFS for a given bounding box.

    Returns a GeoDataFrame with columns including 'identificatie',
    'bouwjaar', 'status', and 'geometry'.
    """
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": layer,
        "bbox": bbox_to_wfs_param(bbox),
        "outputFormat": "application/json",
        "srsName": crs,
        "count": page_size,
    }

    result = _paginated_wfs_fetch(wfs_url, params, page_size, "BAG", crs)

    # Keep only active buildings
    if "status" in result.columns:
        result = result[result["status"] == "Pand in gebruik"]

    logger.info("BAG: fetched %d panden", len(result))
    return result


def fetch_bag_pand_tiled(
    bbox: list[float],
    tile_size_m: float = 2000,
    wfs_url: str = "https://service.pdok.nl/lv/bag/wfs/v2_0",
    layer: str = "bag:pand",
    crs: str = "EPSG:28992",
    page_size: int = 1000,
) -> gpd.GeoDataFrame:
    """Fetch BAG by splitting the bbox into RD-grid tiles.

    PDOK's BAG WFS rejects pagination past ~51k features per bbox query, so
    cities larger than Delft must be tiled. Buildings on tile boundaries are
    deduplicated by `identificatie`.
    """
    minx, miny, maxx, maxy = bbox
    pieces: list[gpd.GeoDataFrame] = []
    n_tiles = 0
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            tile = [x, y, min(x + tile_size_m, maxx), min(y + tile_size_m, maxy)]
            n_tiles += 1
            logger.info("BAG tile %d: bbox=%s", n_tiles, tile)
            piece = fetch_bag_pand(
                bbox=tile, wfs_url=wfs_url, layer=layer, crs=crs, page_size=page_size,
            )
            if not piece.empty:
                pieces.append(piece)
            y += tile_size_m
        x += tile_size_m

    if not pieces:
        return gpd.GeoDataFrame()

    result = pd.concat(pieces, ignore_index=True)
    n_before = len(result)
    result = result.drop_duplicates(subset=["identificatie"])
    logger.info(
        "BAG tiled: %d unique panden across %d tiles (%d duplicates dropped)",
        len(result), n_tiles, n_before - len(result),
    )
    return gpd.GeoDataFrame(result, geometry="geometry", crs=crs)


# ---------------------------------------------------------------------------
# 3.3  3D BAG WFS
# ---------------------------------------------------------------------------

def fetch_3dbag(
    bbox: list[float],
    wfs_url: str = "https://data.3dbag.nl/api/BAG3D/wfs",
    layer: str = "BAG3D:lod12",
    crs: str = "EPSG:28992",
    page_size: int = 10000,
) -> gpd.GeoDataFrame:
    """Fetch 3D BAG building attributes from WFS for a given bounding box."""
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": layer,
        "bbox": bbox_to_wfs_param(bbox),
        "outputFormat": "application/json",
        "srsName": crs,
        "count": page_size,
    }

    result = _paginated_wfs_fetch(wfs_url, params, page_size, "3D BAG", crs)
    logger.info("3D BAG: fetched %d features", len(result))
    return result


# ---------------------------------------------------------------------------
# 3.4  EP-Online (local CSV)
# ---------------------------------------------------------------------------

EP_ONLINE_USECOLS = [
    "BAGPandIDs",
    "Energieklasse",
    "EnergieIndex",
    "Gebouwtype",
    "Gebouwklasse",
    "Bouwjaar",
    "Postcode",
    "Status",
    "Registratiedatum",
]


def load_ep_online(
    csv_path: str | Path,
    postcode_prefix: str | list[str] = "26",
    usecols: list[str] | None = None,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Load EP-Online CSV filtered by postcode prefix(es).

    Accepts a single prefix string ('26' for Delft) or a list of prefixes
    (['10', '11'] for Amsterdam). Uses chunked reading to keep memory usage
    low on the 1.5 GB file.
    """
    if usecols is None:
        usecols = EP_ONLINE_USECOLS

    if isinstance(postcode_prefix, str):
        prefixes: tuple[str, ...] = (postcode_prefix,)
    else:
        prefixes = tuple(postcode_prefix)

    csv_path = Path(csv_path)
    chunks: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        csv_path,
        sep=";",
        skiprows=2,
        usecols=usecols,
        dtype={"BAGPandIDs": str, "Postcode": str, "Status": str, "EnergieIndex": str},
        chunksize=chunksize,
    ):
        mask = chunk["Postcode"].str.startswith(prefixes, na=False)
        filtered = chunk[mask]
        if not filtered.empty:
            chunks.append(filtered)

    if not chunks:
        return pd.DataFrame(columns=usecols)

    df = pd.concat(chunks, ignore_index=True)

    # Keep only residential buildings
    if "Gebouwklasse" in df.columns:
        df = df[df["Gebouwklasse"] == "W"]

    # Keep only existing buildings
    if "Status" in df.columns:
        df = df[df["Status"] == "Bestaand"]

    # Deduplicate: keep the latest registration per BAGPandIDs
    df["Registratiedatum"] = pd.to_datetime(
        df["Registratiedatum"].astype(str), format="%Y%m%d", errors="coerce"
    )
    df = (
        df.sort_values("Registratiedatum")
        .drop_duplicates(subset=["BAGPandIDs"], keep="last")
    )

    logger.info("EP-Online: loaded %d records for postcode prefix(es) %s",
                len(df), list(prefixes))
    return df


# ---------------------------------------------------------------------------
# 3.5  ID cleaning
# ---------------------------------------------------------------------------

_IMBAG_PREFIX = "NL.IMBAG.Pand."


def clean_pand_id(raw_id) -> str | None:
    """Normalize a pand ID to a 16-digit zero-padded string."""
    if pd.isna(raw_id):
        return None
    cleaned = str(raw_id).strip()
    if cleaned.startswith(_IMBAG_PREFIX):
        cleaned = cleaned[len(_IMBAG_PREFIX):]
    # Remove any non-digit characters that might remain
    cleaned = cleaned.split(".")[0]
    if not cleaned.isdigit():
        return None
    return cleaned.zfill(16)


def _add_pand_id_column(
    df: pd.DataFrame | gpd.GeoDataFrame,
    source_col: str,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """Add a normalised 'pand_id' column derived from *source_col*."""
    df = df.copy()
    df["pand_id"] = df[source_col].apply(clean_pand_id)
    df = df.dropna(subset=["pand_id"])
    return df


def prepare_bag_ids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add pand_id to BAG data."""
    return _add_pand_id_column(gdf, "identificatie")


def prepare_3dbag_ids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add pand_id to 3D BAG data (strips NL.IMBAG.Pand. prefix)."""
    return _add_pand_id_column(gdf, "identificatie")


def prepare_ep_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add pand_id to EP-Online data; explode multi-ID entries."""
    df = df.copy()
    df["BAGPandIDs"] = df["BAGPandIDs"].astype(str)
    # Some records list multiple pand IDs separated by commas
    df = df.assign(BAGPandIDs=df["BAGPandIDs"].str.split(",")).explode("BAGPandIDs")
    return _add_pand_id_column(df, "BAGPandIDs")


# ---------------------------------------------------------------------------
# 3.6  Three-way join
# ---------------------------------------------------------------------------

def join_all_sources(
    bag_gdf: gpd.GeoDataFrame,
    bag3d_gdf: gpd.GeoDataFrame,
    ep_df: pd.DataFrame,
    bouwjaar_min: int = 1800,
    bouwjaar_max: int = 2026,
) -> gpd.GeoDataFrame:
    """Inner-join BAG, 3D BAG, and EP-Online on pand_id."""
    bag_gdf = prepare_bag_ids(bag_gdf)
    bag3d_gdf = prepare_3dbag_ids(bag3d_gdf)
    ep_df = prepare_ep_ids(ep_df)

    logger.info(
        "Before join — BAG: %d, 3D BAG: %d, EP-Online: %d",
        len(bag_gdf), len(bag3d_gdf), len(ep_df),
    )

    # BAG + 3D BAG
    bag3d_cols = [c for c in bag3d_gdf.columns if c not in ("geometry",)]
    bag_3dbag = bag_gdf.merge(
        bag3d_gdf[bag3d_cols],
        on="pand_id",
        how="inner",
        suffixes=("", "_3dbag"),
    )
    logger.info("After BAG + 3D BAG join: %d rows", len(bag_3dbag))

    # + EP-Online
    joined = bag_3dbag.merge(
        ep_df,
        on="pand_id",
        how="inner",
        suffixes=("", "_ep"),
    )
    logger.info("After EP-Online join: %d rows", len(joined))

    # Post-join filters
    joined = joined[
        (joined["bouwjaar"] >= bouwjaar_min)
        & (joined["bouwjaar"] <= bouwjaar_max)
        & (joined["Energieklasse"].notna())
        & (joined["Energieklasse"] != "")
    ]

    joined = joined.drop_duplicates(subset=["pand_id"])
    logger.info("After filtering & dedup: %d rows", len(joined))

    logger.info("Keeping all %d columns", len(joined.columns))

    return joined


# ---------------------------------------------------------------------------
# 3.6b  Two-way join (BAG + EP only — for cities without 3D BAG step)
# ---------------------------------------------------------------------------

def join_bag_ep(
    bag_gdf: gpd.GeoDataFrame,
    ep_df: pd.DataFrame,
    bouwjaar_min: int = 1800,
    bouwjaar_max: int = 2026,
) -> gpd.GeoDataFrame:
    """Inner-join BAG and EP-Online on pand_id (no 3D BAG)."""
    bag_gdf = prepare_bag_ids(bag_gdf)
    ep_df = prepare_ep_ids(ep_df)

    logger.info(
        "Before join — BAG: %d, EP-Online: %d",
        len(bag_gdf), len(ep_df),
    )

    joined = bag_gdf.merge(
        ep_df,
        on="pand_id",
        how="inner",
        suffixes=("", "_ep"),
    )
    logger.info("After EP-Online join: %d rows", len(joined))

    # Post-join filters
    joined = joined[
        (joined["bouwjaar"] >= bouwjaar_min)
        & (joined["bouwjaar"] <= bouwjaar_max)
        & (joined["Energieklasse"].notna())
        & (joined["Energieklasse"] != "")
    ]

    joined = joined.drop_duplicates(subset=["pand_id"])
    logger.info("After filtering & dedup: %d rows", len(joined))

    logger.info("Keeping all %d columns", len(joined.columns))

    return joined


# ---------------------------------------------------------------------------
# 3.7  Validation & output
# ---------------------------------------------------------------------------

BUILD_PERIODS = [
    (0, 1849, "1800-1849"),
    (1850, 1899, "1850-1899"),
    (1900, 1949, "1900-1949"),
    (1950, 1999, "1950-1999"),
    (2000, 9999, ">2000"),
]


def classify_build_period(year: int) -> str:
    """Map a construction year to a 50-year build period label."""
    for start, end, label in BUILD_PERIODS:
        if start <= year <= end:
            return label
    return "unknown"


def validate_and_save(
    gdf: gpd.GeoDataFrame,
    output_path: str | Path,
    min_buildings: int = 50,
) -> dict:
    """Run acceptance checks and save the joined dataset to Parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict = {}

    # Total count
    report["total_buildings"] = len(gdf)
    assert len(gdf) >= min_buildings, (
        f"Only {len(gdf)} buildings, need at least {min_buildings}"
    )

    # Energy label distribution
    label_counts = gdf["Energieklasse"].value_counts()
    report["unique_labels"] = len(label_counts)
    report["label_distribution"] = label_counts.to_dict()
    assert len(label_counts) >= 3, (
        f"Only {len(label_counts)} unique energy labels, need at least 3"
    )

    # Build period coverage
    gdf = gdf.copy()
    gdf["build_period"] = gdf["bouwjaar"].apply(classify_build_period)
    period_counts = gdf["build_period"].value_counts()
    report["unique_periods"] = len(period_counts)
    report["period_distribution"] = period_counts.to_dict()
    assert len(period_counts) >= 3, (
        f"Only {len(period_counts)} build periods, need at least 3"
    )

    # Uniqueness
    assert gdf["pand_id"].is_unique, "Duplicate pand_id found"

    # Save
    gdf.to_parquet(output_path, index=False)
    report["output_path"] = str(output_path)

    logger.info("Validation passed. Saved %d buildings to %s",
                len(gdf), output_path)
    return report


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _get_postcode_prefixes(ep_cfg: dict) -> str | list[str]:
    """Read postcode prefix(es) from EP-Online config (list or str key)."""
    if "postcode_prefixes" in ep_cfg:
        return ep_cfg["postcode_prefixes"]
    return ep_cfg["postcode_prefix"]


def run_step1(config: dict | None = None) -> dict:
    """Run the full Step 1 data integration pipeline (BAG + 3D BAG + EP-Online)."""
    if config is None:
        config = load_config()

    bbox = config["study_area"]["bbox_rd"]
    crs = config["study_area"]["crs"]
    wfs = config["wfs"]
    ep_cfg = config["ep_online"]
    filters = config["filters"]
    paths = config["data_paths"]

    # Fetch BAG
    logger.info("=== Fetching BAG data ===")
    bag_gdf = fetch_bag_pand(
        bbox=bbox,
        wfs_url=wfs["bag_url"],
        layer=wfs["bag_layer"],
        crs=crs,
        page_size=wfs.get("bag_page_size", 1000),
    )

    # Fetch 3D BAG
    logger.info("=== Fetching 3D BAG data ===")
    bag3d_gdf = fetch_3dbag(
        bbox=bbox,
        wfs_url=wfs["bag3d_url"],
        layer=wfs["bag3d_layer"],
        crs=crs,
        page_size=wfs.get("bag3d_page_size", 10000),
    )

    # Load EP-Online
    logger.info("=== Loading EP-Online data ===")
    ep_df = load_ep_online(
        csv_path=paths["ep_online_csv"],
        postcode_prefix=_get_postcode_prefixes(ep_cfg),
    )

    # Join
    logger.info("=== Joining data sources ===")
    joined = join_all_sources(
        bag_gdf=bag_gdf,
        bag3d_gdf=bag3d_gdf,
        ep_df=ep_df,
        bouwjaar_min=filters["bouwjaar_min"],
        bouwjaar_max=filters["bouwjaar_max"],
    )

    # Validate & save
    logger.info("=== Validating and saving ===")
    report = validate_and_save(
        gdf=joined,
        output_path=paths["joined_output"],
        min_buildings=filters["min_buildings"],
    )

    return report


def run_step1_bag_ep(config: dict | None = None) -> dict:
    """Run Step 1 in BAG+EP mode (skip 3D BAG fetch and 3-way join)."""
    if config is None:
        config = load_config()

    bbox = config["study_area"]["bbox_rd"]
    crs = config["study_area"]["crs"]
    wfs = config["wfs"]
    ep_cfg = config["ep_online"]
    filters = config["filters"]
    paths = config["data_paths"]

    logger.info("=== Fetching BAG data (tiled) ===")
    bag_gdf = fetch_bag_pand_tiled(
        bbox=bbox,
        tile_size_m=wfs.get("bag_tile_size_m", 2000),
        wfs_url=wfs["bag_url"],
        layer=wfs["bag_layer"],
        crs=crs,
        page_size=wfs.get("bag_page_size", 1000),
    )

    logger.info("=== Loading EP-Online data ===")
    ep_df = load_ep_online(
        csv_path=paths["ep_online_csv"],
        postcode_prefix=_get_postcode_prefixes(ep_cfg),
    )

    logger.info("=== Joining BAG + EP-Online ===")
    joined = join_bag_ep(
        bag_gdf=bag_gdf,
        ep_df=ep_df,
        bouwjaar_min=filters["bouwjaar_min"],
        bouwjaar_max=filters["bouwjaar_max"],
    )

    logger.info("=== Validating and saving ===")
    report = validate_and_save(
        gdf=joined,
        output_path=paths["joined_output"],
        min_buildings=filters["min_buildings"],
    )

    return report


def _parse_args() -> "argparse.Namespace":
    import argparse
    parser = argparse.ArgumentParser(description="Step 1 data integration pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config (default: project-root config.yaml)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    args = _parse_args()
    cfg = load_config(args.config)
    use_3dbag = cfg.get("pipeline", {}).get("use_3dbag", True)
    if use_3dbag:
        report = run_step1(cfg)
    else:
        report = run_step1_bag_ep(cfg)
    print("\n=== Step 1 Report ===")
    for key, value in report.items():
        print(f"  {key}: {value}")
