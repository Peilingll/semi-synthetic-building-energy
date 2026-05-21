"""Step 2: LOD2 derived geometry features from 3D BAG WFS attributes."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_derived_features(df: pd.DataFrame, floor_height: float = 3.0) -> pd.DataFrame:
    """Compute derived LOD2 geometry features from 3D BAG WFS attributes.

    Parameters
    ----------
    df : DataFrame with columns pand_id, b3_volume_lod22, b3_opp_buitenmuur,
         b3_opp_dak_plat, b3_opp_dak_schuin, b3_opp_grond, b3_h_max,
         b3_h_maaiveld, b3_bouwlagen, b3_rmse_lod22.
    floor_height : assumed storey height in metres (used when b3_bouwlagen is null).

    Returns
    -------
    DataFrame with columns: pand_id, volume, envelope_area, shape_factor,
    building_height, num_floors_estimated, floor_area_estimated, lod2_quality_flag.
    """
    out = pd.DataFrame()
    out["pand_id"] = df["pand_id"]

    # Volume — LOD2.2
    out["volume"] = df["b3_volume_lod22"]

    # Envelope area — exterior surfaces only (no party wall)
    out["envelope_area"] = (
        df["b3_opp_buitenmuur"]
        + df["b3_opp_dak_plat"]
        + df["b3_opp_dak_schuin"]
        + df["b3_opp_grond"]
    )

    # Shape factor (surface-to-volume ratio)
    out["shape_factor"] = np.where(
        out["volume"] > 0,
        out["envelope_area"] / out["volume"],
        np.nan,
    )

    # Building height — net height above ground level
    out["building_height"] = df["b3_h_max"] - df["b3_h_maaiveld"]
    out["building_height"] = out["building_height"].clip(lower=0)

    # Number of floors — use 3D BAG value; fallback to height / floor_height
    fallback_floors = np.maximum(1, np.round(out["building_height"] / floor_height))
    out["num_floors_estimated"] = (
        df["b3_bouwlagen"].fillna(fallback_floors).astype(int)
    )

    # Estimated total floor area
    out["floor_area_estimated"] = df["b3_opp_grond"] * out["num_floors_estimated"]

    # Quality flag based on LOD2.2 reconstruction RMSE
    out["lod2_quality_flag"] = df["b3_rmse_lod22"] <= 1.0

    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_lod2_features(df: pd.DataFrame) -> dict:
    """Run acceptance checks on derived LOD2 features. Return a report dict."""
    report: dict = {}
    report["total_buildings"] = len(df)

    # Critical: no nulls in output columns
    null_counts = df.drop(columns=["pand_id"]).isna().sum()
    nulls_any = null_counts[null_counts > 0]
    if not nulls_any.empty:
        logger.warning("Null values found: %s", nulls_any.to_dict())
    report["null_counts"] = null_counts.to_dict()

    # Critical: pand_id uniqueness
    assert df["pand_id"].is_unique, "Duplicate pand_id found"

    # Critical: volume must be positive
    n_zero_vol = (df["volume"] <= 0).sum()
    if n_zero_vol > 0:
        logger.warning("%d buildings with volume <= 0", n_zero_vol)
    report["volume_le_zero"] = int(n_zero_vol)

    # Range warnings (informational, not fatal)
    def _range_check(col, lo, hi):
        outside = ((df[col] < lo) | (df[col] > hi)).sum()
        pct = outside / len(df) * 100
        if outside > 0:
            logger.warning(
                "%s: %d (%.1f%%) outside typical range [%s, %s]",
                col, outside, pct, lo, hi,
            )
        report[f"{col}_outside_{lo}_{hi}"] = int(outside)

    _range_check("volume", 300, 1500)
    _range_check("shape_factor", 0.3, 1.5)
    _range_check("building_height", 2, 100)
    _range_check("num_floors_estimated", 1, 30)

    # Quality flag summary
    n_low_quality = (~df["lod2_quality_flag"]).sum()
    report["low_quality_count"] = int(n_low_quality)
    logger.info("Quality flag: %d/%d buildings flagged as low quality",
                n_low_quality, len(df))

    # Summary stats
    for col in ["volume", "envelope_area", "shape_factor", "building_height"]:
        report[f"{col}_median"] = float(df[col].median())

    logger.info("Validation complete. %d buildings processed.", len(df))
    return report


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def filter_residential(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only buildings whose gebruiksdoel contains 'woonfunctie'."""
    mask = df["gebruiksdoel"].str.contains("woonfunctie", case=False, na=False)
    n_before = len(df)
    df = df[mask].copy()
    logger.info("Residential filter: %d → %d buildings", n_before, len(df))
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with volume = 0, NaN, or negative."""
    mask = df["volume"].isna() | (df["volume"] <= 0)
    n_bad = mask.sum()
    if n_bad > 0:
        logger.warning("Removing %d buildings with invalid volume (0/NaN/negative)", n_bad)
    return df[~mask].copy()


def run_step2(config: dict | None = None) -> dict:
    """Run the full Step 2 LOD2 feature extraction pipeline."""
    if config is None:
        config = load_config()

    paths = config["data_paths"]
    lod2_cfg = config.get("lod2", {})

    # Read Step 1 output
    input_path = paths["joined_output"]
    logger.info("Reading joined data from %s", input_path)
    df = pd.read_parquet(input_path)
    logger.info("Loaded %d buildings with %d columns", len(df), len(df.columns))

    # Filter residential only (mvp_spec Step 1)
    df = filter_residential(df)

    # Compute derived features
    logger.info("Computing LOD2 derived features")
    floor_height = lod2_cfg.get("floor_height_default", 3.0)
    features = compute_derived_features(df, floor_height=floor_height)

    # Remove outliers — volume=0/NaN (mvp_spec Step 1)
    features = remove_outliers(features)

    # Validate
    logger.info("Validating features")
    report = validate_lod2_features(features)

    # Save
    output_path = Path(paths["lod2_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    report["output_path"] = str(output_path)
    logger.info("Saved LOD2 features to %s", output_path)

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    parser = argparse.ArgumentParser(description="Step 2 LOD2 feature extraction")
    parser.add_argument(
        "--config", default=None,
        help="Path to a YAML config (default: project-root config.yaml)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else None
    result = run_step2(cfg)
    print(result)
