"""Step 3: TABULA archetype matching — assign U-values to each building."""

import logging
from pathlib import Path

import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gebouwtype → TABULA base type mapping
# ---------------------------------------------------------------------------

GEBOUWTYPE_MAP: dict[str, str] = {
    "Vrijstaande woning": "SFH",
    "Twee-onder-één-kap": "SFH",
    "Rijwoning tussen": "TH",
    "Rijwoning hoek": "TH",
    "Appartement": "AB",
    "Woongebouw met niet-zelfstandige woonruimte": "MFH",
}

# ---------------------------------------------------------------------------
# Construction year → TABULA period
# ---------------------------------------------------------------------------

TABULA_PERIODS: list[tuple[int, int, str]] = [
    (0, 1964, "NL.01"),
    (1965, 1974, "NL.02"),
    (1975, 1991, "NL.03"),
    (1992, 2005, "NL.04"),
    (2006, 2014, "NL.05"),
    (2015, 9999, "NL.06"),
]


def classify_period(bouwjaar: int) -> str | None:
    """Map a construction year to a TABULA NL period code."""
    for start, end, label in TABULA_PERIODS:
        if start <= bouwjaar <= end:
            return label
    return None


# ---------------------------------------------------------------------------
# Core matching logic
# ---------------------------------------------------------------------------

def load_tabula_csv(path: str | Path) -> pd.DataFrame:
    """Read the cleaned TABULA NL CSV (24 rows: 4 types × 6 periods)."""
    df = pd.read_csv(path)
    assert len(df) > 0, f"Empty TABULA CSV: {path}"
    return df


def match_tabula(df: pd.DataFrame, tabula_df: pd.DataFrame) -> pd.DataFrame:
    """Match each building to a TABULA archetype by Gebouwtype + bouwjaar.

    Parameters
    ----------
    df : DataFrame with at least columns pand_id, Gebouwtype, bouwjaar.
    tabula_df : TABULA NL lookup table with building_type, period, u_* columns.

    Returns
    -------
    DataFrame with original columns plus tabula_building_type, tabula_period,
    u_wall, u_roof, u_floor, u_window.
    """
    out = df.copy()

    # Map Gebouwtype → TABULA building type
    out["tabula_building_type"] = out["Gebouwtype"].map(GEBOUWTYPE_MAP)
    n_unmapped = out["tabula_building_type"].isna().sum()
    if n_unmapped > 0:
        unmapped = out.loc[out["tabula_building_type"].isna(), "Gebouwtype"].value_counts()
        logger.warning("Unmapped Gebouwtype (%d rows): %s", n_unmapped, unmapped.to_dict())

    # Map bouwjaar → TABULA period
    out["tabula_period"] = out["bouwjaar"].apply(classify_period)

    # Merge with TABULA lookup
    out = out.merge(
        tabula_df[["building_type", "period", "u_wall", "u_roof", "u_floor", "u_window"]],
        left_on=["tabula_building_type", "tabula_period"],
        right_on=["building_type", "period"],
        how="left",
    ).drop(columns=["building_type", "period"])

    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_tabula_match(df: pd.DataFrame) -> dict:
    """Run acceptance checks on TABULA-matched data. Return a report dict."""
    report: dict = {}
    total = len(df)
    report["total_buildings"] = total

    # Match rate
    matched = df["u_wall"].notna().sum()
    match_rate = matched / total * 100
    report["matched"] = int(matched)
    report["unmatched"] = int(total - matched)
    report["match_rate_pct"] = round(match_rate, 2)
    logger.info("Match rate: %d/%d (%.1f%%)", matched, total, match_rate)

    if match_rate < 80:
        logger.warning("Match rate %.1f%% is below 80%% threshold", match_rate)

    # U-value ranges
    for col in ["u_wall", "u_roof", "u_floor", "u_window"]:
        vals = df[col].dropna()
        report[f"{col}_min"] = round(float(vals.min()), 4)
        report[f"{col}_max"] = round(float(vals.max()), 4)
        logger.info("%s range: %.4f - %.4f", col, vals.min(), vals.max())

    # Spot check: sample 5 buildings
    sample = df[df["u_wall"].notna()].sample(n=min(5, matched), random_state=42)
    spot_checks = []
    for _, row in sample.iterrows():
        spot_checks.append({
            "pand_id": row["pand_id"],
            "bouwjaar": int(row["bouwjaar"]),
            "Gebouwtype": row["Gebouwtype"],
            "tabula_type": row["tabula_building_type"],
            "tabula_period": row["tabula_period"],
            "u_wall": round(float(row["u_wall"]), 4),
        })
    report["spot_checks"] = spot_checks
    logger.info("Spot checks:")
    for sc in spot_checks:
        logger.info(
            "  pand=%s  bouwjaar=%d  type=%s→%s  period=%s  u_wall=%.4f",
            sc["pand_id"], sc["bouwjaar"], sc["Gebouwtype"],
            sc["tabula_type"], sc["tabula_period"], sc["u_wall"],
        )

    return report


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_step3(config: dict | None = None) -> dict:
    """Run the full Step 3 TABULA matching pipeline.

    If ``data_paths.lod2_output`` exists on disk, the Step 2 LOD2 features are
    merged with the joined data (the original 3-way Delft path). Otherwise the
    joined parquet is used directly (the 2-way BAG+EP path for new cities).
    """
    if config is None:
        config = load_config()

    paths = config["data_paths"]

    # Load TABULA lookup
    tabula_csv_path = paths["tabula_csv"]
    logger.info("Loading TABULA CSV from %s", tabula_csv_path)
    tabula_df = load_tabula_csv(tabula_csv_path)
    logger.info("TABULA lookup: %d archetypes", len(tabula_df))

    joined_path = paths["joined_output"]
    lod2_path = paths.get("lod2_output")

    if lod2_path and Path(lod2_path).exists():
        logger.info("Loading Step 2 features from %s", lod2_path)
        features_df = pd.read_parquet(lod2_path)
        logger.info("Loaded %d buildings with %d columns",
                    len(features_df), len(features_df.columns))

        logger.info("Loading joined data from %s", joined_path)
        joined_df = pd.read_parquet(
            joined_path, columns=["pand_id", "Gebouwtype", "bouwjaar"]
        )
        df = features_df.merge(joined_df, on="pand_id", how="left")
    else:
        logger.info("No LOD2 output (%s) — using joined data directly", lod2_path)
        df = pd.read_parquet(joined_path)
        logger.info("Loaded %d buildings with %d columns",
                    len(df), len(df.columns))

    # Match TABULA
    logger.info("Matching TABULA archetypes")
    matched = match_tabula(df, tabula_df)

    # Validate
    logger.info("Validating matches")
    report = validate_tabula_match(matched)

    # Drop unmatched rows
    n_before = len(matched)
    matched = matched.dropna(subset=["u_wall"])
    if len(matched) < n_before:
        logger.info("Dropped %d unmatched rows", n_before - len(matched))

    # Save
    output_path = Path(paths["tabula_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matched.to_parquet(output_path, index=False)
    report["output_path"] = str(output_path)
    logger.info("Saved %d matched buildings to %s", len(matched), output_path)

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> "argparse.Namespace":
    import argparse
    parser = argparse.ArgumentParser(description="Step 3 TABULA matching pipeline")
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
        format="%(message)s",
    )
    args = _parse_args()
    cfg = load_config(args.config)
    result = run_step3(cfg)
    print(result)
