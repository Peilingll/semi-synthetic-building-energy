"""Phase C — Delft full-coverage OpenFACADES runner.

Splits Delft into a residential grid, calls OpenFACADES `run.py` per cell with
a two-stage flow: (1) footprint download, (2) BAG-residential spatial filter
(in this venv), (3) Mapillary metadata + AoV + pano + GroundingDINO + quality
filter. Writes per-cell outputs under `data/openfacades_output/phase_c_delft_grid/cells/`,
then merges to `…/merged/`.

Usage:
    python -m src.phase_c_grid_runner --build-grid-only       # generate grid.geojson, exit
    python -m src.phase_c_grid_runner --max-cells 1           # smoke run on 1 residential cell
    python -m src.phase_c_grid_runner                         # full run with resume
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, box

REPO = Path(__file__).resolve().parents[1]

# Defaults
DEFAULT_OUTPUT_ROOT = REPO / "data" / "openfacades_output" / "phase_c_delft_grid"
DEFAULT_BAG_GEOM = REPO / "data" / "processed" / "bag_3dbag_ep_joined.parquet"
DEFAULT_RESID = REPO / "data" / "processed" / "residential_tabula_matched.parquet"
DEFAULT_CONDA_PY = Path("D:/conda_envs/openfacades/python.exe")
DEFAULT_OF_REPO = Path("D:/ITBE/Thesis/OpenFACADES")
DEFAULT_HF_CACHE = Path("D:/hf_cache")

CELL_SIZE_M = 600.0
OVERLAP_M = 50.0
SUBPROCESS_TIMEOUT_S = 3600
MAX_RETRIES_PER_CELL = 2


# ---------- env loading ----------

def load_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env parser: KEY=VALUE per line, strips surrounding quotes, ignores #."""
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        out[k.strip()] = v
    return out


# ---------- patch sanity check ----------

def verify_openfacades_patches(of_repo: Path) -> None:
    """Refuse to run if any of the patches is missing on disk."""
    run_py = (of_repo / "run.py").read_text(encoding="utf-8")
    if "args.stages" not in run_py:
        sys.exit(f"PATCH MISSING: --stages flag absent in {of_repo}/run.py — re-apply per "
                 f"doc_processed/log/openfacades_patches.md (Patch #3)")
    if "Patch #5" not in run_py:
        sys.exit("PATCH MISSING: run.py Patch #5 (empty Mapillary panorama guard)")
    dl = (of_repo / "src/openfacades/footprint/downloader.py").read_text(encoding="utf-8")
    if "tuple(bbox)" not in dl:
        sys.exit("PATCH MISSING: downloader.py Patch #1 (tuple(bbox))")
    if "'osmid', 'id'" not in dl and '"osmid", "id"' not in dl:
        sys.exit("PATCH MISSING: downloader.py Patch #2 ('id' in selected_columns)")


# ---------- grid build ----------

def build_grid(
    bag_geom_path: Path,
    resid_path: Path,
    cell_size_m: float,
    overlap_m: float,
) -> gpd.GeoDataFrame:
    """Build a 600x600m grid (in EPSG:28992) over BAG residential bounds.

    Each cell carries:
      - bbox_inner (no overlap, for reporting)
      - bbox_run   (with overlap, what we pass to run.py --bbox in WGS84)
      - n_residential_pand_ids (BAG centroids in cell)
      - status (residential/empty)
    """
    bag = gpd.read_parquet(bag_geom_path)[["pand_id", "geometry"]]
    resid_ids = set(pd.read_parquet(resid_path)["pand_id"])
    bag_resid = bag[bag["pand_id"].isin(resid_ids)].copy()
    bag_resid_centroids = bag_resid.copy()
    bag_resid_centroids["geometry"] = bag_resid.geometry.centroid

    # Bounds in EPSG:28992 (RD New, meters)
    minx, miny, maxx, maxy = bag_resid.total_bounds
    nx = int((maxx - minx) // cell_size_m) + 1
    ny = int((maxy - miny) // cell_size_m) + 1

    rows = []
    cell_idx = 0
    for j in range(ny):
        for i in range(nx):
            x0 = minx + i * cell_size_m
            y0 = miny + j * cell_size_m
            x1 = x0 + cell_size_m
            y1 = y0 + cell_size_m
            inner = box(x0, y0, x1, y1)
            outer = box(x0 - overlap_m, y0 - overlap_m, x1 + overlap_m, y1 + overlap_m)
            rows.append(
                {
                    "cell_id": f"cell_{cell_idx:04d}",
                    "geometry": outer,
                    "bbox_inner_28992": [x0, y0, x1, y1],
                    "bbox_run_28992": [x0 - overlap_m, y0 - overlap_m, x1 + overlap_m, y1 + overlap_m],
                }
            )
            cell_idx += 1

    grid = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:28992")

    # Count residential centroids per cell
    centroids_in = gpd.sjoin(
        bag_resid_centroids[["pand_id", "geometry"]],
        grid[["cell_id", "geometry"]],
        predicate="within",
        how="inner",
    )
    counts = centroids_in.groupby("cell_id").size().rename("n_residential_pand_ids")
    grid = grid.merge(counts, on="cell_id", how="left").fillna({"n_residential_pand_ids": 0})
    grid["n_residential_pand_ids"] = grid["n_residential_pand_ids"].astype(int)
    grid["status"] = grid["n_residential_pand_ids"].apply(
        lambda n: "residential" if n > 0 else "empty"
    )

    # Convert run bbox to WGS84 for run.py
    grid_wgs = grid.to_crs("EPSG:4326")
    grid["bbox_run_wgs"] = [
        list(g.bounds) for g in grid_wgs.geometry  # (minx, miny, maxx, maxy) in WGS84
    ]

    return grid


def write_grid(grid: gpd.GeoDataFrame, out_path: Path) -> None:
    """Write grid.geojson (in WGS84) with bbox_run as JSON-encoded list-of-floats columns."""
    out = grid.to_crs("EPSG:4326").copy()
    # GeoJSON cannot have list-typed columns; serialize as JSON strings
    for col in ("bbox_inner_28992", "bbox_run_28992", "bbox_run_wgs"):
        out[col] = out[col].apply(json.dumps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(out_path, driver="GeoJSON")


def read_grid(path: Path) -> gpd.GeoDataFrame:
    """Read grid.geojson; bbox cols may come back as ndarray (pyogrio auto-detect)
    or JSON string (older GDAL). Normalise to plain list[float]."""
    g = gpd.read_file(path)
    for col in ("bbox_inner_28992", "bbox_run_28992", "bbox_run_wgs"):
        def _to_list(v):
            if isinstance(v, str):
                return json.loads(v)
            try:
                return [float(x) for x in v]
            except Exception:
                return v
        g[col] = g[col].apply(_to_list)
    return g


# ---------- BAG residential filter ----------

def filter_footprint_to_residential(
    fp_path: Path, bag_residential: gpd.GeoDataFrame
) -> tuple[int, int]:
    """Overwrite fp_path with the residential subset; return (before, after) counts."""
    fp = gpd.read_file(fp_path)
    n_before = len(fp)
    if n_before == 0:
        return 0, 0

    fp_proj = fp.to_crs(bag_residential.crs)
    centroids = fp_proj.copy()
    centroids["geometry"] = fp_proj.geometry.centroid
    joined = gpd.sjoin(
        centroids[["building_id", "geometry"]],
        bag_residential[["pand_id", "geometry"]],
        predicate="within",
        how="inner",
    )
    keep = set(joined["building_id"])
    fp_filt = fp[fp["building_id"].isin(keep)].copy()

    fp_path.with_suffix(".geojson.bak").write_bytes(fp_path.read_bytes())
    fp_filt.to_file(fp_path, driver="GeoJSON")
    return n_before, len(fp_filt)


# ---------- per-cell runner ----------

def run_one_cell(
    cell: dict,
    output_root: Path,
    bag_residential: gpd.GeoDataFrame,
    mapillary_token: str,
    conda_py: Path,
    of_repo: Path,
    hf_cache: Path,
) -> dict:
    cell_id = cell["cell_id"]
    cell_dir = output_root / "cells" / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    log_path = cell_dir / "run.log"

    bbox = cell["bbox_run_wgs"]
    bbox_str = f"[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]"
    base_cmd = [
        str(conda_py),
        str(of_repo / "run.py"),
        f"--bbox={bbox_str}",
        f"--api_key={mapillary_token}",
    ]
    env = {**os.environ, "HF_HOME": str(hf_cache), "MAPILLARY_API_KEY": mapillary_token}
    run_kwargs = dict(cwd=cell_dir, env=env, timeout=SUBPROCESS_TIMEOUT_S)

    result = {"cell_id": cell_id, "started_at": datetime.now(timezone.utc).isoformat()}
    t_start = time.time()

    # --- Stage 1: footprint download ---
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"=== {cell_id} bbox_wgs={bbox_str} ===\n")
        f.flush()
        r1 = subprocess.run(
            base_cmd + ["--stages=1"],
            stdout=f, stderr=subprocess.STDOUT, **run_kwargs,
        )
    result["stage1_rc"] = r1.returncode
    if r1.returncode != 0:
        result["status"] = "failed"
        result["error"] = "stage1 returncode != 0"
        result["duration_s"] = round(time.time() - t_start, 1)
        return result

    fp_path = cell_dir / "output" / "01_data" / "footprint.geojson"
    if not fp_path.exists():
        result["status"] = "failed"
        result["error"] = "footprint.geojson missing after stage 1"
        result["duration_s"] = round(time.time() - t_start, 1)
        return result

    # --- BAG residential filter ---
    n_before, n_after = filter_footprint_to_residential(fp_path, bag_residential)
    result["n_footprint_before_filter"] = n_before
    result["n_footprint_after_filter"] = n_after
    if n_after == 0:
        result["status"] = "skipped_after_filter"
        result["duration_s"] = round(time.time() - t_start, 1)
        return result

    # --- Stage 2-6 ---
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n=== filtered footprint: {n_before} -> {n_after} ===\n")
        f.flush()
        r2 = subprocess.run(
            base_cmd + ["--stages=2-6"],
            stdout=f, stderr=subprocess.STDOUT, **run_kwargs,
        )
    result["stage26_rc"] = r2.returncode

    sel = cell_dir / "output" / "02_img" / "individual_building_select.csv"
    pids_raw = cell_dir / "output" / "01_data" / "pids_urls.csv"
    if r2.returncode == 0 and sel.exists():
        result["status"] = "done"
        try:
            result["n_images_select"] = int(pd.read_csv(sel).shape[0])
        except Exception:
            result["n_images_select"] = -1
    elif r2.returncode == 0 and not pids_raw.exists():
        # Patch #5 path: bbox has no 360 panoramas; OpenFacades exited cleanly.
        # Don't waste retries on this cell.
        result["status"] = "no_pano_coverage"
        result["n_images_select"] = 0
    else:
        result["status"] = "failed"
        result["error"] = f"stage2-6 rc={r2.returncode}, select_exists={sel.exists()}"

    result["duration_s"] = round(time.time() - t_start, 1)
    return result


# ---------- progress + driver ----------

def load_progress(progress_path: Path) -> dict:
    if progress_path.exists():
        return json.loads(progress_path.read_text(encoding="utf-8"))
    return {"started_at": datetime.now(timezone.utc).isoformat(), "cells": {}}


def save_progress(progress_path: Path, prog: dict) -> None:
    progress_path.write_text(json.dumps(prog, indent=2), encoding="utf-8")


def run_all_cells(
    grid: gpd.GeoDataFrame,
    output_root: Path,
    bag_residential: gpd.GeoDataFrame,
    mapillary_token: str,
    conda_py: Path,
    of_repo: Path,
    hf_cache: Path,
    resume: bool,
    max_cells: int | None,
) -> None:
    progress_path = output_root / "progress.json"
    prog = load_progress(progress_path) if resume else {
        "started_at": datetime.now(timezone.utc).isoformat(), "cells": {}
    }

    residential_cells = grid[grid["status"] == "residential"].copy()
    # Sort by residential density desc — high-density first so partial runs are useful
    residential_cells = residential_cells.sort_values(
        "n_residential_pand_ids", ascending=False
    ).reset_index(drop=True)

    if max_cells is not None:
        residential_cells = residential_cells.head(max_cells)

    total = len(residential_cells)
    print(f"[grid] {len(grid)} cells total | {total} residential | "
          f"{len(grid)-total} empty (skipped)")

    for idx, row in residential_cells.iterrows():
        cell_id = row["cell_id"]
        prev = prog["cells"].get(cell_id)
        if resume and prev and prev.get("status") in ("done", "no_pano_coverage", "skipped_after_filter"):
            print(f"[{idx+1}/{total}] {cell_id}  SKIP ({prev.get('status')})")
            continue
        attempts = (prev or {}).get("attempts", 0)
        if attempts >= MAX_RETRIES_PER_CELL:
            print(f"[{idx+1}/{total}] {cell_id}  SKIP (max retries reached)")
            continue

        print(f"[{idx+1}/{total}] {cell_id}  n_resid={row['n_residential_pand_ids']}  running...")
        cell_dict = {
            "cell_id": cell_id,
            "bbox_run_wgs": row["bbox_run_wgs"],
        }
        result = run_one_cell(
            cell_dict, output_root, bag_residential, mapillary_token,
            conda_py, of_repo, hf_cache,
        )
        result["attempts"] = attempts + 1
        prog["cells"][cell_id] = result
        save_progress(progress_path, prog)
        print(f"    -> {result['status']}  duration={result.get('duration_s')}s  "
              f"n_after={result.get('n_footprint_after_filter')}  "
              f"n_images={result.get('n_images_select')}")


# ---------- merge ----------

def merge_outputs(output_root: Path) -> None:
    cells_dir = output_root / "cells"
    merged_dir = output_root / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    fps, aovs, sels = [], [], []
    for cell in sorted(cells_dir.iterdir()):
        out = cell / "output"
        fp = out / "01_data" / "footprint.geojson"
        aov = out / "01_data" / "aov.csv"
        sel = out / "02_img" / "individual_building_select.csv"
        if fp.exists():
            g = gpd.read_file(fp)
            g["__cell"] = cell.name
            fps.append(g)
        if aov.exists():
            a = pd.read_csv(aov); a["__cell"] = cell.name; aovs.append(a)
        if sel.exists():
            s = pd.read_csv(sel); s["__cell"] = cell.name; sels.append(s)

    if fps:
        fp_all = gpd.GeoDataFrame(pd.concat(fps, ignore_index=True), crs=fps[0].crs)
        fp_all = fp_all.drop_duplicates(subset=["building_id"], keep="first")
        fp_all.to_file(merged_dir / "footprint.geojson", driver="GeoJSON")
        print(f"[merge] footprint.geojson: {len(fp_all)} unique buildings")
    if aovs:
        a_all = pd.concat(aovs, ignore_index=True).drop_duplicates(
            subset=["pid", "building_id"], keep="first"
        )
        a_all.to_csv(merged_dir / "aov.csv", index=False)
        print(f"[merge] aov.csv: {len(a_all)} unique (pid, building_id)")
    if sels:
        s_all = pd.concat(sels, ignore_index=True).drop_duplicates(
            subset=["image_name"], keep="first"
        )
        s_all.to_csv(merged_dir / "individual_building_select.csv", index=False)
        print(f"[merge] individual_building_select.csv: {len(s_all)} unique images, "
              f"{s_all['building_id'].nunique()} unique buildings")


# ---------- main ----------

def _apply_city_config(args: argparse.Namespace) -> None:
    """If --city is set, override default-valued paths from configs/<city>.yaml.

    Explicit user-supplied --bag-geom / --residential / --output-root still win.
    """
    if not args.city:
        return
    import yaml
    cfg_path = REPO / "configs" / f"{args.city}.yaml"
    if not cfg_path.exists():
        sys.exit(f"configs/{args.city}.yaml not found")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    paths = cfg.get("data_paths", {})
    if args.bag_geom == DEFAULT_BAG_GEOM and "joined_output" in paths:
        args.bag_geom = REPO / paths["joined_output"]
    if args.residential == DEFAULT_RESID and "tabula_output" in paths:
        args.residential = REPO / paths["tabula_output"]
    if args.output_root == DEFAULT_OUTPUT_ROOT:
        args.output_root = REPO / "data" / "openfacades_output" / f"phase_c_{args.city}_grid"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", type=str, default=None,
                    help="City name (e.g., utrecht). Loads configs/<city>.yaml and "
                         "derives BAG/residential/output paths automatically. "
                         "Output goes to data/openfacades_output/phase_c_<city>_grid/.")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--bag-geom", type=Path, default=DEFAULT_BAG_GEOM)
    ap.add_argument("--residential", type=Path, default=DEFAULT_RESID)
    ap.add_argument("--conda-python", type=Path, default=DEFAULT_CONDA_PY)
    ap.add_argument("--openfacades-repo", type=Path, default=DEFAULT_OF_REPO)
    ap.add_argument("--hf-cache", type=Path, default=DEFAULT_HF_CACHE)
    ap.add_argument("--cell-size", type=float, default=CELL_SIZE_M)
    ap.add_argument("--overlap", type=float, default=OVERLAP_M)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--max-cells", type=int, default=None,
                    help="Limit the number of cells to run (for smoke tests)")
    ap.add_argument("--build-grid-only", action="store_true",
                    help="Build and save grid.geojson, then exit")
    ap.add_argument("--merge-only", action="store_true",
                    help="Skip running cells, just merge existing outputs")
    args = ap.parse_args()

    _apply_city_config(args)

    # Patch sanity
    verify_openfacades_patches(args.openfacades_repo)

    # Token
    env = load_dotenv(REPO / ".env")
    token = env.get("MAPILLARY_API_KEY") or os.environ.get("MAPILLARY_API_KEY")
    if not token and not args.build_grid_only and not args.merge_only:
        sys.exit("MAPILLARY_API_KEY not in .env or environment")

    # Output dir
    args.output_root.mkdir(parents=True, exist_ok=True)
    grid_path = args.output_root / "grid.geojson"

    if args.merge_only:
        merge_outputs(args.output_root)
        return

    # Build / load grid
    if not grid_path.exists():
        print(f"[grid] building from {args.bag_geom} (residential filtered by {args.residential})")
        grid = build_grid(args.bag_geom, args.residential, args.cell_size, args.overlap)
        write_grid(grid, grid_path)
        n_resid = (grid["status"] == "residential").sum()
        print(f"[grid] wrote {grid_path}: {len(grid)} cells, {n_resid} residential")
    else:
        grid = read_grid(grid_path)
        n_resid = (grid["status"] == "residential").sum()
        print(f"[grid] loaded {grid_path}: {len(grid)} cells, {n_resid} residential")

    if args.build_grid_only:
        return

    # Load BAG residential gdf once
    bag_all = gpd.read_parquet(args.bag_geom)[["pand_id", "geometry"]]
    resid_ids = set(pd.read_parquet(args.residential)["pand_id"])
    bag_residential = bag_all[bag_all["pand_id"].isin(resid_ids)].copy()
    print(f"[bag] {len(bag_residential)} residential polygons (CRS: {bag_residential.crs})")

    # Run cells
    run_all_cells(
        grid=grid,
        output_root=args.output_root,
        bag_residential=bag_residential,
        mapillary_token=token,
        conda_py=args.conda_python,
        of_repo=args.openfacades_repo,
        hf_cache=args.hf_cache,
        resume=not args.no_resume,
        max_cells=args.max_cells,
    )

    # Merge
    merge_outputs(args.output_root)


if __name__ == "__main__":
    main()
