"""Phase B image sanity check.

Randomly samples N images from svi_manifest.parquet, verifies each opens with
PIL and has both dimensions >= MIN_DIM. Catches corrupt or empty files before
they crash a multi-hour training run.
"""

import argparse
import logging
import random
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

MIN_DIM = 128  # PIL RandomResizedCrop(224) can up-sample smaller images;
               # this threshold catches obviously broken files only
DINOV2_INPUT = 224  # buckets for distribution reporting


def sanity_check(manifest_path: Path, n: int = 200, seed: int = 42) -> dict:
    m = pd.read_parquet(manifest_path)
    paths = m["file_path"].tolist()
    rng = random.Random(seed)
    sample = rng.sample(paths, min(n, len(paths)))

    bad_read = []
    too_small = []
    sizes = []
    for p in sample:
        try:
            with Image.open(p) as img:
                img.verify()
            with Image.open(p) as img:
                w, h = img.size
                _ = img.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as e:
            bad_read.append((p, str(e)[:80]))
            continue
        sizes.append((h, w))
        if h < MIN_DIM or w < MIN_DIM:
            too_small.append((p, h, w))

    report = {
        "n_sampled": len(sample),
        "n_total_manifest": len(paths),
        "n_bad_read": len(bad_read),
        "n_too_small": len(too_small),
        "bad_read": [{"path": p, "err": e} for p, e in bad_read[:10]],
        "too_small": [{"path": p, "h": h, "w": w} for p, h, w in too_small[:10]],
    }

    if sizes:
        hs = [h for h, _ in sizes]
        ws = [w for _, w in sizes]
        min_dims = [min(h, w) for h, w in sizes]
        report["height_min"] = min(hs)
        report["height_max"] = max(hs)
        report["width_min"] = min(ws)
        report["width_max"] = max(ws)
        report["pct_below_224"] = round(100 * sum(1 for d in min_dims if d < 224) / len(min_dims), 1)
        report["pct_below_128"] = round(100 * sum(1 for d in min_dims if d < 128) / len(min_dims), 1)

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    repo_root = Path(__file__).resolve().parents[2]
    manifest = Path(args.manifest) if args.manifest else repo_root / "data" / "processed" / "svi_manifest.parquet"
    report = sanity_check(manifest, args.n, args.seed)

    logger.info("sanity sample: %d / %d total manifest images",
                report["n_sampled"], report["n_total_manifest"])
    logger.info("bad_read: %d, too_small: %d", report["n_bad_read"], report["n_too_small"])
    if "height_min" in report:
        logger.info("height range: [%d, %d], width range: [%d, %d]",
                    report["height_min"], report["height_max"],
                    report["width_min"], report["width_max"])
        logger.info("pct with min(h,w) < 224 (DINOv2 input): %.1f%%", report["pct_below_224"])
        logger.info("pct with min(h,w) < 128 (broken threshold): %.1f%%", report["pct_below_128"])

    if report["n_bad_read"] > 0:
        logger.error("BROKEN FILES (cannot proceed): %s", report["bad_read"])
        raise SystemExit(1)
    if report["n_too_small"] > 0:
        logger.warning("%d / %d images have min(h,w) < %d (will be heavily upscaled by aug). "
                       "Examples: %s", report["n_too_small"], report["n_sampled"], MIN_DIM,
                       report["too_small"][:3])

    logger.info("OK: %d sampled images all PIL-readable, no broken files", report["n_sampled"])


if __name__ == "__main__":
    main()
