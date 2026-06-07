"""Draw the fixed dev-set sample used for VLM prompt iteration.

Prompt v2 was tuned on 20 hold-out buildings (mild test leakage). v3 prompt
iteration instead uses this frozen sample of 200 dev buildings; the hold-out
is only run once after the prompt is final.

Stratification: fixed per-type quotas that oversample the minority classes
(AB 80 / TH 60 / SFH 40 / MFH 20) — a proportional draw would be ~89% AB and
could not detect whether the AB-default bias is resolved. Within each type,
buildings are allocated across cities proportionally to availability.

Usage:
    python -m src.stage1.vlm.dev_sample
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEV_FOLDS_PATH = REPO_ROOT / "data" / "processed" / "dev_fold_indices.parquet"
OUT_PATH = REPO_ROOT / "reports" / "stage1" / "vlm_internvl3" / "dev_prompt_iter_pand_ids.parquet"

SEED = 42
TYPE_QUOTAS = {"AB": 80, "TH": 60, "SFH": 40, "MFH": 20}


def draw_sample(dev: pd.DataFrame) -> pd.DataFrame:
    picked: list[pd.DataFrame] = []
    for btype, quota in TYPE_QUOTAS.items():
        pool = dev[dev["building_type"] == btype]
        quota = min(quota, len(pool))
        # proportional allocation across cities (largest remainder)
        counts = pool["city"].value_counts()
        exact = counts / counts.sum() * quota
        alloc = exact.astype(int)
        remainder = quota - alloc.sum()
        for city in (exact - alloc).sort_values(ascending=False).index[:remainder]:
            alloc[city] += 1
        for city, n in alloc.items():
            if n == 0:
                continue
            grp = pool[pool["city"] == city]
            picked.append(grp.sample(n=min(n, len(grp)), random_state=SEED))
        logger.info("%s: quota %d from pool %d (%s)", btype, quota, len(pool),
                    ", ".join(f"{c}={n}" for c, n in alloc.items() if n))
    out = pd.concat(picked, ignore_index=True)
    return out[["pand_id", "city", "building_type", "tabula_period"]].sort_values(
        ["building_type", "city", "pand_id"]).reset_index(drop=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dev = pd.read_parquet(DEV_FOLDS_PATH)
    dev["pand_id"] = dev["pand_id"].astype(str)
    sample = draw_sample(dev)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(OUT_PATH, index=False)
    logger.info("wrote %s (%d buildings)", OUT_PATH, len(sample))
    print(pd.crosstab(sample["city"], sample["building_type"], margins=True))


if __name__ == "__main__":
    main()
