"""M2-VLM: zero-shot InternVL3-2B asked DIRECTLY for the A-G energy label,
bypassing type/year/floor and TABULA (the end-to-end analogue for the VLM).

Reuses the Stage 1 InternVLChat + top-3 manifest machinery; only the prompt and
the parser change (single key "energy_label"). Per-image rows; aggregation to
per-pand_id (majority vote) is done in m2_vlm_eval.py.

ENV: conda openfacades (same as the Stage 1 VLM runner — needs load_image).
    "D:/conda_envs/openfacades/python.exe" -m src.stage3.m2_vlm_runner --split holdout --resume
    "D:/conda_envs/openfacades/python.exe" -m src.stage3.m2_vlm_runner --split holdout --sample 20  # smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import pandas as pd

from src.stage1.vlm.internvl3_runner import (
    DEFAULT_MODEL,
    CHECKPOINT_EVERY,
    InternVLChat,
    REPO_ROOT,
    load_top3_manifest,
)

logger = logging.getLogger(__name__)

ENERGY_WHITELIST = {"A", "B", "C", "D", "E", "F", "G"}

PROMPT = """Look at this Dutch residential building street-view photo. Estimate the building's official Dutch energy label (energielabel, NTA 8800), which rates a home's primary fossil energy use from A (most energy-efficient) to G (least energy-efficient).

Judge ONLY from visible evidence in the photo:
- window glazing: single-pane (older, less efficient) vs double / HR++ glazing
- solar panels on the roof
- signs of recent renovation or added wall insulation (new render, cladding, modern detailing)
- general age and upkeep of the facade and roof
Buildings that look newly built or recently renovated, with modern glazing or solar panels, tend toward A-B. Old, un-renovated buildings with single glazing and worn facades tend toward F-G. Most ordinary housing falls in the middle.

Respond with EXACTLY one JSON object, no markdown fences, no extra text. You MUST provide a value. NEVER return null.

Required key:
- "energy_label": exactly one of "A", "B", "C", "D", "E", "F", "G"

Example: {"energy_label": "C"}"""


def parse_energy(raw: str | None) -> dict:
    out = {"parse_ok": False, "pred_label": None, "parse_error": None}
    if raw is None:
        out["parse_error"] = "no_response"
        return out
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    block = m.group(0) if m else None
    if block is None:
        out["parse_error"] = "no_json_block"
        return out
    block = re.sub(r",(\s*[}\]])", r"\1", block)
    try:
        obj = json.loads(block)
    except json.JSONDecodeError as exc:
        out["parse_error"] = f"json_decode:{exc.msg}"
        return out
    val = obj.get("energy_label")
    if isinstance(val, str) and val.strip().upper() in ENERGY_WHITELIST:
        out["pred_label"] = val.strip().upper()
        out["parse_ok"] = True
    else:
        out["parse_error"] = f"invalid_label:{val!r}"
    return out


def run(args: argparse.Namespace) -> None:
    manifest_path = REPO_ROOT / "data" / "processed" / "svi_manifest.parquet"
    split_path = REPO_ROOT / "data" / "processed" / "holdout_test_pand_ids.parquet"
    out_dir = REPO_ROOT / "reports" / "stage3"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"m2vlm_{args.split}_per_image"
    if args.sample is not None:
        tag = f"{tag}_sample{args.sample}"
    partial_path = out_dir / f"{tag}.partial.parquet"
    final_path = out_dir / f"{tag}.parquet"

    split_pids = set(pd.read_parquet(split_path)["pand_id"].astype(str))
    todo = load_top3_manifest(manifest_path, split_pids)
    if args.sample is not None:
        todo = todo.head(args.sample)
        logger.info("sample mode: first %d images", len(todo))

    existing_df = pd.DataFrame()
    if args.resume and partial_path.exists():
        existing_df = pd.read_parquet(partial_path)
        done = set(existing_df["file_path"].astype(str))
        todo = todo[~todo["file_path"].astype(str).isin(done)].reset_index(drop=True)
        logger.info("resume: %d done, %d to run", len(done), len(todo))

    if len(todo) == 0:
        logger.info("nothing to do; finalising → %s", final_path.name)
        if not final_path.exists() and partial_path.exists():
            partial_path.rename(final_path)
        return

    logger.info("loading model: %s", args.model)
    t0 = time.time()
    proc = InternVLChat(args.model)
    logger.info("model loaded in %.1fs", time.time() - t0)

    new_rows, n_ok = [], 0
    for i, row in todo.iterrows():
        fp = str(row["file_path"])
        t1 = time.time()
        try:
            raw = proc.process_image(fp, PROMPT)
            err = None
        except Exception as exc:  # noqa: BLE001
            raw, err = None, f"inference:{type(exc).__name__}:{exc}"
        dt = time.time() - t1
        p = parse_energy(raw)
        if p["parse_ok"]:
            n_ok += 1
        new_rows.append({
            "pand_id": row["pand_id"], "file_path": fp, "city": row["city"],
            "image_idx": int(row["image_idx"]), "raw_response": raw,
            "inference_sec": round(dt, 2), "inference_error": err,
            "parse_ok": p["parse_ok"], "pred_label": p["pred_label"],
            "parse_error": p["parse_error"],
        })
        logger.info("[%d/%d] %.1fs ok=%s label=%s", i + 1, len(todo), dt,
                    p["parse_ok"], p["pred_label"])
        if (i + 1) % CHECKPOINT_EVERY == 0:
            pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True).to_parquet(partial_path, index=False)
            logger.info("checkpoint @ %d/%d ok_rate=%.3f", i + 1, len(todo), n_ok / (i + 1))

    combined = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
    combined.to_parquet(final_path, index=False)
    if partial_path.exists():
        partial_path.unlink()
    logger.info("DONE: %s (%d rows, parse_ok %.3f)", final_path.name, len(combined),
                combined["parse_ok"].mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="holdout", choices=["holdout"])
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
