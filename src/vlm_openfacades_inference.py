"""Phase 2 — OpenFACADES InternVL3-2B FT inference on Phase C cropped building images.

Off-the-shelf transfer baseline from doc §11 Step 2. Loads
`seshing/openfacades-internvl3-2b` (already cached at `D:/hf_cache/`) via
`InternVLProcessor` from the OpenFACADES sibling repo and runs the multi-attribute
JSON prompt extracted from the training jsonl on every image listed in
`merged/individual_building_select.csv`.

**Run with the conda env Python**, not the main `.venv`:
    HF_HOME=D:/hf_cache "D:/conda_envs/openfacades/python.exe" -m src.vlm_openfacades_inference [--max-images N]

Output: `data/processed/vlm_predictions_openfacades.csv` — one row per image, raw
OpenFACADES schema (8 fields). Per-pand_id aggregation is a separate step.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

# OpenFACADES sibling repo provides `load_image` (dynamic patch preprocessing).
# We deliberately avoid `InternVLProcessor` itself: its `device_map='auto'` path
# breaks with current transformers (`AttributeError: all_tied_weights_keys`),
# and we only need a single-GPU loader anyway.
OPENFACADES_SRC = Path("D:/ITBE/Thesis/OpenFACADES/src")
sys.path.insert(0, str(OPENFACADES_SRC))
from openfacades.vlm.base import load_image  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE_C_ROOT = REPO_ROOT / "data" / "openfacades_output" / "phase_c_delft_grid"
DEFAULT_INPUT_CSV = PHASE_C_ROOT / "merged" / "individual_building_select.csv"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "vlm_predictions_openfacades.csv"
DEFAULT_MODEL = "seshing/openfacades-internvl3-2b"

# Verbatim multi-attribute prompt from
# D:/ITBE/Thesis/OpenFACADES/train/data/jsonl/train.jsonl (most-frequent training
# prompt, ~19,443/58,942 examples). DO NOT modify — fine-tune was conditioned on
# this exact format including indentation. Leading "<image>\n" is added by InternVL
# chat at runtime, so we omit it here.
PROMPT = """ Provide concise labels for each category using the following JSON format. Select appropriate values from the provided options for each category:
				        {
				    "building_type": "(choose one option from: 'apartments', 'house', 'retail', 'office', 'hotel', 'industrial', 'religious', 'education', 'public', 'garage')",
				    "alternate_building_type": "(choose another option from: 'apartments', 'house', 'retail', 'office', 'hotel', 'industrial', 'religious', 'education', 'public', 'garage')",
				    "building_age": "(a 4-digit year indicating the approximate construction date of the building)",
				    "floors": "(a numeric value representing the total number of floors)",
				    "surface_material": "(choose one option from: 'brick', 'wood', 'concrete', 'metal', 'stone', 'glass', 'plaster')",
				    "alternate_surface_material": "(choose another option from: 'brick', 'wood', 'concrete', 'metal', 'stone', 'glass', 'plaster')",
				    "construction_material": "(choose one option from: 'brick', 'wood', 'concrete', 'steel', 'other')",
				    "alternate_construction_material": "(choose another option from: 'brick', 'wood', 'concrete', 'steel', 'other')"
				}
                """

OF_FIELDS = [
    "building_type",
    "alternate_building_type",
    "building_age",
    "floors",
    "surface_material",
    "alternate_surface_material",
    "construction_material",
    "alternate_construction_material",
]


def parse_response(raw: str) -> dict:
    """Extract OpenFACADES JSON from the raw model response.

    Training response is wrapped in ```json ... ``` markdown fences, but be
    permissive: also accept bare JSON or trailing text.
    """
    if raw is None:
        return {"_parse_error": "empty response"}

    text = raw.strip()
    # Strip ```json ... ``` fences if present
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        # Otherwise grab the first {...} block
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            text = m.group(0)

    # Fine-tune quirk: model occasionally emits an empty key line `,\n,\n`
    # between fields (typically after `"floors": N,`). Collapse `,\s*,` → `,`.
    text = re.sub(r",\s*,", ",", text)
    # Also a trailing comma before } / ] is invalid JSON
    text = re.sub(r",(\s*[}\]])", r"\1", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {"_parse_error": f"json decode: {e}"}


class InternVLChatLite:
    """Minimal single-GPU loader for InternVL chat model.

    Replaces `openfacades.vlm.InternVLProcessor` to avoid a transformers
    incompatibility (`device_map='auto'` triggers `infer_auto_device_map`
    which expects `model.all_tied_weights_keys` not present in InternVL custom
    modeling code). We just `.cuda()` after load.
    """

    def __init__(self, model_id: str, max_num_patches: int = 12) -> None:
        self.max_num_patches = max_num_patches
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False,
        )

    def process_image(self, image_path: str, question: str) -> str | None:
        pixel_values = load_image(image_path, max_num=self.max_num_patches)
        if pixel_values is None:
            return None
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        gen_cfg = {"max_new_tokens": 1024, "do_sample": False}
        response, _ = self.model.chat(
            self.tokenizer, pixel_values, question, gen_cfg,
            history=None, return_history=True,
        )
        return response


def resolve_image_path(row: pd.Series) -> Path:
    """Phase C cells write images under cells/<__cell>/output/02_img/individual_building/<image_name>."""
    return (
        PHASE_C_ROOT
        / "cells"
        / row["__cell"]
        / "output"
        / "02_img"
        / "individual_building"
        / row["image_name"]
    )


def run_inference(
    input_csv: Path,
    output_csv: Path,
    model_id: str,
    max_images: int | None,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    logger.info("Loaded %d rows from %s", len(df), input_csv)

    if max_images is not None:
        df = df.head(max_images)
        logger.info("Smoke mode: limiting to %d images", len(df))

    # Resume: skip image_names already in output (with no error)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    done_names: set[str] = set()
    existing_rows: list[dict] = []
    if output_csv.exists():
        prev = pd.read_csv(output_csv)
        for _, r in prev.iterrows():
            existing_rows.append(r.to_dict())
            if pd.isna(r.get("error")) or r.get("error") == "":
                done_names.add(str(r["image_name"]))
        logger.info("Resume: %d already done, %d total prior rows", len(done_names), len(prev))

    # Lazy load model (skip if everything is already done in resume)
    todo = df[~df["image_name"].isin(done_names)].reset_index(drop=True)
    if len(todo) == 0:
        logger.info("Nothing to do — all images already processed.")
        return pd.DataFrame(existing_rows)

    logger.info("Loading model: %s", model_id)
    t0 = time.time()
    processor = InternVLChatLite(model_id)
    logger.info("Model loaded in %.1fs", time.time() - t0)

    new_rows: list[dict] = []
    for i, row in todo.iterrows():
        img_path = resolve_image_path(row)
        out_row = {
            "image_name": row["image_name"],
            "pid": row.get("pid"),
            "building_id": row.get("building_id"),
            "__cell": row.get("__cell"),
            **{f: None for f in OF_FIELDS},
            "raw_response": None,
            "error": None,
        }

        if not img_path.exists():
            out_row["error"] = f"image not found: {img_path}"
            logger.warning("[%d/%d] %s — image not found", i + 1, len(todo), row["image_name"])
            new_rows.append(out_row)
            continue

        t_img = time.time()
        try:
            raw = processor.process_image(str(img_path), PROMPT)
        except Exception as exc:  # noqa: BLE001 — catch-all so one bad image doesn't kill the run
            out_row["error"] = f"inference: {exc}"
            logger.error("[%d/%d] %s — inference failed: %s", i + 1, len(todo), row["image_name"], exc)
            new_rows.append(out_row)
            continue

        out_row["raw_response"] = raw
        parsed = parse_response(raw)
        if "_parse_error" in parsed:
            out_row["error"] = parsed["_parse_error"]
        else:
            for f in OF_FIELDS:
                if f in parsed:
                    out_row[f] = parsed[f]

        dt = time.time() - t_img
        logger.info(
            "[%d/%d] %s  %.1fs  type=%s  age=%s  floors=%s  material=%s",
            i + 1, len(todo), row["image_name"], dt,
            out_row.get("building_type"), out_row.get("building_age"),
            out_row.get("floors"), out_row.get("surface_material"),
        )
        new_rows.append(out_row)

        # Periodic save every 50 images so a crash mid-run doesn't lose progress
        if (i + 1) % 50 == 0:
            partial = pd.DataFrame(existing_rows + new_rows)
            partial.to_csv(output_csv, index=False)
            logger.info("Checkpoint saved at %d/%d", i + 1, len(todo))

    final = pd.DataFrame(existing_rows + new_rows)
    final.to_csv(output_csv, index=False)
    logger.info("Saved %d rows to %s", len(final), output_csv)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit to first N images (for smoke testing)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    run_inference(args.input_csv, args.output_csv, args.model, args.max_images)


if __name__ == "__main__":
    main()
