"""Stage 1 VLM inference runner: original InternVL3-2B + 3-field prompt (v3).

Loads `OpenGVLab/InternVL3-2B` (NOT the OpenFACADES fine-tune, per
spec §3.2.1), iterates over top-3 images of the chosen split, and writes
one per-image row to a parquet file. Saves a partial parquet every
CHECKPOINT_EVERY images so a crash never loses more than that many.

Prompt v3 (2026.06.05 plan): drops construction_period (TABULA boundary
years in the prompt anchored year predictions onto 1975/1991/1992),
drops facade_material (no GT), removes the AB-default instruction
(88% of v2 predictions were AB), adds a TH-vs-AB front-door test.

Usage:
    # prompt iteration on the frozen dev sample (run dev_sample.py first)
    python -m src.stage1.vlm.internvl3_runner --split dev_iter --resume

    # final: full hold-out, ~5300 images, ~7 hr
    python -m src.stage1.vlm.internvl3_runner --split holdout --resume

    # smoke: first 20 images only
    python -m src.stage1.vlm.internvl3_runner --split holdout --sample 20

Use the conda env that already has OpenFACADES sibling deps for `load_image`:
    "D:/conda_envs/openfacades/python.exe" -m src.stage1.vlm.internvl3_runner ...
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from src.stage1.vlm.parse import parse_response

OPENFACADES_SRC = Path("D:/ITBE/Thesis/OpenFACADES/src")
sys.path.insert(0, str(OPENFACADES_SRC))
from openfacades.vlm.base import load_image  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = "OpenGVLab/InternVL3-2B"
CHECKPOINT_EVERY = 100
TOP_K_IMAGES = 3
PROMPT_VERSION = "v3"

PROMPT = """Look at this Dutch residential building street-view photo. Estimate three physical attributes of the main building in view.

Respond with EXACTLY one JSON object, no markdown fences, no extra text. You MUST provide values for ALL fields. NEVER return null.

Required keys:

- "building_type": one of
    "SFH" = single-family house, FREE-STANDING: open space visible on at
            least one side. A building sharing BOTH side walls in a
            continuous row is NEVER SFH.
    "TH"  = terraced / row house: a dwelling in a continuous row of similar
            houses, typically 2-3 storeys, in a residential street
    "AB"  = apartment block: multiple dwellings stacked in one building.
            Cues: a shared entrance with many doorbells/mailboxes, external
            access galleries, a wide flat repetitive facade, OR a narrow
            historic building of 3+ storeys in a dense city-centre street
            (these are usually subdivided into stacked apartments)
    "MFH" = collective housing without self-contained dwellings (student
            dormitory, rooming house, elderly care home)
  If unsure between TH and AB: a 2-3 storey row house in a quiet residential
  street is TH; 4+ storeys, shops at ground level, or a dense city-centre
  street means AB.

- "construction_year": integer 1800-2025. Estimate from visible evidence:
  facade style, brickwork, window shapes, roof form, detailing.

- "num_floors": integer 1-30, visible storeys above ground (count window rows).

Your JSON object must contain ALL three keys, in this order. If uncertain,
give your best single estimate - never omit a key.

Example: {"building_type": "TH", "construction_year": 1932, "num_floors": 3}"""


class InternVLChat:
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
        gen_cfg = {"max_new_tokens": 512, "do_sample": False}
        response, _ = self.model.chat(
            self.tokenizer, pixel_values, question, gen_cfg,
            history=None, return_history=True,
        )
        return response


def load_top3_manifest(manifest_path: Path, split_pand_ids: set[str]) -> pd.DataFrame:
    manifest = pd.read_parquet(manifest_path)
    manifest["pand_id"] = manifest["pand_id"].astype(str)
    todo = manifest[
        manifest["pand_id"].isin(split_pand_ids)
        & (manifest["image_idx"] < TOP_K_IMAGES)
    ].copy().reset_index(drop=True)
    logger.info("top-%d × %d buildings = %d images", TOP_K_IMAGES, len(split_pand_ids), len(todo))
    return todo


def load_done_paths(partial_path: Path) -> tuple[set[str], pd.DataFrame]:
    if not partial_path.exists():
        return set(), pd.DataFrame()
    prev = pd.read_parquet(partial_path)
    done = set(prev["file_path"].astype(str).tolist())
    logger.info("resume: found %d already-processed images in %s", len(done), partial_path.name)
    return done, prev


def inference_one(processor: InternVLChat, file_path: str) -> tuple[str | None, float, str | None]:
    t0 = time.time()
    try:
        raw = processor.process_image(file_path, PROMPT)
    except Exception as exc:
        return None, time.time() - t0, f"inference:{type(exc).__name__}:{exc}"
    return raw, time.time() - t0, None


def run(args: argparse.Namespace) -> None:
    manifest_path = REPO_ROOT / "data" / "processed" / "svi_manifest.parquet"
    if args.split == "holdout":
        split_path = REPO_ROOT / "data" / "processed" / "holdout_test_pand_ids.parquet"
    elif args.split == "dev_iter":
        split_path = REPO_ROOT / "reports" / "stage1" / "vlm" / "dev_prompt_iter_pand_ids.parquet"
        assert split_path.exists(), "run `python -m src.stage1.vlm.dev_sample` first"
    else:
        raise SystemExit(f"unsupported split: {args.split}")

    out_dir = REPO_ROOT / "reports" / "stage1" / "vlm"
    out_dir.mkdir(parents=True, exist_ok=True)
    # versioned tag: v2 (6-field prompt) artifacts are kept as the prompt-
    # ablation baseline and must not be overwritten
    tag = f"internvl3_{PROMPT_VERSION}_{args.split}_per_image"
    if args.sample is not None:
        tag = f"{tag}_sample{args.sample}"
    partial_path = out_dir / f"{tag}.partial.parquet"
    final_path = out_dir / f"{tag}.parquet"

    split_pids = set(pd.read_parquet(split_path)["pand_id"].astype(str))
    todo = load_top3_manifest(manifest_path, split_pids)

    if args.sample is not None:
        todo = todo.head(args.sample)
        logger.info("sample mode: limited to first %d images", len(todo))

    done_paths, existing_df = (
        load_done_paths(partial_path) if args.resume else (set(), pd.DataFrame())
    )
    todo = todo[~todo["file_path"].astype(str).isin(done_paths)].reset_index(drop=True)
    logger.info("after resume filter: %d images to run", len(todo))

    if len(todo) == 0:
        logger.info("nothing to do; finalising partial → %s", final_path.name)
        if not final_path.exists() and partial_path.exists():
            partial_path.rename(final_path)
        return

    logger.info("loading model: %s", args.model)
    t0 = time.time()
    processor = InternVLChat(args.model)
    logger.info("model loaded in %.1fs", time.time() - t0)

    new_rows: list[dict] = []
    n_parse_ok = 0
    for i, row in todo.iterrows():
        file_path = str(row["file_path"])
        raw, dt, err = inference_one(processor, file_path)

        out_row = {
            "pand_id": row["pand_id"],
            "panorama_id": row.get("panorama_id"),
            "file_path": file_path,
            "city": row["city"],
            "image_idx": int(row["image_idx"]),
            "aov_geo": float(row["aov_geo"]) if pd.notna(row.get("aov_geo")) else None,
            "raw_response": raw,
            "inference_sec": round(dt, 2),
            "inference_error": err,
            "parse_ok": False,
            "pred_type": None,
            "pred_year": None,
            "pred_floors": None,
            "parse_error": None,
        }
        if raw is not None:
            parsed = parse_response(raw)
            out_row.update({k: parsed[k] for k in (
                "parse_ok", "pred_type", "pred_year", "pred_floors",
                "parse_error",
            )})
            if parsed["parse_ok"]:
                n_parse_ok += 1

        new_rows.append(out_row)
        logger.info(
            "[%d/%d] %.1fs  ok=%s  type=%s  y=%s  fl=%s",
            i + 1, len(todo), dt, out_row["parse_ok"],
            out_row["pred_type"], out_row["pred_year"], out_row["pred_floors"],
        )

        if (i + 1) % CHECKPOINT_EVERY == 0:
            combined = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
            combined.to_parquet(partial_path, index=False)
            logger.info(
                "checkpoint @ %d/%d  parse_ok_rate=%.3f  avg_sec=%.2f",
                i + 1, len(todo),
                n_parse_ok / max(1, i + 1),
                sum(r["inference_sec"] for r in new_rows) / max(1, len(new_rows)),
            )

    combined = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
    combined.to_parquet(final_path, index=False)
    if partial_path.exists():
        partial_path.unlink()
    logger.info(
        "DONE: wrote %s (%d rows, parse_ok %d / %d = %.3f)",
        final_path, len(combined),
        combined["parse_ok"].sum(), len(combined),
        combined["parse_ok"].mean(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="holdout", choices=["holdout", "dev_iter"])
    parser.add_argument("--sample", type=int, default=None, help="run first N images only")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--resume", action="store_true",
                        help="skip images already in partial parquet")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
