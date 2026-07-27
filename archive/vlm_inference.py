"""Step 6: VLM inference — send building images to Gemini for feature extraction."""

import base64
import json
import logging
import time
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types

from src.config import load_config

logger = logging.getLogger(__name__)

PROMPT = """You are an expert in Dutch residential architecture. Analyze this street-view photo of a building and estimate the following properties.

Respond ONLY with a valid JSON object, no other text:

{
  "building_type": "SFH or TH or MFH or AB",
  "building_type_reasoning": "brief explanation",
  "surface_material": "brick or concrete or plaster or wood or mixed or other",
  "construction_year": 1960,
  "construction_year_reasoning": "brief explanation",
  "num_floors": 2,
  "window_to_wall_ratio": 0.25
}

Definitions:
- SFH = Single Family House (detached, vrijstaande woning)
- TH = Terraced House (row house, rijwoning)
- MFH = Multi-Family House (small apartment building, 2-3 units)
- AB = Apartment Block (large apartment building, 4+ units)
- window_to_wall_ratio = estimated fraction of facade area covered by windows (0.0 to 1.0)
- construction_year = your best estimate of when this building was originally built
- num_floors = count of above-ground stories"""


def encode_image(image_path: Path) -> str:
    """Read an image file and return base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def predict_single(client: genai.Client, image_path: Path, model: str = "gemini-2.0-flash") -> dict:
    """Send one image to Gemini and parse the JSON response."""
    image_data = encode_image(image_path)

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=PROMPT),
                    types.Part.from_bytes(data=base64.b64decode(image_data), mime_type="image/png"),
                ],
            )
        ],
    )

    raw = response.text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]

    return json.loads(raw)


def run_vlm_inference(
    image_dir: str | Path,
    ground_truth_csv: str | Path,
    output_csv: str | Path,
    api_key: str,
    model: str = "gemini-2.0-flash",
) -> pd.DataFrame:
    """Run VLM inference on all images and save predictions."""
    image_dir = Path(image_dir)
    gt = pd.read_csv(ground_truth_csv)

    client = genai.Client(api_key=api_key)

    # Resume: load existing predictions and skip already-succeeded ones
    output_path = Path(output_csv)
    existing = {}
    if output_path.exists():
        prev = pd.read_csv(output_path)
        for _, r in prev.iterrows():
            pid = str(r["pand_id"]).zfill(16)
            if "error" not in r or pd.isna(r.get("error")):
                existing[pid] = r.to_dict()
        logger.info("Resuming: %d already completed", len(existing))

    results = list(existing.values())
    for i, row in gt.iterrows():
        pand_id = str(row["pand_id"]).zfill(16)
        img_path = image_dir / f"{pand_id}.png"

        if not img_path.exists():
            logger.warning("Image not found: %s", img_path)
            continue

        if pand_id in existing:
            logger.info("[%d/%d] Skipping %s (already done)", i + 1, len(gt), pand_id)
            continue

        logger.info("[%d/%d] Processing %s", i + 1, len(gt), pand_id)

        # Retry up to 3 times with backoff for rate limits
        for attempt in range(3):
            try:
                pred = predict_single(client, img_path, model=model)
                pred["pand_id"] = pand_id
                results.append(pred)
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    wait = 60 * (attempt + 1)
                    logger.warning("Rate limited, waiting %ds before retry...", wait)
                    time.sleep(wait)
                else:
                    logger.error("Failed for %s: %s", pand_id, e)
                    results.append({"pand_id": pand_id, "error": str(e)})
                    break

        # Rate limit spacing
        time.sleep(5)

    pred_df = pd.DataFrame(results)
    pred_df.to_csv(output_csv, index=False)
    logger.info("Saved %d predictions to %s", len(pred_df), output_csv)
    return pred_df


def evaluate_predictions(pred_csv: str | Path, gt_csv: str | Path) -> dict:
    """Compare VLM predictions against ground truth."""
    pred = pd.read_csv(pred_csv)
    gt = pd.read_csv(gt_csv)

    merged = gt.merge(pred, on="pand_id", how="inner", suffixes=("_gt", "_pred"))
    report = {"total": len(merged)}

    # 1. Building type accuracy
    valid_type = merged["building_type"].notna()
    if valid_type.any():
        type_correct = merged.loc[valid_type, "tabula_building_type"] == merged.loc[valid_type, "building_type"]
        report["type_accuracy"] = round(type_correct.mean(), 4)
        report["type_correct"] = int(type_correct.sum())
        report["type_total"] = int(valid_type.sum())

    # 2. Construction year MAE
    valid_year = merged["construction_year"].notna()
    if valid_year.any():
        year_error = (merged.loc[valid_year, "bouwjaar"] - merged.loc[valid_year, "construction_year"]).abs()
        report["year_mae"] = round(year_error.mean(), 2)
        report["year_median_ae"] = round(year_error.median(), 2)

    # 3. Floor count MAE
    valid_floor = merged["num_floors"].notna() & merged["b3_bouwlagen"].notna()
    if valid_floor.any():
        floor_error = (merged.loc[valid_floor, "b3_bouwlagen"] - merged.loc[valid_floor, "num_floors"]).abs()
        report["floor_mae"] = round(floor_error.mean(), 2)

    # Pass/fail thresholds
    report["pass_type"] = report.get("type_accuracy", 0) > 0.60
    report["pass_year"] = report.get("year_mae", 999) < 20
    report["pass_floor"] = report.get("floor_mae", 999) < 1.5

    return report


if __name__ == "__main__":
    import os
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY environment variable first")
        sys.exit(1)

    config = load_config()
    image_dir = "data/processed/mvp_testing_data"
    gt_csv = "data/processed/vlm_ground_truth.csv"
    output_csv = "data/processed/vlm_predictions.csv"

    print("=== Running VLM inference ===")
    run_vlm_inference(image_dir, gt_csv, output_csv, api_key)

    print("\n=== Evaluating predictions ===")
    report = evaluate_predictions(output_csv, gt_csv)
    for k, v in report.items():
        print(f"  {k}: {v}")
