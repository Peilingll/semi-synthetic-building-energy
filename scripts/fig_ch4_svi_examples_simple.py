"""Fig. 4.x (simplified) -- six holdout examples: image | Reference | DINOv2 | ResNet-50 | InternVL3.

Reads the same six buildings as Thesis_reports/fig_ch4_svi_examples.png from
Thesis_reports/fig_ch4_svi_examples_candidates/candidates.csv and the first
retained crop of each building from data/processed/svi_manifest.parquet.

Marks per attribute (three rows per building):
  type   correct if predicted size class == reference size class
  year   correct if predicted year falls in the reference TABULA-NL period
  floors correct if round(predicted floors) == reference floor count

Outputs (not yet wired into the LaTeX):
  Thesis_reports/fig_ch4_svi_examples_simple.html / .png

Run:  .venv/Scripts/python.exe scripts/fig_ch4_svi_examples_simple.py
"""
from __future__ import annotations

import base64
import io
import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image, ImageChops

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "Thesis_reports"
CAND = OUT / "fig_ch4_svi_examples_candidates" / "candidates.csv"
MANIFEST = REPO / "data/processed/svi_manifest.parquet"
CHROME = Path(r"C:/Program Files/Google/Chrome/Application/chrome.exe")

GREEN, ORANGE, INK, GREY = "#A2AD00", "#E37222", "#1F2A37", "#6B7280"

# (pand_id, description) in display order -- the same six buildings as the current figure
EXAMPLES = [
    ("0363100012096905", "Clear view"),
    ("0363100012091355", "Partial occlusion"),
    ("0363100012135983", "Clear view"),
    ("0344100000060111", "Low quality"),
    ("0363100012238347", "Distortion view"),
    ("0363100012236172", "Full occlusion"),
]
MODELS = [("DINOv2", "DINOv2"), ("ResNet", "ResNet-50"), ("VLM", "InternVL3")]
IMG_H = 300  # px, image box height


def b64_image(path: Path, max_h: int = IMG_H, max_w: int = 400) -> tuple[str, int, int]:
    im = Image.open(path).convert("RGB")
    scale = min(max_h / im.height, max_w / im.width)
    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode(), im.width, im.height


def marks(row: pd.Series, key: str) -> list[bool]:
    return [
        row[f"{key}_type"] == row["true_type"],
        row[f"{key}_period"] == row["true_period"],
        round(float(row[f"{key}_floors"])) == int(row["true_num_floors"]),
    ]


def mark_html(ok: bool) -> str:
    return ('<span class="ok">&#10003;</span>' if ok else '<span class="bad">&#215;</span>')


def build_html() -> str:
    cand = pd.read_csv(CAND, dtype={"pand_id": str}).set_index("pand_id")
    man = pd.read_parquet(MANIFEST).sort_values(["pand_id", "image_idx"])
    first = man.groupby("pand_id").first()
    rows = []
    for pid, desc in EXAMPLES:
        r = cand.loc[pid]
        b64, w, h = b64_image(Path(first.loc[pid, "file_path"]))
        ref = (f'<div class="ref"><span class="lab">Type</span>{r["true_type"]}</div>'
               f'<div class="ref"><span class="lab">Year</span>{int(r["true_bouwjaar"])} ({r["true_period"]})</div>'
               f'<div class="ref"><span class="lab">Floors</span>{int(r["true_num_floors"])}</div>')
        cells = ""
        for key, _ in MODELS:
            m = marks(r, key)
            cells += '<td class="m">' + "".join(f'<div class="mk">{mark_html(x)}</div>' for x in m) + "</td>"
        rows.append(
            f'<tr><td class="img"><img src="data:image/jpeg;base64,{b64}" width="{w}" height="{h}">'
            f'<div class="desc">{desc}</div></td><td class="refcol">{ref}</td>{cells}</tr>')
    head = ("<tr><th>Image</th><th>Reference</th>" + "".join(f"<th>{n}</th>" for _, n in MODELS) + "</tr>")
    css = f"""
      body {{ margin: 0; background: #fff; font-family: Arial, Helvetica, sans-serif; color: {INK}; }}
      table {{ border-collapse: collapse; width: 1560px; }}
      th {{ font-size: 34px; font-weight: 700; text-align: center; padding: 14px 18px 16px;
            border-bottom: 3px solid {INK}; border-left: 2px solid #C9CDD3; }}
      th:first-child, td:first-child {{ border-left: none; }}
      td {{ vertical-align: middle; padding: 22px 18px; border-bottom: 2px solid #C9CDD3;
            border-left: 2px solid #C9CDD3; font-size: 30px; }}
      tr:last-child td {{ border-bottom: 3px solid {INK}; }}
      td.img {{ width: 440px; text-align: center; }}
      td.img img {{ display: block; margin: 0 auto; border-radius: 4px; }}
      .desc {{ font-size: 30px; color: {INK}; margin-top: 12px; line-height: 1.25; text-align: center; }}
      td.refcol {{ width: 400px; }}
      .ref {{ line-height: 64px; white-space: nowrap; }}
      .lab {{ display: inline-block; width: 120px; color: {GREY}; }}
      td.m {{ text-align: center; width: 180px; }}
      .mk {{ line-height: 64px; font-size: 36px; font-family: "Cambria Math", "Times New Roman", serif; color: {INK}; }}
      .ok {{ }} .bad {{ font-size: 40px; }}
    """
    return (f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>SVI examples</title>'
            f'<style>{css}</style></head><body><table>{head}{"".join(rows)}</table></body></html>')


def render_png(html_path: Path, png_path: Path, width: int, height: int, scale: int = 2):
    tmp = png_path.with_name(png_path.stem + "_raw.png")
    cmd = [str(CHROME), "--headless=new", "--disable-gpu", "--hide-scrollbars",
           f"--force-device-scale-factor={scale}", f"--window-size={width},{height}",
           f"--screenshot={tmp}", html_path.resolve().as_uri()]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    im = Image.open(tmp).convert("RGB")
    pad = 16 * scale
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bbox = ImageChops.difference(im, bg).getbbox()
    if bbox:
        im = im.crop((max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                      min(im.width, bbox[2] + pad), min(im.height, bbox[3] + pad)))
    im.save(png_path)
    tmp.unlink()


def main():
    html = build_html()
    hp = OUT / "fig_ch4_svi_examples_simple.html"
    hp.write_text(html, encoding="utf-8")
    render_png(hp, OUT / "fig_ch4_svi_examples_simple.png", 1620, 2700)
    print("ok")


if __name__ == "__main__":
    main()
