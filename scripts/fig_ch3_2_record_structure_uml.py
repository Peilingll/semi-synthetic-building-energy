"""Fig. 3.2 (candidate) -- Stage 1 record structure as a simplified UML class diagram.

Two classes joined by a one-to-many association on the BAG building identifier:
  BuildingReferenceRecord <- data/processed/stage1_gt.parquet    (one row per BAG building)
  StreetViewImageRecord   <- data/processed/svi_manifest.parquet (one row per retained crop)

Attribute names are the real column names of the two parquet files; the
multiplicity and counts are computed from the files at run time.

Outputs (Thesis_reports/, not yet wired into the LaTeX):
  fig_3_2_record_structure_uml.svg / .html / .png

Run:  .venv/Scripts/python.exe scripts/fig_ch3_2_record_structure_uml.py
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image, ImageChops

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "Thesis_reports"
CHROME = Path(r"C:/Program Files/Google/Chrome/Application/chrome.exe")

# TUM palette
BLUE, DBLUE, LBLUE, MBLUE, GREY, GREEN, ORANGE = (
    "#0065BD", "#005293", "#98C6EA", "#64A0C8", "#DAD7CB", "#A2AD00", "#E37222")
INK = "#1F2A37"

REF_ATTRS = [  # (column, type, gloss, tag)
    ("pand_id", "str", "BAG building identifier", "PK"),
    ("city", "str", "CITY_GLOSS", ""),
    ("bouwjaar", "int", "construction year (BAG)", ""),
    ("Gebouwtype", "str", "residential category (EP-Online, Dutch)", ""),
    ("building_type", "str", "size class SFH / TH / MFH / AB", ""),
    ("num_floors", "int", "floor count (3DBAG)", ""),
    ("Energieklasse", "str", "registered energy class (EP-Online)", ""),
]
IMG_ATTRS = [
    ("pand_id", "str", "BAG building identifier", "FK"),
    ("panorama_id", "str", "Mapillary panorama", ""),
    ("bdid", "str", "OpenFACADES footprint id (OSM / Overture)", ""),
    ("file_path", "str", "cropped building image", ""),
    ("city", "str", "city of the panorama", ""),
    ("aov_geo", "float", "angle of view to the building (degrees)", ""),
    ("distance", "float", "camera-to-building distance (m)", ""),
    ("image_idx", "int", "crop index within the building", ""),
]


def stats():
    g = pd.read_parquet(REPO / "data/processed/stage1_gt.parquet", columns=["pand_id", "city"])
    m = pd.read_parquet(REPO / "data/processed/svi_manifest.parquet", columns=["pand_id", "image_idx"])
    per = m.groupby("pand_id").size()
    return dict(n_ref=len(g), n_cities=g["city"].nunique(), cities=", ".join(sorted(c.title() for c in g["city"].unique())),
                n_img=len(m), n_bld=per.size,
                lo=int(per.min()), hi=int(per.max()), med=int(per.median()))


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def klass(x, y, w, name, stereotype, attrs, header_fill, body_fill, h=None):
    hh, rh, pad = 124, 94, 24  # two text lines per attribute row
    h = h or hh + pad + rh * len(attrs) + pad
    s = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{body_fill}" stroke="{DBLUE}" stroke-width="3"/>',
         f'<rect x="{x}" y="{y}" width="{w}" height="{hh}" rx="6" fill="{header_fill}"/>',
         f'<rect x="{x}" y="{y + hh - 8}" width="{w}" height="8" fill="{header_fill}"/>',
         f'<text x="{x + w / 2}" y="{y + 48}" class="stereo" text-anchor="middle">{esc(stereotype)}</text>',
         f'<text x="{x + w / 2}" y="{y + 100}" class="cname" text-anchor="middle">{esc(name)}</text>',
         f'<line x1="{x}" y1="{y + hh}" x2="{x + w}" y2="{y + hh}" stroke="{DBLUE}" stroke-width="3"/>']
    ty = y + hh + pad + 40
    for col, typ, gloss, tag in attrs:
        tagtxt = "  {" + tag + "}" if tag else ""
        s.append(f'<text x="{x + 26}" y="{ty}" class="attr"><tspan class="mono">{esc(col)}</tspan>'
                 f'<tspan class="typ"> : {esc(typ)}</tspan><tspan class="tag">{esc(tagtxt)}</tspan></text>')
        s.append(f'<text x="{x + 56}" y="{ty + 40}" class="gloss">{esc(gloss)}</text>')
        ty += rh
    return "\n".join(s), h


def note(x, y, w, lines, h=None):
    lh, pad = 42, 26
    h = h or pad * 2 + lh * len(lines)
    fold = 22
    path = f'M{x},{y} H{x + w - fold} L{x + w},{y + fold} V{y + h} H{x} Z'
    s = [f'<path d="{path}" fill="#F4F3EF" stroke="#9CA3AF" stroke-width="2.5"/>',
         f'<path d="M{x + w - fold},{y} V{y + fold} H{x + w}" fill="none" stroke="#9CA3AF" stroke-width="2.5"/>']
    ty = y + pad + 32
    for i, ln in enumerate(lines):
        cls = "notehead" if i == 0 else "note"
        s.append(f'<text x="{x + 20}" y="{ty}" class="{cls}">{esc(ln)}</text>')
        ty += lh
    return "\n".join(s), h


def build_svg(st):
    W = 1730
    cw = 720
    lx, rx = 70, W - 70 - cw
    ny = 60
    lines2 = ["Sources and processing", "OpenStreetMap and Overture Maps",
              "(candidate footprints)",
              "Mapillary (panoramas)",
              "OpenFACADES (image processing)",
              "(Liang et al., 2025)"]
    nh = 26 * 2 + 42 * len(lines2)
    n1, nh1 = note(lx, ny, cw, ["Sources", "BAG (attributes and footprints)",
                                 "3DBAG (reconstructed geometry)",
                                 "EP-Online (building category, energy class)"], h=nh)
    n2, nh2 = note(rx, ny, cw, lines2, h=nh)
    cy = ny + max(nh1, nh2) + 70
    ref_attrs = [(c, t, (st["cities"] if g == "CITY_GLOSS" else g), k) for c, t, g, k in REF_ATTRS]
    ch = 124 + 24 + 94 * max(len(ref_attrs), len(IMG_ATTRS)) + 24  # equal class height
    c1, ch1 = klass(lx, cy, cw, "BuildingReferenceRecord", "\u00abtable\u00bb",
                    ref_attrs, BLUE, "#F3F8FC", h=ch)
    c2, ch2 = klass(rx, cy, cw, "StreetViewImageRecord", "\u00abtable\u00bb",
                    IMG_ATTRS, MBLUE, "#F5F9FB", h=ch)
    anchors = (f'<line x1="{lx + cw / 2}" y1="{ny + nh1}" x2="{lx + cw / 2}" y2="{cy}" stroke="#9CA3AF" stroke-width="2.5" stroke-dasharray="10 8"/>'
               f'<line x1="{rx + cw / 2}" y1="{ny + nh2}" x2="{rx + cw / 2}" y2="{cy}" stroke="#9CA3AF" stroke-width="2.5" stroke-dasharray="10 8"/>')
    ay = cy + 124 + 24 + 29  # level of the first attribute row (pand_id)
    ax1, ax2 = lx + cw, rx
    mid = (ax1 + ax2) / 2
    assoc = [
        f'<line x1="{ax1}" y1="{ay}" x2="{ax2}" y2="{ay}" stroke="{DBLUE}" stroke-width="4"/>',
        f'<text x="{ax1 + 14}" y="{ay - 18}" class="mult">1</text>',
        f'<text x="{ax2 - 14}" y="{ay - 18}" class="mult" text-anchor="end">{st["lo"]}..{st["hi"]}</text>',
    ]
    footer = ''
    H = cy + max(ch1, ch2) + 40
    css = f"""
      text {{ font-family: Arial, Helvetica, sans-serif; fill: {INK}; }}
      .stereo {{ font-size: 30px; fill: #FFFFFF; font-style: italic; }}
      .cname {{ font-size: 40px; font-weight: 700; fill: #FFFFFF; }}
      .attr {{ font-size: 30px; }}
      .mono {{ font-weight: 700; fill: {INK}; }}
      .typ {{ fill: #4B5563; }}
      .tag {{ font-weight: 700; fill: {ORANGE}; }}
      .gloss {{ font-size: 30px; fill: #4B5563; }}
      .notehead {{ font-size: 30px; font-weight: 700; fill: {INK}; }}
      .note {{ font-size: 30px; fill: {INK}; }}
      .mult {{ font-size: 32px; font-weight: 700; fill: {DBLUE}; }}
      .assoc {{ font-size: 26px; font-weight: 700; fill: {DBLUE}; }}
      .assocsub {{ font-size: 22px; fill: #4B5563; }}
      .foot {{ font-size: 18px; fill: #6B7280; }}
    """
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
           f'<style>{css}</style><rect width="{W}" height="{H}" fill="#FFFFFF"/>'
           f'{n1}{n2}{anchors}{c1}{c2}{"".join(assoc)}{footer}</svg>')
    return svg, W, H


def render_png(html_path: Path, png_path: Path, width: int, height: int, scale: int = 2):
    tmp = png_path.with_name(png_path.stem + "_raw.png")
    cmd = [str(CHROME), "--headless=new", "--no-sandbox", "--disable-gpu",
           "--disable-dev-shm-usage", "--hide-scrollbars",
           f"--force-device-scale-factor={scale}", f"--window-size={width},{height}",
           f"--screenshot={tmp}", html_path.resolve().as_uri()]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    im = Image.open(tmp).convert("RGB")
    pad = 20 * scale
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bbox = ImageChops.difference(im, bg).getbbox()
    if bbox:
        im = im.crop((max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                      min(im.width, bbox[2] + pad), min(im.height, bbox[3] + pad)))
    im.save(png_path)
    tmp.unlink()


def main():
    st = stats()
    svg, W, H = build_svg(st)
    (OUT / "fig_3_2_record_structure_uml.svg").write_text(svg, encoding="utf-8")
    html = (f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>Stage 1 record structure</title>'
            f'<style>body{{margin:0;background:#fff}}</style></head><body>{svg}</body></html>')
    hp = OUT / "fig_3_2_record_structure_uml.html"
    hp.write_text(html, encoding="utf-8")
    render_png(hp, OUT / "fig_3_2_record_structure_uml.png", W + 20, H + 20)
    print(st)


if __name__ == "__main__":
    main()
