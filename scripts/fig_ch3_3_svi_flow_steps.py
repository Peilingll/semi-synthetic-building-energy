"""Fig. 3.3 -- five-step SVI image transformation, one worked example (Rotterdam).

Steps (one real image per step, all read from the OpenFACADES cell output):
  1  Mapillary panorama            02_img/mly_svi/batch_1/<pid>.png (raw download, 2048x1024)
  2  Candidate building / view     01_data/footprint.geojson + 01_data/aov.csv  (drawn as a map)
  3  GroundingDINO detection       02_img/annotated_img/<pid>.png + 02_img/building_bbox.csv
  4  Perspective reprojection      equirectangular slice of the raw panorama at the detected box
                                   -> 02_img/individual_building/pid_<pid>_bdid_<bdid>.png
  5  Building-level SVI crop       data/processed/svi_manifest.parquet rows for the BAG pand_id

Example pair: panorama 686679844161937, OpenFACADES building 276637663,
BAG pand_id 0599100000611463 (Rotterdam, cell_0252). Same pair as the previous
two-panel version of the figure.

Outputs
  Thesis_reports/CH3/3_2_2/fig_svi_flow_steps.html   (+ flow_steps/ image assets)
  Thesis_reports/CH3/3_2_2/fig_svi_flow_steps.png    (Chrome headless screenshot, trimmed)

Run:  .venv/Scripts/python.exe scripts/fig_ch3_3_svi_flow_steps.py
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image, ImageChops

REPO = Path(__file__).resolve().parents[1]
CELL = REPO / "data/openfacades_output/phase_c_rotterdam_grid/cells/cell_0252/output"
OUT_DIR = REPO / "Thesis_reports/CH3/3_2_2"
ASSETS = OUT_DIR / "flow_steps"
MANIFEST = REPO / "data/processed/svi_manifest.parquet"

PID = "686679844161937"
BDID = "276637663"
CHROME = Path(r"C:/Program Files/Google/Chrome/Application/chrome.exe")

# TUM corporate palette
BLUE, DBLUE, LBLUE, MBLUE = "#0065BD", "#005293", "#98C6EA", "#64A0C8"
GREY, GREEN, ORANGE = "#DAD7CB", "#A2AD00", "#E37222"


# ----------------------------------------------------------------------------- data
def load_bbox() -> tuple[int, int, int, int, float]:
    """Detected box in panorama pixels (x0, y0, w, h) and confidence."""
    with open(CELL / "02_img/building_bbox.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["pid"] == PID and r["building_id"] == BDID:
                cx, cy, w, h, conf = json.loads(r["boxes"])
                W, H = 2048, 1024
                return (int(round((cx - w / 2) * W)), int(round((cy - h / 2) * H)),
                        int(round(w * W)), int(round(h * H)), conf)
    raise SystemExit("box not found")


def load_candidates() -> list[dict]:
    with open(CELL / "01_data/aov.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["building_id"] == BDID]
    for r in rows:
        for k in ("lat", "lng", "compass_angle", "left_angle_geo", "right_angle_geo",
                  "aov_geo", "distance"):
            r[k] = float(r[k])
    return rows


def load_manifest_rows() -> pd.DataFrame:
    m = pd.read_parquet(MANIFEST)
    rows = m[m["bdid"].astype(str) == BDID].copy()
    rows["cell"] = rows["file_path"].str.extract(r"(cell_\d+)")
    return rows


# ----------------------------------------------------------------------------- assets
def make_assets(box):
    ASSETS.mkdir(exist_ok=True)
    x0, y0, w, h, _ = box
    pano = Image.open(CELL / f"02_img/mly_svi/batch_1/{PID}.png").convert("RGB")
    pano.save(ASSETS / "step1_panorama.jpg", quality=88)

    # zoom window of the annotated panorama around the detected box
    zoom = (x0 - 150, y0 - 90, x0 + w + 150, y0 + h + 70)
    ann = Image.open(CELL / f"02_img/annotated_img/{PID}.png").convert("RGB")
    ann.crop(zoom).save(ASSETS / "step3_detection_zoom.jpg", quality=90)

    pano.crop((x0, y0, x0 + w, y0 + h)).save(ASSETS / "step4_equirect_slice.jpg", quality=90)
    crop = Image.open(CELL / f"02_img/individual_building/pid_{PID}_bdid_{BDID}.png").convert("RGB")
    crop.save(ASSETS / "step4_rectified_crop.jpg", quality=90)
    return zoom, crop.size


def copy_building_crops(rows: pd.DataFrame) -> list[dict]:
    """One thumbnail per retained panorama (manifest rows of cell_0252)."""
    out = []
    sel = rows[rows["cell"] == "cell_0252"].sort_values("image_idx")
    for _, r in sel.iterrows():
        src = Path(r["file_path"])
        im = Image.open(src).convert("RGB")
        name = f"step5_{r['panorama_id']}.jpg"
        im.save(ASSETS / name, quality=88)
        out.append({"pid": str(r["panorama_id"]), "file": name, "size": im.size,
                    "aov": float(r["aov_geo"]), "dist": float(r["distance"])})
    return out


# ----------------------------------------------------------------------------- map (step 2)
def build_map_svg(cands: list[dict], retained: set[str], ox: float, oy: float,
                  W: int, H: int) -> str:
    """Local-metre map: target footprint, neighbours, candidate cameras, AoV wedge."""
    cam = next(r for r in cands if r["pid"] == PID)
    lat0, lng0 = cam["lat"], cam["lng"]
    kx = 111_320 * math.cos(math.radians(lat0))
    ky = 111_320

    def m(lng, lat):  # metres east/north of the example camera
        return (lng - lng0) * kx, (lat - lat0) * ky

    # extent in metres, camera placed right of centre (footprint lies to the west)
    xmin, xmax, ymin, ymax = -46.0, 34.0, -37.0, 37.0
    s = min(W / (xmax - xmin), H / (ymax - ymin))

    def px(e, n):
        return ox + (e - xmin) * s, oy + (ymax - n) * s

    g = json.load(open(CELL / "01_data/footprint.geojson"))
    polys = []
    for f in g["features"]:
        if f["geometry"]["type"] != "Polygon":
            continue
        ring = f["geometry"]["coordinates"][0]
        pts = [m(x, y) for x, y in ring]
        if all(e < xmin - 5 or e > xmax + 5 or n < ymin - 5 or n > ymax + 5 for e, n in pts):
            continue
        polys.append((f["properties"]["building_id"], pts))

    parts = [f'<clipPath id="mapclip"><rect x="{ox}" y="{oy}" width="{W}" height="{H}"/></clipPath>',
             f'<rect x="{ox}" y="{oy}" width="{W}" height="{H}" fill="#f7f6f2"/>',
             '<g clip-path="url(#mapclip)">']
    for bid, pts in polys:
        d = "M" + " L".join(f"{px(e, n)[0]:.1f},{px(e, n)[1]:.1f}" for e, n in pts) + " Z"
        if bid == BDID:
            parts.append(f'<path d="{d}" fill="{LBLUE}" stroke="{BLUE}" stroke-width="3"/>')
        else:
            parts.append(f'<path d="{d}" fill="{GREY}" stroke="#8a8677" stroke-width="1.5"/>')

    # 30 m isovist search radius around the example camera
    cx, cy = px(0, 0)
    parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{30 * s:.1f}" fill="none" '
                 f'stroke="{MBLUE}" stroke-width="2" stroke-dasharray="8 6"/>')

    # AoV wedge (geo bearings, clockwise from north)
    L = cam["distance"] + 6
    wedge = [px(0, 0)]
    a0, a1 = cam["left_angle_geo"], cam["right_angle_geo"]
    for i in range(21):
        a = math.radians(a0 + (a1 - a0) * i / 20)
        wedge.append(px(L * math.sin(a), L * math.cos(a)))
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in wedge) + " Z"
    parts.append(f'<path d="{d}" fill="{ORANGE}" fill-opacity="0.28" stroke="{ORANGE}" stroke-width="2.5"/>')

    # candidate cameras
    for r in cands:
        e, n = m(r["lng"], r["lat"])
        x, y = px(e, n)
        if r["pid"] == PID:
            continue
        if r["pid"] in retained:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{ORANGE}" stroke="#fff" stroke-width="2"/>')
        else:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#fff" stroke="#6b6b6b" stroke-width="2"/>')
    parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="10" fill="{ORANGE}" stroke="#fff" stroke-width="3"/>')
    parts.append('</g>')

    # scale bar and north arrow
    bx, by = ox + 20, oy + H - 22
    parts.append(f'<line x1="{bx}" y1="{by}" x2="{bx + 10 * s:.1f}" y2="{by}" stroke="#333" stroke-width="4"/>')
    parts.append(f'<text x="{bx}" y="{by - 8}" class="t-map">10 m</text>')
    nx, ny = ox + W - 30, oy + 60
    parts.append(f'<path d="M{nx},{ny - 34} L{nx - 10},{ny} L{nx},{ny - 8} L{nx + 10},{ny} Z" fill="#333"/>')
    parts.append(f'<text x="{nx}" y="{ny + 22}" class="t-map" text-anchor="middle">N</text>')

    # no in-map text labels (scale bar and north arrow only)
    parts.append(f'<rect x="{ox}" y="{oy}" width="{W}" height="{H}" fill="none" stroke="#777" stroke-width="1.5"/>')
    return "\n".join(parts)


# ----------------------------------------------------------------------------- html
def build_html(box, zoom, crop_size, cands, retained_pids, thumbs, pand_id, n_manifest_rows):
    x0, y0, w, h, conf = box
    n_cand = len(cands)
    n_ret = len(retained_pids)

    # ---------- layout (viewBox 2400 x 1500, body text 24px) ----------
    VB_W, VB_H = 2400, 1380
    ROW1_Y, ROW1_H = 110, 600
    ROW2_Y, ROW2_H = 810, 530
    p1 = dict(x=40, w=880)
    p2 = dict(x=980, w=600)
    p3 = dict(x=1640, w=720)
    p4 = dict(x=40, w=1000)
    p5 = dict(x=1100, w=1260)
    HEAD = 96  # header band height inside a panel

    def panel(p, y, hgt, n, title, sub):
        return (f'<rect x="{p["x"]}" y="{y}" width="{p["w"]}" height="{hgt}" rx="10" class="panel"/>'
                f'<circle cx="{p["x"] + 38}" cy="{y + 40}" r="24" fill="{BLUE}"/>'
                f'<text x="{p["x"] + 38}" y="{y + 49}" class="t-step" text-anchor="middle">{n}</text>'
                f'<text x="{p["x"] + 76}" y="{y + 40}" class="t-title">{title}</text>'
                f'<text x="{p["x"] + 76}" y="{y + 74}" class="t-sub">{sub}</text>')

    def fit(iw, ih, bw, bh):
        s = min(bw / iw, bh / ih)
        return iw * s, ih * s

    parts = []
    # ---------- step 1: panorama ----------
    parts.append(panel(p1, ROW1_Y, ROW1_H, 1, "Mapillary 360&#176; panorama",
                       f"raw equirectangular download, 2,048 &#215; 1,024 px &#183; panorama {PID}"))
    iw, ih = fit(2048, 1024, p1["w"] - 40, ROW1_H - HEAD - 60)
    ix, iy = p1["x"] + 20, ROW1_Y + HEAD
    parts.append(f'<image href="flow_steps/step1_panorama.jpg" x="{ix}" y="{iy}" width="{iw:.0f}" height="{ih:.0f}"/>')
    parts.append(f'<rect x="{ix}" y="{iy}" width="{iw:.0f}" height="{ih:.0f}" class="imgframe"/>')
    sc = iw / 2048
    zx, zy, zx1, zy1 = zoom
    parts.append(f'<rect x="{ix + zx * sc:.1f}" y="{iy + zy * sc:.1f}" width="{(zx1 - zx) * sc:.1f}" '
                 f'height="{(zy1 - zy) * sc:.1f}" fill="none" stroke="{BLUE}" stroke-width="3" stroke-dasharray="10 6"/>')
    parts.append(f'<text x="{ix + zx * sc:.0f}" y="{iy + zy * sc - 10:.0f}" class="t-note" fill="{DBLUE}">window shown in step 3</text>')
    parts.append(f'<text x="{ix}" y="{iy + ih + 34:.0f}" class="t-note">Imagery &#169; Mapillary contributors, CC BY-SA 4.0 &#183; Rotterdam</text>')

    # ---------- step 2: candidate building / view selection ----------
    parts.append(panel(p2, ROW1_Y, ROW1_H, 2, "Candidate building and view selection",
                       "isovist pairing (30 m) and AoV ranking"))
    mw, mh = p2["w"] - 40, ROW1_H - HEAD - 60
    parts.append(build_map_svg(cands, retained_pids, p2["x"] + 20, ROW1_Y + HEAD, mw, mh))
    ly = ROW1_Y + HEAD + mh + 34
    lx = p2["x"] + 20
    parts.append(f'<circle cx="{lx + 8}" cy="{ly - 8}" r="7" fill="{ORANGE}" stroke="#fff" stroke-width="2"/>'
                 f'<text x="{lx + 24}" y="{ly}" class="t-note">retained ({n_ret})</text>'
                 f'<circle cx="{lx + 190}" cy="{ly - 8}" r="6" fill="#fff" stroke="#6b6b6b" stroke-width="2"/>'
                 f'<text x="{lx + 206}" y="{ly}" class="t-note">other candidate cameras ({n_cand - n_ret})</text>')

    # ---------- step 3: GroundingDINO detection ----------
    parts.append(panel(p3, ROW1_Y, ROW1_H, 3, "GroundingDINO building detection",
                       f"box for building_id {BDID}, confidence {conf:.2f}"))
    zw, zh = zx1 - zx, zy1 - zy
    iw3, ih3 = fit(zw, zh, p3["w"] - 40, ROW1_H - HEAD - 60)
    ix3, iy3 = p3["x"] + 20, ROW1_Y + HEAD
    parts.append(f'<image href="flow_steps/step3_detection_zoom.jpg" x="{ix3}" y="{iy3}" width="{iw3:.0f}" height="{ih3:.0f}"/>')
    parts.append(f'<rect x="{ix3}" y="{iy3}" width="{iw3:.0f}" height="{ih3:.0f}" class="imgframe"/>')
    parts.append(f'<text x="{ix3}" y="{iy3 + ih3 + 34:.0f}" class="t-note">box and label drawn by OpenFACADES; zoomed window</text>')

    # ---------- step 4: perspective reprojection ----------
    parts.append(panel(p4, ROW2_Y, ROW2_H, 4, "Perspective reprojection",
                       "the detected region is re-projected from the sphere to a rectilinear view and cropped"))
    ah = ROW2_H - HEAD - 90
    sw, sh = fit(w, h, 420, ah)
    sx, sy = p4["x"] + 40, ROW2_Y + HEAD
    parts.append(f'<image href="flow_steps/step4_equirect_slice.jpg" x="{sx}" y="{sy}" width="{sw:.0f}" height="{sh:.0f}"/>')
    parts.append(f'<rect x="{sx}" y="{sy}" width="{sw:.0f}" height="{sh:.0f}" class="imgframe"/>')
    parts.append(f'<text x="{sx + sw / 2:.0f}" y="{sy + sh + 34:.0f}" class="t-note" text-anchor="middle">equirectangular region, {w} &#215; {h} px</text>')
    cw, ch = fit(*crop_size, 420, ah)
    cx4 = sx + sw + 110
    parts.append(f'<path d="M{sx + sw + 30},{sy + sh / 2:.0f} L{cx4 - 30},{sy + sh / 2:.0f}" class="arr"/>')
    parts.append(f'<image href="flow_steps/step4_rectified_crop.jpg" x="{cx4:.0f}" y="{sy}" width="{cw:.0f}" height="{ch:.0f}"/>')
    parts.append(f'<rect x="{cx4:.0f}" y="{sy}" width="{cw:.0f}" height="{ch:.0f}" class="imgframe" style="stroke:{GREEN};stroke-width:5"/>')
    parts.append(f'<text x="{cx4 + cw / 2:.0f}" y="{sy + ch + 34:.0f}" class="t-note" text-anchor="middle">rectified facade crop, {crop_size[0]} &#215; {crop_size[1]} px</text>')

    # ---------- step 5: building-level SVI crop ----------
    parts.append(panel(p5, ROW2_Y, ROW2_H, 5, "Building-level SVI crops",
                       f"BAG pand_id {pand_id} &#8592; building_id {BDID} &#183; {n_ret} of {n_cand} candidates retained, AoV rank, cap 8"))
    th = 340
    gap = 22
    tx = p5["x"] + 24
    ty = ROW2_Y + HEAD + 10
    avail = p5["w"] - 48 - gap * (len(thumbs) - 1)
    widths = [fit(t["size"][0], t["size"][1], 10_000, th)[0] for t in thumbs]
    scale = min(1.0, avail / sum(widths))
    for t, tw in zip(thumbs, widths):
        tw *= scale
        th_i = th * scale
        cls = f' style="stroke:{GREEN};stroke-width:5"' if t["pid"] == PID else ""
        parts.append(f'<image href="flow_steps/{t["file"]}" x="{tx:.0f}" y="{ty}" width="{tw:.0f}" height="{th_i:.0f}"/>')
        parts.append(f'<rect x="{tx:.0f}" y="{ty}" width="{tw:.0f}" height="{th_i:.0f}" class="imgframe"{cls}/>')
        parts.append(f'<text x="{tx + tw / 2:.0f}" y="{ty + th_i + 30:.0f}" class="t-note" text-anchor="middle">AoV {t["aov"]:.1f}&#176; &#183; {t["dist"]:.1f} m</text>')
        parts.append(f'<text x="{tx + tw / 2:.0f}" y="{ty + th_i + 56:.0f}" class="t-note mono" text-anchor="middle">{t["pid"]}</text>')
        tx += tw + gap
    parts.append(f'<text x="{p5["x"] + 24}" y="{ROW2_Y + ROW2_H - 26}" class="t-note">green frame = the panorama followed in steps 1&#8211;4; '
                 f'source: data/processed/svi_manifest.parquet</text>')

    # ---------- flow arrows ----------
    my = ROW1_Y + ROW1_H / 2
    parts.append(f'<path d="M{p1["x"] + p1["w"] + 8},{my} L{p2["x"] - 8},{my}" class="arr"/>')
    parts.append(f'<path d="M{p2["x"] + p2["w"] + 8},{my} L{p3["x"] - 8},{my}" class="arr"/>')
    railY = ROW1_Y + ROW1_H + 50
    parts.append(f'<path d="M{p3["x"] + p3["w"] / 2},{ROW1_Y + ROW1_H + 4} L{p3["x"] + p3["w"] / 2},{railY} '
                 f'L{p4["x"] + p4["w"] / 2},{railY} L{p4["x"] + p4["w"] / 2},{ROW2_Y - 8}" class="arr"/>')
    my2 = ROW2_Y + ROW2_H / 2
    parts.append(f'<path d="M{p4["x"] + p4["w"] + 8},{my2} L{p5["x"] - 8},{my2}" class="arr"/>')

    svg = "\n".join(parts)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Fig 3.3 - SVI image transformation in five steps (worked example)</title>
<style>
  body {{ font-family: "Segoe UI", Arial, sans-serif; background: #fff; color: #111; margin: 0; }}
  .figwrap {{ width: {VB_W}px; margin: 0 auto; }}
  svg {{ width: {VB_W}px; height: {VB_H}px; display: block; }}
  .caption {{ max-width: 1400px; margin: 16px auto 24px; font-size: 22px; color: #333; line-height: 1.5; }}
  .panel {{ fill: #fff; stroke: #9a9a9a; stroke-width: 2; }}
  .imgframe {{ fill: none; stroke: #555; stroke-width: 1.5; }}
  .arr {{ stroke: #333; stroke-width: 4; fill: none; marker-end: url(#arrK); }}
  .t-step  {{ font: 700 26px "Segoe UI", Arial, sans-serif; fill: #fff; }}
  .t-title {{ font: 700 30px "Segoe UI", Arial, sans-serif; fill: #111; }}
  .t-sub   {{ font: 24px "Segoe UI", Arial, sans-serif; fill: #444; }}
  .t-note  {{ font: 22px "Segoe UI", Arial, sans-serif; fill: #444; }}
  .t-map   {{ font: 600 22px "Segoe UI", Arial, sans-serif; fill: #333; }}
  .mono    {{ font-family: Consolas, "Courier New", monospace; }}
</style>
</head>
<body>
<div class="figwrap">
<svg viewBox="0 0 {VB_W} {VB_H}" xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Five-step transformation from a Mapillary panorama to building-level SVI crops">
  <defs>
    <marker id="arrK" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#333"/></marker>
  </defs>
  <text x="40" y="66" class="t-title">From a Mapillary panorama to building-level SVI crops &#183; one worked example (Rotterdam)</text>
{svg}
</svg>
<p class="caption">
<b>Fig. 3.3 &mdash; Image transformation from a retained 360&deg; panorama to building-level facade crops.</b>
Every panel shows real pipeline output for one panorama&ndash;building pair (panorama <span class="mono">{PID}</span>,
OpenFACADES <span class="mono">building_id {BDID}</span>, BAG <span class="mono">pand_id {pand_id}</span>; Rotterdam, grid cell 0252).
(1) Raw equirectangular panorama. (2) Map drawn from <span class="mono">01_data/footprint.geojson</span> and
<span class="mono">01_data/aov.csv</span>: the target footprint, the {n_cand} candidate cameras that see it within the 30 m isovist
radius, the AoV wedge of the followed panorama and the {n_ret} cameras retained after AoV ranking.
(3) GroundingDINO detection as annotated by OpenFACADES (<span class="mono">02_img/annotated_img</span>, box from
<span class="mono">02_img/building_bbox.csv</span>). (4) The detected region as it appears in the equirectangular image and
after perspective reprojection (<span class="mono">02_img/individual_building</span>). (5) All crops of the same BAG building
in the per-building SVI manifest, ranked by AoV. Imagery &copy; Mapillary contributors (CC BY-SA 4.0).
</p>
</div>
</body>
</html>
"""
    return html


MAP_CSS = """
  body { font-family: "Segoe UI", Arial, sans-serif; background: #fff; margin: 0; }
  svg { display: block; }
  .t-map  { font: 600 22px "Segoe UI", Arial, sans-serif; fill: #333; }
  .t-note { font: 22px "Segoe UI", Arial, sans-serif; fill: #444; }
"""


def build_map_html(cands, retained, n_cand, n_ret, W=600, H=470):
    """Map-only version of step 2, saved as its own figure."""
    svg = build_map_svg(cands, retained, 20, 20, W, H)
    ly = 20 + H + 40
    legend = (f'<circle cx="28" cy="{ly - 8}" r="7" fill="{ORANGE}" stroke="#fff" stroke-width="2"/>'
              f'<text x="44" y="{ly}" class="t-note">retained ({n_ret})</text>'
              f'<circle cx="210" cy="{ly - 8}" r="6" fill="#fff" stroke="#6b6b6b" stroke-width="2"/>'
              f'<text x="226" y="{ly}" class="t-note">other candidate cameras ({n_cand - n_ret})</text>')
    vw, vh = W + 40, H + 80
    return (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
            f'<title>Candidate building and view selection map</title><style>{MAP_CSS}</style></head><body>'
            f'<svg viewBox="0 0 {vw} {vh}" width="{vw}" height="{vh}" xmlns="http://www.w3.org/2000/svg">'
            f'{svg}{legend}</svg></body></html>')


def render_png(html_path: Path, png_path: Path, width: int, height: int, scale: int = 1):
    tmp = png_path.with_name(png_path.stem + "_raw.png")
    cmd = [str(CHROME), "--headless=new", "--disable-gpu", "--hide-scrollbars",
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
    box = load_bbox()
    cands = load_candidates()
    rows = load_manifest_rows()
    pand_id = rows["pand_id"].iloc[0]
    retained = set(rows["panorama_id"].astype(str))
    zoom, crop_size = make_assets(box)
    thumbs = copy_building_crops(rows)
    html = build_html(box, zoom, crop_size, cands, retained, thumbs, pand_id, len(rows))
    html_path = OUT_DIR / "fig_svi_flow_steps.html"
    html_path.write_text(html, encoding="utf-8")
    png_path = OUT_DIR / "fig_svi_flow_steps.png"
    render_png(html_path, png_path, 2400, 1400)
    map_html = OUT_DIR / "fig_svi_view_selection_map.html"
    map_html.write_text(build_map_html(cands, retained, len(cands), len(retained)), encoding="utf-8")
    map_png = OUT_DIR / "fig_svi_view_selection_map.png"
    render_png(map_html, map_png, 700, 600, scale=2)
    print("wrote", map_png, Image.open(map_png).size)
    print("box", box)
    print(f"candidates {len(cands)}, retained {sorted(retained)}, manifest rows {len(rows)} "
          f"(cells {sorted(rows['cell'].unique())})")
    print("wrote", html_path, png_path, Image.open(png_path).size)


if __name__ == "__main__":
    main()
