"""Robust JSON parser for the 6-field Stage 1 VLM prompt response.

Extracts the first {...} block (tolerating markdown fences and trailing
text), validates the six required keys, enforces whitelists for the three
categorical fields (building_type / construction_period / facade_material),
and computes a year_period_consistent flag.

A response is `parse_ok=True` only if every required key is present and
type/value valid. Otherwise the function returns a dict with `parse_ok=False`
and as many recoverable fields as possible.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

BUILDING_TYPE_WHITELIST = {"SFH", "TH", "MFH", "AB"}
PERIOD_WHITELIST = {"NL.01", "NL.02", "NL.03", "NL.04", "NL.05", "NL.06"}
MATERIAL_WHITELIST = {
    "brick", "concrete", "wood", "stucco",
    "metal", "stone", "mixed", "other",
}

PERIOD_RANGES: dict[str, tuple[int, int]] = {
    "NL.01": (1800, 1964),
    "NL.02": (1965, 1974),
    "NL.03": (1975, 1991),
    "NL.04": (1992, 2005),
    "NL.05": (2006, 2014),
    "NL.06": (2015, 2025),
}

YEAR_MIN, YEAR_MAX = 1800, 2025
FLOORS_MIN, FLOORS_MAX = 1, 30


def _extract_json_block(raw: str) -> str | None:
    if raw is None:
        return None
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        return fence.group(1)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0) if m else None


def _safe_int(v, lo: int, hi: int) -> int | None:
    try:
        i = int(float(v))
    except (TypeError, ValueError):
        return None
    return i if lo <= i <= hi else None


def _safe_float(v, lo: float, hi: float) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if lo <= f <= hi else None


def parse_response(raw: str) -> dict:
    """Return a dict with keys parse_ok, pred_type, pred_year, pred_period,
    pred_floors, pred_material, pred_wwr, year_period_consistent, parse_error.

    parse_ok=True only if all six predictions are valid. Otherwise return
    whatever was recoverable plus a parse_error string."""
    out = {
        "parse_ok": False,
        "pred_type": None,
        "pred_year": None,
        "pred_period": None,
        "pred_floors": None,
        "pred_material": None,
        "year_period_consistent": None,
        "parse_error": None,
    }

    block = _extract_json_block(raw)
    if block is None:
        out["parse_error"] = "no_json_block"
        return out

    text = re.sub(r",\s*,", ",", block)
    text = re.sub(r",(\s*[}\]])", r"\1", text)

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        out["parse_error"] = f"json_decode:{exc.msg}"
        return out

    if not isinstance(obj, dict):
        out["parse_error"] = "not_a_dict"
        return out

    raw_type = obj.get("building_type")
    if isinstance(raw_type, str) and raw_type.upper() in BUILDING_TYPE_WHITELIST:
        out["pred_type"] = raw_type.upper()

    out["pred_year"] = _safe_int(obj.get("construction_year"), YEAR_MIN, YEAR_MAX)

    raw_period = obj.get("construction_period")
    if isinstance(raw_period, str) and raw_period.upper() in PERIOD_WHITELIST:
        out["pred_period"] = raw_period.upper()

    out["pred_floors"] = _safe_int(obj.get("num_floors"), FLOORS_MIN, FLOORS_MAX)

    raw_material = obj.get("facade_material")
    if isinstance(raw_material, str) and raw_material.lower() in MATERIAL_WHITELIST:
        out["pred_material"] = raw_material.lower()

    if out["pred_year"] is not None and out["pred_period"] is not None:
        lo, hi = PERIOD_RANGES[out["pred_period"]]
        out["year_period_consistent"] = bool(lo <= out["pred_year"] <= hi)

    required = ["pred_type", "pred_year", "pred_period", "pred_floors",
                "pred_material"]
    missing = [k for k in required if out[k] is None]
    if missing:
        out["parse_error"] = "missing_or_invalid:" + ",".join(missing)
    else:
        out["parse_ok"] = True

    return out


if __name__ == "__main__":
    import sys
    samples = [
        ('{"building_type": "TH", "construction_year": 1965, '
         '"construction_period": "NL.02", "num_floors": 3, '
         '"facade_material": "brick"}'),
        ('```json\n{"building_type": "AB", "construction_year": 1980, '
         '"construction_period": "NL.03", "num_floors": 5, '
         '"facade_material": "concrete"}\n```'),
        ('Sure! Here is the answer:\n'
         '{"building_type": "SFH", "construction_year": 1925, '
         '"construction_period": "NL.01", "num_floors": 2, '
         '"facade_material": "brick"}'),
        # year/period inconsistent
        ('{"building_type": "TH", "construction_year": 1972, '
         '"construction_period": "NL.01", "num_floors": 3, '
         '"facade_material": "brick"}'),
        # invalid type
        ('{"building_type": "office", "construction_year": 1990, '
         '"construction_period": "NL.03", "num_floors": 4, '
         '"facade_material": "glass"}'),
        # malformed
        ("not a json at all"),
    ]
    for i, s in enumerate(samples):
        r = parse_response(s)
        print(f"[{i}] ok={r['parse_ok']} err={r['parse_error']} "
              f"type={r['pred_type']} y={r['pred_year']} p={r['pred_period']} "
              f"yp_consistent={r['year_period_consistent']}")
    sys.exit(0)
