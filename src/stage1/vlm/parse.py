"""Robust JSON parser for the 3-field Stage 1 VLM prompt response (v3).

Extracts the first {...} block (tolerating markdown fences and trailing
text), validates the three required keys, and enforces a whitelist for
building_type.

Prompt v3 dropped construction_period and facade_material: listing the
TABULA period boundaries in the prompt anchored year predictions onto
boundary years (v2 failure mode), and material has no GT to validate.
The TABULA period is now derived from the year downstream
(src.tabula_matcher.classify_period) at the aggregation step.

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


def parse_response(raw: str) -> dict:
    """Return a dict with keys parse_ok, pred_type, pred_year, pred_floors,
    parse_error.

    parse_ok=True only if all three predictions are valid. Otherwise return
    whatever was recoverable plus a parse_error string."""
    out = {
        "parse_ok": False,
        "pred_type": None,
        "pred_year": None,
        "pred_floors": None,
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
    out["pred_floors"] = _safe_int(obj.get("num_floors"), FLOORS_MIN, FLOORS_MAX)

    required = ["pred_type", "pred_year", "pred_floors"]
    missing = [k for k in required if out[k] is None]
    if missing:
        out["parse_error"] = "missing_or_invalid:" + ",".join(missing)
    else:
        out["parse_ok"] = True

    return out


if __name__ == "__main__":
    import sys
    samples = [
        '{"building_type": "TH", "construction_year": 1932, "num_floors": 3}',
        ('```json\n{"building_type": "AB", "construction_year": 1980, '
         '"num_floors": 5}\n```'),
        ('Sure! Here is the answer:\n'
         '{"building_type": "SFH", "construction_year": 1925, "num_floors": 2}'),
        # extra keys from an older prompt are ignored
        ('{"building_type": "TH", "construction_year": 1972, '
         '"construction_period": "NL.01", "num_floors": 3, '
         '"facade_material": "brick"}'),
        # invalid type
        '{"building_type": "office", "construction_year": 1990, "num_floors": 4}',
        # year out of range
        '{"building_type": "TH", "construction_year": 1750, "num_floors": 2}',
        # missing floors
        '{"building_type": "AB", "construction_year": 1995}',
        # malformed
        "not a json at all",
    ]
    for i, s in enumerate(samples):
        r = parse_response(s)
        print(f"[{i}] ok={r['parse_ok']} err={r['parse_error']} "
              f"type={r['pred_type']} y={r['pred_year']} fl={r['pred_floors']}")
    sys.exit(0)
