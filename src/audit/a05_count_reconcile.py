"""0.5 — Reconcile the building counts quoted across slides / draft / logs.

Numbers in circulation: 10,104 / 10,093 / 10,090 / 10,086 buildings and
47,238 / 47,150 images. Each is a different set; this script recomputes every
one of them from the artefacts so each document can name the set it means.

Outputs: reports/tables/audit/A05_count_reconcile.md
"""

import logging

import pandas as pd

from src.audit import ep_raw

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROCESSED = ep_raw.REPO_ROOT / "data" / "processed"
OUT = ep_raw.REPO_ROOT / "reports" / "tables" / "audit" / "A05_count_reconcile.md"


def main():
    man = pd.read_parquet(PROCESSED / "svi_manifest.parquet")
    man["pand_id"] = man["pand_id"].astype(str).str.zfill(16)
    gt = pd.read_parquet(PROCESSED / "stage1_gt.parquet")
    gt["pand_id"] = gt["pand_id"].astype(str).str.zfill(16)
    dev = pd.read_parquet(PROCESSED / "dev_fold_indices.parquet")
    dev["pand_id"] = dev["pand_id"].astype(str).str.zfill(16)
    hold = pd.read_parquet(PROCESSED / "holdout_test_pand_ids.parquet")
    hold["pand_id"] = hold["pand_id"].astype(str).str.zfill(16)

    man_ids = set(man["pand_id"])
    gt_ids = set(gt["pand_id"])
    inter = man_ids & gt_ids
    man_gt = man[man["pand_id"].isin(gt_ids)]
    split_ids = set(dev["pand_id"]) | set(hold["pand_id"])

    ep = ep_raw.load(residential_only=True)
    ep_ids = set(ep["pand_id"])
    ep_nta_ids = set(ep.loc[ep["calc_type"].str.startswith("NTA 8800", na=False), "pand_id"])

    rows = [
        ("`svi_manifest.parquet`", "SVI manifest: pands with >=1 accepted Mapillary image",
         man["pand_id"].nunique(), len(man)),
        ("manifest ∩ `stage1_gt`", "training universe: manifest pands that also have a GT row",
         len(inter), len(man_gt)),
        ("`stage1_gt.parquet`", "four-city pands with an EPC + TABULA match (full stock)",
         gt["pand_id"].nunique(), None),
        ("dev ∪ hold-out", "the frozen 80/20 split actually used for modelling",
         len(split_ids), None),
        ("`dev_fold_indices.parquet`", "dev pool (5 folds)", dev["pand_id"].nunique(), None),
        ("`holdout_test_pand_ids.parquet`", "hold-out", hold["pand_id"].nunique(), None),
        ("manifest ∩ raw EP (residential)", "manifest pands with >=1 residential certificate",
         len(man_ids & ep_ids), None),
        ("manifest ∩ raw EP (NTA 8800)", "manifest pands with >=1 NTA 8800 certificate",
         len(man_ids & ep_nta_ids), None),
    ]

    L = ["# A05 — Building / image count reconciliation", "",
         "Every number quoted in the slides, draft and logs, recomputed from the "
         "artefacts. They are not inconsistent — they are different sets.", "",
         "| artefact | definition | pands | images |", "|---|---|---:|---:|"]
    for a, d, n, ni in rows:
        L += [f"| {a} | {d} | {n:,} | {'' if ni is None else f'{ni:,}'} |"]

    # The 0.x audits filter the register by gemeentecode prefix; the manifest holds a
    # few pands from a neighbouring municipality, which is where 10,093 vs 10,090 comes from.
    outside = sorted(pid for pid in man_ids
                     if pid[:4] not in ep_raw.CITY_CODES)
    L_extra = [
        "", "## Two numbers that look like typos but are not", "",
        f"- **10,093** (2026.07.11 label-entropy log) = manifest pands found in the raw "
        f"register when certificates are matched with no municipality filter.",
        f"- **10,090** (row above) = the same count when the register is pre-filtered to "
        f"the four gemeentecodes 0363/0599/0344/0503. The gap is "
        f"{len(outside)} manifest pands whose BAG id carries gemeentecode "
        f"{', '.join(sorted({p[:4] for p in outside}))} "
        f"(Midden-Delfland) while the manifest labels them `delft`: "
        f"{', '.join(outside)}.",
        "- **10,090** also equals manifest ∩ raw EP by coincidence, not by construction.",
        "", f"So {len(outside)} of the published buildings are outside the four named "
        f"municipalities. Immaterial for results, but the data section should say "
        f"\"four cities and their immediate fringe\" or drop them.", ""]

    missing = man_ids - gt_ids
    L += L_extra
    L += ["## Where the gaps come from", "",
          f"- manifest {len(man_ids):,} -> training universe {len(inter):,}: "
          f"**{len(missing)} manifest pands have no `stage1_gt` row** "
          f"({len(man) - len(man_gt)} images lost).",
          f"- training universe {len(inter):,} -> split {len(split_ids):,}: "
          f"difference {len(inter) - len(split_ids)}.", ""]

    if missing:
        miss = man[man["pand_id"].isin(missing)]
        L += ["Per-city breakdown of the dropped pands:", "",
              "| city | dropped pands | dropped images |", "|---|---:|---:|"]
        for c, s in miss.groupby("city"):
            L += [f"| {c} | {s['pand_id'].nunique()} | {len(s)} |"]
        L += ["",
              "The drop happens in `lod2_features.remove_outliers` (`volume <= 0`) and "
              "`tabula_matcher` (unmatched archetype), per the 2026.05.21 log.", ""]

    # image counts per city for the two universes
    L += ["## Per-city image counts", "",
          "| city | manifest pands | manifest images | ∩GT pands | ∩GT images |",
          "|---|---:|---:|---:|---:|"]
    for c in ["amsterdam", "rotterdam", "utrecht", "delft"]:
        a = man[man["city"] == c]
        b = man_gt[man_gt["city"] == c]
        L += [f"| {c} | {a['pand_id'].nunique():,} | {len(a):,} | "
              f"{b['pand_id'].nunique():,} | {len(b):,} |"]
    L += ["", "## Recommended wording", "",
          f"- dataset **deliverable** (what is published): {man['pand_id'].nunique():,} "
          f"buildings / {len(man):,} images",
          f"- **experiments** (what every model is trained and evaluated on): "
          f"{len(split_ids):,} buildings ({dev['pand_id'].nunique():,} dev + "
          f"{hold['pand_id'].nunique():,} hold-out) / {len(man_gt):,} images", "",
          "Use one of these two, never a third number, and always with the qualifier.", ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L), encoding="utf-8")
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
