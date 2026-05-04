# `data/processed/` — Data Dictionary

This folder holds outputs from two pipeline generations. Each city has its own subfolder.
**Delft** uses the original three-source pipeline (BAG + 3D BAG + EP-Online).
**Amsterdam / Utrecht / Rotterdam** use a lighter two-source pipeline (BAG + EP-Online), since the OpenFACADES VLM track only needs BAG attributes + EP labels + TABULA archetypes (no LOD2 geometry).

The Traditional Chinese version of this document is archived at
`doc_processed/README.zh-TW.md`.

---

## Layout

```
data/processed/
├── tabula_nl.csv                             # NL-wide TABULA lookup (24 archetypes) — shared
├── vlm_predictions.csv                       # MVP manual VLM (35 buildings)
├── vlm_predictions_full_openfacades.csv      # Phase 2 OpenFACADES per-image
├── vlm_predictions_vote_openfacades.csv      # Phase 2 OpenFACADES per-pand_id vote aggregate
├── vlm_ground_truth.csv                      # Phase 2 ground truth
├── vlm_phase2_eval.json                      # Phase 2 evaluation report
├── mvp_testing_data/                         # MVP test images (35 PNGs)
├── README.md                                 # this file
├── delft/                                    # 3-source pipeline (BAG + 3D BAG + EP-Online)
│   ├── bag_3dbag_ep_joined.parquet
│   ├── residential_with_3d_features.parquet
│   └── residential_tabula_matched.parquet
├── amsterdam/                                # 2-source pipeline (BAG + EP-Online)
│   ├── bag_ep_joined.parquet
│   ├── residential_tabula_matched.parquet
│   ├── step1.log                             # run log (audit trail)
│   └── step3.log
├── utrecht/
│   ├── bag_ep_joined.parquet
│   ├── residential_tabula_matched.parquet
│   ├── step1.log
│   └── step3.log
└── rotterdam/
    ├── bag_ep_joined.parquet
    ├── residential_tabula_matched.parquet
    ├── step1.log
    └── step3.log
```

---

## Pipeline differences

| Dimension | 3-source pipeline (Delft) | 2-source pipeline (other cities) |
|---|---|---|
| Sources | BAG + 3D BAG + EP-Online | BAG + EP-Online |
| Config | `config.yaml` (project root) | `configs/<city>.yaml` |
| `pipeline.use_3dbag` | `true` (default) | `false` |
| Step 2 (LOD2 features) | Runs (produces `residential_with_3d_features.parquet`) | **Skipped** |
| Main joined file | `bag_3dbag_ep_joined.parquet` | `bag_ep_joined.parquet` |
| `b3_*` 3D fields | Yes | No |
| TABULA-matched output | Includes LOD2 features (volume, envelope_area, …) | Just BAG + EP + TABULA |

---

## File schemas

### `delft/bag_3dbag_ep_joined.parquet` (3-source)

Produced by `src.data_loader.run_step1()` driven by the root `config.yaml`.

Key columns:

- `pand_id` (16-digit zero-padded string, primary key)
- BAG: `identificatie`, `bouwjaar`, `status`, `gebruiksdoel`, `Gebouwtype`, `geometry` (RD)
- 3D BAG: `b3_volume_lod22`, `b3_opp_buitenmuur`, `b3_opp_dak_plat`, `b3_opp_dak_schuin`, `b3_opp_grond`, `b3_h_max`, `b3_h_maaiveld`, `b3_bouwlagen`, `b3_rmse_lod22`, …
- EP-Online: `Energieklasse`, `EnergieIndex`, `Bouwjaar`, `Postcode`, `Status`, `Registratiedatum`
- Derived: `build_period` (50-year viz bin defined in `data_loader.py:316-322`)

### `delft/residential_with_3d_features.parquet` (3-source)

Produced by `src.lod2_features.run_step2()`. Filtered residential subset with derived LOD2 features.

Key columns: `pand_id`, `volume`, `envelope_area`, `shape_factor`, `building_height`, `num_floors_estimated`, `floor_area_estimated`, `lod2_quality_flag`.

### `<city>/residential_tabula_matched.parquet`

Produced by `src.tabula_matcher.run_step3()`.

- **Delft (3-source)**: starts from Step 2 LOD2 features, merges in BAG `Gebouwtype` + `bouwjaar`, then matches TABULA.
- **Other cities (2-source)**: starts from `<city>/bag_ep_joined.parquet` directly, skips LOD2 merge, then matches TABULA.

Both add:

- `tabula_building_type` ∈ {SFH, TH, MFH, AB} (from `GEBOUWTYPE_MAP`, `tabula_matcher.py:17-24`)
- `tabula_period` ∈ {NL.01, NL.02, NL.03, NL.04, NL.05, NL.06} (from `TABULA_PERIODS`, `tabula_matcher.py:30-37`)
- `u_wall`, `u_roof`, `u_floor`, `u_window` (from `tabula_nl.csv` lookup)

### `tabula_nl.csv`

24 NL TABULA archetypes (4 building types × 6 construction periods) with U-values. **Shared across all cities.**

### `<city>/bag_ep_joined.parquet` (2-source)

Produced by `src.data_loader.run_step1_bag_ep()` driven by `configs/<city>.yaml`. Same schema as `bag_3dbag_ep_joined.parquet` minus all `b3_*` fields and `geometry_3dbag`.

### `vlm_*` files (legacy)

VLM evaluation artefacts; unrelated to the new-city pipelines, kept for existing notebooks:

- `vlm_predictions.csv` — Phase 1 MVP (35 manually-labelled buildings)
- `vlm_predictions_full_openfacades.csv` — Phase 2 OpenFACADES raw per-image
- `vlm_predictions_vote_openfacades.csv` — Phase 2 OpenFACADES per-pand_id vote aggregate
- `vlm_ground_truth.csv` — Phase 2 ground truth
- `vlm_phase2_eval.json` — Phase 2 evaluation metrics

---

## TABULA period codes (`src/tabula_matcher.py:30-37`)

| Code | Construction year |
|---|---|
| NL.01 | ≤ 1964 |
| NL.02 | 1965 – 1974 |
| NL.03 | 1975 – 1991 |
| NL.04 | 1992 – 2005 |
| NL.05 | 2006 – 2014 |
| NL.06 | ≥ 2015 |

Source: EU EPISCOPE / TABULA Netherlands country dossier. The same six bands apply to all four cities.

---

## Postcode prefixes per city

| City | Postcode prefixes |
|---|---|
| Delft | `26` |
| Amsterdam | `10`, `11` |
| Utrecht | `35` |
| Rotterdam | `30` |

EP-Online filtering is done by `src/data_loader.py:load_ep_online()` via `Postcode.str.startswith(tuple(prefixes))`.

---

## How to regenerate

### Delft (3-source)

```bash
uv run python -m src.data_loader      # reads config.yaml
uv run python -m src.lod2_features
uv run python -m src.tabula_matcher
```

> **Note:** the root `config.yaml` currently writes to top-level paths
> (`data/processed/bag_3dbag_ep_joined.parquet`, etc.). The Delft files have
> been **manually moved into `data/processed/delft/`**; re-running the
> existing config would write back to the top level. Either move the
> regenerated files into `delft/`, or update `config.yaml`'s `data_paths`
> to point at `data/processed/delft/...` before re-running.

### Other cities (2-source)

```bash
uv run python -m src.data_loader     --config configs/amsterdam.yaml
uv run python -m src.tabula_matcher  --config configs/amsterdam.yaml

uv run python -m src.data_loader     --config configs/utrecht.yaml
uv run python -m src.tabula_matcher  --config configs/utrecht.yaml

uv run python -m src.data_loader     --config configs/rotterdam.yaml
uv run python -m src.tabula_matcher  --config configs/rotterdam.yaml
```

### Derive bboxes (one-off)

```bash
uv run python scripts/derive_city_bboxes.py
```

Paste the printed `bbox_rd` and `bbox_wgs` values into the corresponding `configs/<city>.yaml`.

---

## Source URLs

- BAG WFS: `https://service.pdok.nl/lv/bag/wfs/v2_0` (layer `bag:pand`). PDOK enforces a hard ~51k-feature cap per bbox query, so the new-city pipeline uses `fetch_bag_pand_tiled()` to slice the bbox into 2 km tiles, then dedupes (`src/data_loader.py`). Delft's bbox is small enough not to need tiling.
- 3D BAG WFS: `https://data.3dbag.nl/api/BAG3D/wfs` (layer `BAG3D:lod12`) — only used by the 3-source pipeline.
- EP-Online: local CSV at `data/raw/v20260401_v4_csv/v20260401_v4_csv.csv` (snapshot dated 2026-04-01).
- Administrative boundaries WFS: `https://service.pdok.nl/kadaster/bestuurlijkegebieden/wfs/v1_0` (layer `bestuurlijkegebieden:Gemeentegebied`).

---

## Row counts (refresh after each rerun)

| File | Rows | Last updated |
|---|---|---|
| `delft/bag_3dbag_ep_joined.parquet` | ~3,500 | 2026-04 |
| `delft/residential_with_3d_features.parquet` | ~3,495 | 2026-04 |
| `delft/residential_tabula_matched.parquet` | ~3,495 (match rate 99.97%) | 2026-04 |
| `amsterdam/bag_ep_joined.parquet` | 63,785 | 2026-05-04 |
| `amsterdam/residential_tabula_matched.parquet` | 63,436 (match rate 99.45%; 349 unmapped: 348 Logieswoning + 1 Woonboot) | 2026-05-04 |
| `utrecht/bag_ep_joined.parquet` | 20,060 | 2026-05-04 |
| `utrecht/residential_tabula_matched.parquet` | 20,059 (match rate 100.00%; 1 unmapped: Logieswoning) | 2026-05-04 |
| `rotterdam/bag_ep_joined.parquet` | 38,037 | 2026-05-04 |
| `rotterdam/residential_tabula_matched.parquet` | 38,036 (match rate 100.00%; 1 unmapped: Woonboot) | 2026-05-04 |
