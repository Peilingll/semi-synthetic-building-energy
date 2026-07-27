# A05 — Building / image count reconciliation

Every number quoted in the slides, draft and logs, recomputed from the artefacts. They are not inconsistent — they are different sets.

| artefact | definition | pands | images |
|---|---|---:|---:|
| `svi_manifest.parquet` | SVI manifest: pands with >=1 accepted Mapillary image | 10,104 | 47,238 |
| manifest ∩ `stage1_gt` | training universe: manifest pands that also have a GT row | 10,086 | 47,150 |
| `stage1_gt.parquet` | four-city pands with an EPC + TABULA match (full stock) | 124,784 |  |
| dev ∪ hold-out | the frozen 80/20 split actually used for modelling | 10,086 |  |
| `dev_fold_indices.parquet` | dev pool (5 folds) | 8,068 |  |
| `holdout_test_pand_ids.parquet` | hold-out | 2,018 |  |
| manifest ∩ raw EP (residential) | manifest pands with >=1 residential certificate | 10,090 |  |
| manifest ∩ raw EP (NTA 8800) | manifest pands with >=1 NTA 8800 certificate | 10,090 |  |

## Two numbers that look like typos but are not

- **10,093** (2026.07.11 label-entropy log) = manifest pands found in the raw register when certificates are matched with no municipality filter.
- **10,090** (row above) = the same count when the register is pre-filtered to the four gemeentecodes 0363/0599/0344/0503. The gap is 3 manifest pands whose BAG id carries gemeentecode 1842 (Midden-Delfland) while the manifest labels them `delft`: 1842100000002754, 1842100000002938, 1842100000002960.
- **10,090** also equals manifest ∩ raw EP by coincidence, not by construction.

So 3 of the published buildings are outside the four named municipalities. Immaterial for results, but the data section should say "four cities and their immediate fringe" or drop them.

## Where the gaps come from

- manifest 10,104 -> training universe 10,086: **18 manifest pands have no `stage1_gt` row** (88 images lost).
- training universe 10,086 -> split 10,086: difference 0.

Per-city breakdown of the dropped pands:

| city | dropped pands | dropped images |
|---|---:|---:|
| amsterdam | 8 | 45 |
| rotterdam | 9 | 42 |
| utrecht | 1 | 1 |

The drop happens in `lod2_features.remove_outliers` (`volume <= 0`) and `tabula_matcher` (unmatched archetype), per the 2026.05.21 log.

## Per-city image counts

| city | manifest pands | manifest images | ∩GT pands | ∩GT images |
|---|---:|---:|---:|---:|
| amsterdam | 8,019 | 36,569 | 8,011 | 36,524 |
| rotterdam | 1,424 | 7,366 | 1,415 | 7,324 |
| utrecht | 488 | 2,233 | 487 | 2,232 |
| delft | 173 | 1,070 | 173 | 1,070 |

## Recommended wording

- dataset **deliverable** (what is published): 10,104 buildings / 47,238 images
- **experiments** (what every model is trained and evaluated on): 10,086 buildings (8,068 dev + 2,018 hold-out) / 47,150 images

Use one of these two, never a third number, and always with the qualifier.
