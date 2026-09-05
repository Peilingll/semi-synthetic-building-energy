# A05 - Building and image count reconciliation

Every number quoted in the slides, draft, and logs was recomputed from the artefacts. The counts refer to different sets.

| artefact | definition | pands | images |
|---|---|---:|---:|
| `svi_manifest.parquet` | SVI manifest: pands with >=1 accepted Mapillary image | 10,104 | 47,238 |
| manifest intersection `stage1_gt` | experimental dataset: manifest pands that also have a reference record | 10,086 | 47,150 |
| `stage1_gt.parquet` | four-city pands with an EPC + TABULA match (full stock) | 124,784 |  |
| development plus holdout | predefined 80/20 building split used for modelling | 10,086 |  |
| `dev_fold_indices.parquet` | development set (5 folds) | 8,068 |  |
| `holdout_test_pand_ids.parquet` | fixed holdout set | 2,018 |  |
| manifest ∩ raw EP (residential) | manifest pands with >=1 residential certificate | 10,090 |  |
| manifest ∩ raw EP (NTA 8800) | manifest pands with >=1 NTA 8800 certificate | 10,090 |  |

## Historical raw-register counts

- **10,093** was reported by the historical 2026.07.11 label consistency run for manifest pands found in the raw register when certificates were matched without a municipality filter.
- **10,090** (row above) = the same count when the register is pre-filtered to the four gemeentecodes 0363/0599/0344/0503. The gap is 3 manifest pands whose BAG id carries gemeentecode 1842 (Midden-Delfland) while the manifest labels them `delft`: 1842100000002754, 1842100000002938, 1842100000002960.
- **10,090** also equals manifest ∩ raw EP by coincidence, not by construction.

The thesis therefore uses the term "four Dutch study areas" rather than claiming that every building falls strictly within the four named municipalities.

The current certificate consistency analysis is restricted to the 10,086-building experimental dataset. Of these buildings, 10,075 can be linked back to at least one raw residential certificate. The historical 10,093 count is not used in the thesis results.

## Where the gaps come from

- manifest 10,104 -> experimental dataset 10,086: **18 manifest pands have no `stage1_gt` row** (88 images lost).
- experimental dataset 10,086 -> split 10,086: difference 0.

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
- **experimental dataset**: 10,086 buildings (8,068 development + 2,018 holdout) / 47,150 images

Use one of these two, never a third number, and always with the qualifier.
