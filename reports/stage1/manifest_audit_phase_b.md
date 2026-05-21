# SVI Manifest Audit (Stage 1 Phase B)

- Manifest: `svi_manifest.parquet`
- Total images: 47,238
- Unique buildings (pand_id): 10,104

## Per-city totals (full manifest)

| City | Buildings | Images | Mean/bldg | Median/bldg |
|---|---:|---:|---:|---:|
| amsterdam | 8,019 | 36,569 | 4.56 | 4.0 |
| delft | 173 | 1,070 | 6.18 | 8.0 |
| rotterdam | 1,424 | 7,366 | 5.17 | 5.0 |
| utrecht | 488 | 2,233 | 4.58 | 4.0 |
| **total** | **10,104** | **47,238** | **4.68** | **4.0** |

## Image-count distribution per pand_id (full manifest)

| n_images | Buildings | % |
|---|---:|---:|
| 1 | 1,172 | 11.6% |
| 2-4 | 4,028 | 39.9% |
| 5-7 | 2,320 | 23.0% |
| 8 (cap) | 2,584 | 25.6% |
| **total** | **10,104** | 100.0% |

## Manifest ∩ GT (training universe)

- Buildings in both manifest and Stage 1 GT: **10,086**
- Manifest buildings without GT row (filtered out): 18

### Per-city training universe

| City | Buildings | Images | Mean/bldg |
|---|---:|---:|---:|
| amsterdam | 8,011 | 36,524 | 4.56 |
| delft | 173 | 1,070 | 6.18 |
| rotterdam | 1,415 | 7,324 | 5.18 |
| utrecht | 487 | 2,232 | 4.58 |
| **total** | **10,086** | **47,150** | **4.67** |

### building_type composition (training universe)

| City | SFH | TH | MFH | AB | Total |
|---|---:|---:|---:|---:|---:|
| amsterdam | 17 | 394 | 41 | 7559 | 8011 |
| delft | 8 | 53 | 0 | 112 | 173 |
| rotterdam | 93 | 339 | 2 | 981 | 1415 |
| utrecht | 14 | 206 | 3 | 264 | 487 |
| **total** | **132** | **992** | **46** | **8916** | **10086** |
