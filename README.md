# Semi-Synthetic Building Energy Dataset

An automated pipeline that combines Dutch public building data (BAG, 3D BAG, EP-Online), street view visual features (OpenFACADES), and TABULA energy archetypes to produce semi-synthetic building energy datasets for machine learning.

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Setup

```bash
git clone <repo-url>
cd semi-synthetic-building-energy
uv sync
```

### Run Step 1 Pipeline

```bash
uv run python -m src.data_loader
```

This fetches BAG + 3D BAG data via WFS APIs and joins with local EP-Online data for the Delft study area. Output: `data/processed/bag_3dbag_ep_joined.parquet`

### Explore Results

Open `notebooks/01_data_exploration.ipynb` in VS Code, select the `.venv` kernel, and run all cells.

## Project Structure

```
├── config.yaml              # Study area, WFS endpoints, filter parameters
├── pyproject.toml            # Dependencies (managed by uv)
├── src/
│   ├── config.py             # Load config.yaml
│   ├── data_loader.py        # Step 1: BAG + 3D BAG + EP-Online join
│   ├── lod2_features.py      # Step 2: LOD2 geometry features (TODO)
│   ├── tabula_matcher.py     # Step 3: TABULA archetype matching (TODO)
│   ├── dataset_builder.py    # Final dataset assembly (TODO)
│   └── evaluation.py         # Step 4: Ablation study (TODO)
├── notebooks/
│   └── 01_data_exploration.ipynb
├── data/
│   ├── raw/                  # EP-Online CSV (not tracked)
│   ├── processed/            # Intermediate outputs (not tracked)
│   └── output/               # Final dataset (not tracked)
├── spec_doc/                 # Design specifications
└── doc/log/                  # Development logs
```

## Data Sources

| Source | Method | Key Fields |
|--------|--------|------------|
| [BAG](https://www.pdok.nl/) | PDOK WFS API | pand_id, bouwjaar, geometry |
| [3D BAG](https://3dbag.nl/) | 3D BAG WFS API | roof type, height, volume, surface areas |
| [EP-Online](https://www.ep-online.nl/) | Local CSV download | energy label (A-G), energy index |

## MVP Pipeline Status

| Step | Description | Status |
|------|-------------|--------|
| 1 | Data integration (BAG + 3D BAG + EP-Online) | Done |
| 2 | LOD2 geometry features | TODO |
| 3 | TABULA archetype matching | TODO |
| 4 | Random Forest prediction + ablation | TODO |
