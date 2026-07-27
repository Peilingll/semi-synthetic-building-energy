# Data-integrity audits (0.x)

Diagnostics run before thesis writing. Nothing here feeds a model; each table answers one
question about whether the data means what the experiments assume.

| table | question | verdict |
|---|---|---|
| `A01_compactheid_source.md` | is `compactheid` BAG geometry or an EP-Online column? | **leak** — EP certificate, per dwelling unit; the clean 3DBAG ratio is worth +0.027 macro-F1, not +0.085 |
| `A02_ep1_ep2.md` | how much of the label can envelope information reach? | envelope-only ceiling ≈ acc 0.71 / macro-F1 0.55 (7-class) |
| `A03_within_cell_labels.md` | does a TABULA cell determine the EPC label? | no — the cell removes 4.7% of label entropy; cell-only oracle macro-F1 0.130 |
| `A04_bin_consistency.md` | is `Energieklasse` exactly `bin(PrimaireFossieleEnergie)`? | 78.4% — the register bins `...EMGForfaitair` (99.985%); our regression target is the other column |
| `A05_count_reconcile.md` | 10,104 vs 10,093 vs 10,090 vs 10,086? | four different sets; use 10,104 (deliverable) / 10,086 (experiments) |
| `A06_archetype_vs_measured.md` | can a demand-vs-measured-EP1 metric detect Stage-1 quality? | **no** — registry-cell archetype beats "predict the stock mean" by only 2.8 kWh/m²·yr, so Stage 1's whole error budget is 1.1–1.9; also confirms the `TH\|NL.01` TABULA row is wrong |

Scripts: `src/audit/a0{1..5}_*.py`, shared certificate cache `src/audit/ep_raw.py`.
Write-up: `doc_processed/log/2026.07.26_audit_0x_leakage_and_label_definition.md`.
