# Mouse MuSC Atlas Collection and Preprocessing

## Overview

This document describes the data collection and preprocessing workflow used to assemble the mouse MuSC training atlas.
The workflow combines seven public `.h5ad` source datasets, harmonizes gene identifiers and metadata, isolates MuSC populations, links repeated donors across sources, and refines the merged atlas through iterative purification.

## Source Datasets

Seven public mouse scRNA-seq datasets were collected:

| Source alias | Reported ages in source study |
|---|---|
| `walter2024_main` | Approximately 5, 20, and 26 months across 0--7 dpi |
| `GSE226907_wt` | Young 3--4 months; old greater than 20 months at 7 dpi |
| `GSE150366_noninj` | Timepoint-annotated MuSC dataset; retained cells carried unresolved age labels |
| `SKM_mouse_raw` | Binary age labels mapped to young and old |
| `TabulaMuris_limb_10x` | 3m, 18m, and 24m |
| `TabulaMuris_limb_smartseq2` | 3m, 18m, and 24m |
| `TabulaMuris_diaphragm_smartseq2` | 3m, 18m, and 24m |

After MuSC filtering and age restriction, six sources contributed to the trainable atlas.
`GSE150366_noninj` was retained biologically but excluded from training because its MuSC cells had unresolved `unknown_age` labels.

## Preprocessing Pipeline

### 1. Raw dataset loading

Source files are loaded through `step_load_mouse_datasets()` in [`../scripts/mouse_workflow_core.py`](../scripts/mouse_workflow_core.py).

### 2. Gene and age harmonization

Each source is checked for gene symbol vs Ensembl ID format.
Tabula Muris sources are converted from Ensembl IDs to gene symbols using a configured mouse mapping table.
After conversion, gene names are uppercased and deduplicated with `var_names_make_unique(join="first")`.

Age metadata are standardized into a shared `Age_group` field:

- `SKM_mouse_raw`: `Age_bin` mapped to `young` or `old`
- Tabula Muris sources: age strings (`3m`, `18m`, `24m`) parsed to months, then binarized at 18 months
- `GSE150366_noninj`: cells retained `unknown_age` and were excluded from training

### 3. Source-specific MuSC filtering

MuSC filtering uses dataset-specific rules because source files do not share a cell-type vocabulary:

- `walter2024_main`: `celltype` contains `MuSC` or `Myoblast`
- `GSE226907_wt`: `celltype == "MuSC"`
- `GSE150366_noninj`: `celltype` contains `MuSC`
- `SKM_mouse_raw`: `annotation == "MuSC"`
- Tabula Muris sources: `cell_type == "skeletal muscle satellite cell"`

### 4. Metadata standardization

After filtering, all sources are mapped to a shared schema:

- `sample_id_std`, `Sex_std`, `celltype_std`, `Age_group_std`, `source`

### 5. Common-gene intersection and age restriction

All sources are concatenated on their shared gene set, then restricted to cells with `Age_group_std` in `{young, old}`.
This yields the harmonized training atlas with **11,251 cells** and a common gene space.

### 6. Donor linkage across sources

Two donor identifiers are constructed:

- `donor_bootstrap_id = source::sample_id_std`
- `donor_split_id` — linkage-aware: linked samples use the prefix `LINK::` to prevent the same biological donor from appearing in both training and test partitions across sources

In the current atlas: **61 unique sample identifiers**, of which **8 recur across two sources** (all from the paired Tabula Muris Smart-seq2 limb and diaphragm subsets).

## Atlas Purification

The purification workflow is implemented in [`../scripts/mouse_aging_map.py`](../scripts/mouse_aging_map.py).
This stage starts from the harmonized training atlas, not from the raw source files.

### Cleaning steps

1. Retain only MuSC-like labels (`MuSC`, `skeletal muscle satellite cell`, `Quiescent_MuSCs`, `Activated_MuSCs`).
2. Remove mitochondrial, ribosomal, metallothionein, and heat-shock genes.
3. **Round 1:** compute HVGs, train scVI, Leiden clustering — remove clusters inconsistent with the MuSC program.
4. **Round 2:** retrain on the filtered atlas, recluster, remove remaining non-MuSC clusters.
5. Recompute HVGs while forcing canonical marker genes to remain available.
6. Retrain the final scVI manifold, generate UMAP and Leiden solution, and apply MuSC subtype annotations.
7. Export purified and annotated atlas outputs, including `annotated_atlas.h5ad`.

This yields the **10,329-cell purified reference manifold** used for all downstream biological interpretation and classifier training.

### Reproducibility artifacts

The cleaning workflow records:

- Deterministic seeds propagated to Python, NumPy, PyTorch, and `scvi.settings.seed`
- Run metadata in `run_manifest.json`
- Pass-specific `.h5ad` snapshots: `input_after_qc`, `pass1_before_filter`, `pass1_after_filter`, `pass2_before_filter`, `pass2_after_filter`, `final_purified_atlas`, `final_annotated_atlas`
