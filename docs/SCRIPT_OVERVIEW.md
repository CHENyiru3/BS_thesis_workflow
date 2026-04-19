# Script Overview

Maps repository scripts to the thesis workflow.
Emphasis is on scripts that support the final dissertation narrative.

## Path convention

Use relative paths from the repository root:

```
scripts/   analysis entry points and figure-generation scripts
docs/      workflow documentation
env/       conda environment freeze files
```

Some scripts depend on external raw-data paths not committed here.
Update data paths through script arguments or configuration rather than hard-coded paths.

## Mouse core workflow

### `mouse_workflow_core.py`

Mouse workflow backbone.
Handles dataset loading, metadata harmonization, donor linkage-aware partitioning, metacell generation, donor-aware training and evaluation, and artifact writing.

### `mouse_aging_map.py`

Purified and annotated mouse MuSC atlas builder.
Implements iterative scVI integration, cluster cleanup, UMAP generation, and MuSC state annotation.

### `clock_system_mouse_updated.py`

Primary mouse age-state classifier driver.
Includes baseline training, donor-disjoint evaluation, explicit source holdout, and threshold-transfer diagnostics.

### `run_monocle3_musc_atlas.R`

Monocle3 trajectory bridge.
Reads the MuSC `.h5ad`, builds the Monocle3 CDS, and exports pseudotime outputs.

### `run_scvi_reference_verify_prepared.py`

scVI and scArches reference-mapping workflow for the local pseudotime-aware extension.
Handles fixed-atlas loading, query mapping into the atlas latent space, reference pseudotime transfer, local-model scoring, and mapped-query summaries.

### `plot_scvi_post_injury_local.py`

Post-injury local-model output summarizer.
Generates sample tables, age-group-by-timepoint summaries, and baseline-versus-local comparison figures.

### `plot_musc_verification_overview.py`

Compact overview plotting for the post-injury verification dataset.
Generates dataset-composition overviews, donor-group pseudotime summaries, and post-injury overview figures.

### `run_mouse_clock_supplementary_trajectory.py`

Supplementary trajectory script for the baseline mouse classifier.
Reapplies the baseline model to verification MuSC data and generates pseudotime summaries of age-state scores.

### `post_mouse_anlaysis.py`

Mouse downstream biological interpretation workflow (filename typo preserved from source).
Performs state-stratified young-versus-old differential expression, GO enrichment, GRN inference within each MuSC state, and regulon comparison.

## Human workflow

### `human_clock_core.py`

Human workflow backbone.

### `clock_system_human.py`

Human age-state classifier driver for the proof-of-concept branch.

### `post_human_analysis.py`

Human downstream interpretation script.
Performs whole-atlas young-versus-old differential expression, GO enrichment, GRN inference, and regulon comparison.

## Figure generation

### `code_for_vislaization.py`

Central thesis figure-generation script (filename typo preserved from source).
Covers atlas figures, pseudotime figures, baseline and local mouse classifier figures, post-injury figures, human proof-of-concept figures, and downstream biological interpretation figures.

## Notes

- The baseline age-state classifier from `clock_system_mouse_updated.py` is the primary model described in the thesis.
- The local pseudotime-aware branch is an interpretive extension, not the main benchmark model.
- The human branch is a proof-of-concept extension, not a full-scale benchmark.
