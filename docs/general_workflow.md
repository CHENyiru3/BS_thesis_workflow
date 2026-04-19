# Mouse MuSC Age-State Classifier Workflow

This document summarizes the mouse workflow used in the thesis.
It follows the actual analysis structure in the repository and keeps the emphasis on biological inputs, model transformations, and interpretable outputs.

The current workflow starts from processed single-cell expression objects in `.h5ad` format.
Upstream study-specific steps such as alignment, ambient RNA correction, and doublet removal are not documented here as part of the core thesis workflow.

## Phase 1

**Phase name:** Cross-study MuSC harmonization and donor-aware atlas assembly

**Input:** Multiple mouse skeletal muscle single-cell RNA-seq datasets from independent studies, each containing donor metadata, age labels, and study-specific gene identifiers.

**Transformation:** MuSC populations are isolated from each dataset, gene identifiers are mapped into a unified symbol space, age metadata are standardized, and all studies are merged on shared genes.
Donor identities are then reconciled across studies to build a linkage-aware, donor-disjoint training atlas that prevents cross-source information leakage.

**Core packages / tools:** AnnData and Scanpy for dataset handling and matrix operations, with custom harmonization logic in `scripts/mouse_workflow_core.py` for MuSC filtering, metadata standardization, shared-gene intersection, and donor linkage-aware partitioning.

**Output:** A harmonized MuSC atlas with standardized donor and age metadata, a shared gene space, and preserved raw-count layers for downstream atlas refinement and age-state modelling.

## Phase 2

**Phase name:** Iterative MuSC atlas purification and state annotation

**Input:** The harmonized MuSC training atlas with raw counts and donor identities.

**Transformation:** Highly variable genes are embedded with scVI to learn a donor-corrected latent manifold.
The atlas is iteratively reclustered, clusters inconsistent with the MuSC program are removed, and the latent manifold is rebuilt to enrich for bona fide MuSC states.
The purified manifold is then visualized with UMAP and annotated using canonical myogenic marker programs.

**Core packages / tools:** Scanpy for highly variable gene selection, neighborhood graph construction, Leiden clustering, UMAP, and marker-based visualization, and scvi-tools for donor-corrected latent representation learning.

**Output:** A purified and annotated MuSC trajectory atlas with donor-corrected latent coordinates, UMAP embeddings, and biologically defined states such as quiescent, activated, and differentiating/proliferating MuSCs.

## Phase 3

**Phase name:** Donor-aware metacell age-state classifier training and benchmark evaluation

**Input:** The purified mouse MuSC atlas and donor-level age labels.

**Transformation:** Cells are aggregated into bootstrap metacells within donor partitions to stabilize expression estimates and reduce single-cell noise.
The metacell matrix is filtered, log-normalized, restricted to highly variable genes, and used to train an Elastic-Net logistic regression classifier that separates young-like and old-like transcriptional states.
Performance is evaluated with donor-disjoint validation and explicit external source-holdout analysis.

**Core packages / tools:** Custom bootstrap metacell generation in `abclock.metacells`, Scanpy for metacell preprocessing, scikit-learn for Elastic-Net logistic regression and diagnostics, and donor-aware validation logic in `scripts/mouse_workflow_core.py`.

**Output:** A baseline mouse MuSC age-state classifier defined by weighted genes, metacell-level scores, and donor-disjoint benchmark summaries.

## Phase 4

**Phase name:** Local pseudotime-aware extension and post-injury mapping

**Input:** The annotated MuSC atlas, exported pseudotime coordinates, and post-injury verification data.

**Transformation:** A local extension reweights classifier behaviour across overlapping pseudotime windows instead of assuming a single global gene-weight pattern.
The reference atlas is also used for query mapping and exploratory post-injury scoring.

**Core packages / tools:** Monocle3 for pseudotime inference, scikit-learn for local Elastic-Net models, scVI and scArches for reference mapping, and plotting utilities in the mouse verification scripts.

**Output:** A pseudotime-aware extension used for interpretation and exploratory post-injury application, rather than as a replacement for the baseline classifier.

## Phase 5

**Phase name:** State-resolved biological interpretation and human proof-of-concept extension

**Input:** The annotated mouse atlas, fitted classifier coefficients, age-stratified MuSC subsets, and the retained human MuSC dataset.

**Transformation:** Young and old MuSC populations are compared within biological states using differential expression, GO enrichment, and regulatory-network analysis.
A smaller human branch applies the same general logic to a retained proof-of-concept cohort rather than a full benchmark setting.

**Core packages / tools:** Scanpy for state-resolved differential expression, GSEApy for enrichment analysis, RegDiffusion and pySCENIC for regulatory summaries, and the human workflow scripts for qualitative transfer.

**Output:** State-resolved aging programs, regulatory summaries, and a proof-of-concept human extension aligned with the thesis conclusions.
