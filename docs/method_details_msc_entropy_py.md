# Environment Setup

## Two environments

The workflow uses two conda environments:

| Environment | Purpose | Key packages |
|---|---|---|
| `msc_entropy_py` | Python analysis: atlas construction, scVI, classifier training, plotting | Python 3.11, Scanpy, scVI, scArches, scikit-learn, PyTorch, GSEApy, pySCENIC, RegDiffusion |
| `msc_entropy_r` | R-based trajectory and plotting: Monocle3, Seurat | R 4.3, Monocle3, Seurat 5, scran |

Both freeze files are in `../env/`.

## Setup

```bash
# Python environment
conda env create -f env/msc_entropy_py.yml
conda activate msc_entropy_py

# R environment (for Monocle3 trajectory)
conda env create -f env/msc_entropy_r.yml
conda activate msc_entropy_r
```

The R environment is needed only for the Monocle3 pseudotime step (`run_monocle3_musc_atlas.R`). All other scripts use the Python environment.

## Core package versions (msc\_entropy\_py)

| Package | Version |
|---|---:|
| Python | 3.11 |
| scanpy | 1.11.5 |
| scvi-tools | 1.4.1 |
| scArches | 0.6.1 |
| scikit-learn | 1.8.0 |
| scipy | 1.16.3 |
| anndata | 0.12.0 |
| pandas | 3.0.0 |
| numpy | 2.3.5 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| gseapy | 1.1.11 |
| pyscenic | 0.12.1 |
| regdiffusion | 0.2.0 |
| torch | 2.10.0 |
| pytorch-lightning | 2.6.1 |
| h5py | 3.15.1 |
| numba | 0.63.1 |
| umap-learn | 0.5.11 |
| igraph | 1.0.0 |
| leidenalg | 0.11.0 |
