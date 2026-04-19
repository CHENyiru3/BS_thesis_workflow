# Legacy exploratory notebook cells with machine-specific paths were removed to
# keep this script repository-relative and portable.
import os
import random

os.environ.setdefault("PYTHONHASHSEED", "43")

import scvi
import torch
import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import phate
import matplotlib.pyplot as plt
from matplotlib import rcParams
from path_config import ARTIFACT_ROOT, PROJECT_ROOT

# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
ATLAS_PATH = str(ARTIFACT_ROOT / "clock_artifacts" / "training_atlas_trainonly_raw.h5ad")
MUSC_RANDOM_SEED = 43
MUSC_N_HVG = 2000
MUSC_SCVI_N_LATENT = 25
MUSC_SCVI_N_LAYERS = 2
MUSC_SCVI_MAX_EPOCHS = 400
MUSC_LEIDEN_RESOLUTION = 0.6
MUSC_UMAP_MIN_DIST = 0.1
MUSC_N_NEIGHBORS = 30

# Output Directories Configuration
MUSC_OUTPUT_DIR = Path(ATLAS_PATH).resolve().parent / "musc_annotation"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = MUSC_OUTPUT_DIR / "runs" / RUN_ID
DIR_PASS1 = RUN_DIR / "pass1"
DIR_PASS2 = RUN_DIR / "pass2"
DIR_INPUT = RUN_DIR / "00_input"
DIR_EXPLORATORY = RUN_DIR / "01_Exploratory"
DIR_PUB_MAIN = RUN_DIR / "02_Main_Figures"
DIR_PUB_SUPP = RUN_DIR / "03_Supplementary_Figures"

for d in [RUN_DIR, DIR_INPUT, DIR_PASS1, DIR_PASS2, DIR_EXPLORATORY, DIR_PUB_MAIN, DIR_PUB_SUPP]:
    d.mkdir(parents=True, exist_ok=True)

MUSC_KEEP_LABELS = {
    "MuSC",
    "skeletal muscle satellite cell",
    "Quiescent_MuSCs",
    "Activated_MuSCs",
}

MUSC_SUBTYPES = {
    "Deep_Quiescent_MuSC": ["Pax7", "Myf5", "Gpx3", "Ryr3", "Cd34"],
    "Primed_Quiescent_MuSC": ["Pax7", "Myf5", "Gpx3", "Junb", "Hes1"],
    "Early_Activated": ["Pax7", "Myf5", "Gpx3", "Anxa1", "Gpx1"],
    "Late_Activation": ["Pax7", "Myf5", "Anxa1", "Gpx1"],
    "Proliferating_MuSC": ["Mki67", "Cenpa"],
    "Myod1_high_MuSC": ["Myod1", "Cdkn1c", "Notch1", "Anxa2"],
    "Myog_high_MuSC": ["Myog", "Actc1"],
    "Renewal": ["Cdk6", "Ccnd1", "Tgfbr3", "Smad4"],
}

phate_markers = {
    "Quiescence": ["Pax7", "Myf5"],
    "Activation": ["Myod1", "Anxa1"],
    "Proliferation": ["Mki67", "Cenpa"],
    "Differentiation": ["Myog", "Actc1"],
}

all_markers = set(gene for genes in MUSC_SUBTYPES.values() for gene in genes)
all_markers.update(gene for genes in phate_markers.values() for gene in genes)
RUN_SUMMARY_ROWS = []

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def _sanitize_dataframe_for_h5ad(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean.index = clean.index.astype(object)
    clean.columns = pd.Index([str(col) for col in clean.columns], name=clean.columns.name)
    for col in clean.columns:
        series = clean[col]
        if str(series.dtype).startswith("string") or isinstance(getattr(series, "_values", None), pd.arrays.ArrowStringArray):
            clean[col] = series.astype(object)
            continue
        if isinstance(series.dtype, pd.CategoricalDtype):
            cats = series.cat.categories
            if str(cats.dtype).startswith("string") or isinstance(getattr(cats, "_values", None), pd.arrays.ArrowStringArray):
                clean[col] = series.cat.rename_categories(cats.astype(object))
    return clean

def write_h5ad_safe(adata: sc.AnnData, output_path: str | Path, **kwargs) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_adata = adata.copy()
    safe_adata.obs = _sanitize_dataframe_for_h5ad(safe_adata.obs)
    safe_adata.var = _sanitize_dataframe_for_h5ad(safe_adata.var)
    safe_adata.obs_names = safe_adata.obs.index.copy()
    safe_adata.var_names = safe_adata.var.index.copy()
    
    # Safely sanitize raw.var to fix the PyArrow write error
    if safe_adata.raw is not None:
        clean_raw_var = _sanitize_dataframe_for_h5ad(safe_adata.raw.var)
        temp_raw = sc.AnnData(X=safe_adata.raw.X, var=clean_raw_var)
        safe_adata.raw = temp_raw

    kwargs.setdefault("convert_strings_to_categoricals", False)
    safe_adata.write_h5ad(output_path, **kwargs)
    return output_path


def _filter_marker_dict(var_names: pd.Index, marker_dict: dict[str, list[str]]) -> dict[str, list[str]]:
    var_name_set = set(var_names.astype(str))
    return {label: [gene for gene in genes if gene in var_name_set] for label, genes in marker_dict.items()}


def _resolve_pass_rep_key(adata: sc.AnnData, cluster_key: str) -> str:
    if cluster_key == "leiden_pass1" and "X_scVI_pass1" in adata.obsm:
        return "X_scVI_pass1"
    if cluster_key == "leiden_pass2" and "X_scVI_pass2" in adata.obsm:
        return "X_scVI_pass2"
    if "X_scVI" in adata.obsm:
        return "X_scVI"
    raise ValueError(f"No compatible scVI representation found for {cluster_key}.")


def _write_run_manifest() -> None:
    manifest = {
        "run_id": RUN_ID,
        "atlas_path": ATLAS_PATH,
        "seed": MUSC_RANDOM_SEED,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_deterministic_algorithms": True,
        "n_hvg": MUSC_N_HVG,
        "n_latent": MUSC_SCVI_N_LATENT,
        "n_layers": MUSC_SCVI_N_LAYERS,
        "max_epochs": MUSC_SCVI_MAX_EPOCHS,
        "leiden_resolution": MUSC_LEIDEN_RESOLUTION,
        "n_neighbors": MUSC_N_NEIGHBORS,
        "passes": RUN_SUMMARY_ROWS,
    }
    with open(RUN_DIR / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def set_global_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    scvi.settings.seed = int(seed)


def save_pass_snapshot(
    adata: sc.AnnData,
    output_dir: Path,
    snapshot_name: str,
    note: str,
) -> Path:
    snapshot_path = output_dir / f"{snapshot_name}.h5ad"
    write_h5ad_safe(adata, snapshot_path)
    meta_path = output_dir / f"{snapshot_name}.json"
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "snapshot": snapshot_name,
                "path": str(snapshot_path),
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "note": note,
            },
            handle,
            indent=2,
        )
    return snapshot_path


def save_pass_evidence(
    adata: sc.AnnData,
    cluster_key: str,
    output_dir: Path,
    pass_name: str,
    description: str,
    removed_clusters: list[str] | None = None,
) -> None:
    removed_clusters = [str(x) for x in (removed_clusters or [])]
    evidence_adata = adata.copy()

    # Standardize gene names so marker dictionaries and plots use one casing.
    evidence_adata.var_names = evidence_adata.var_names.str.capitalize()

    # Use log-normalized expression on the temporary object for DE and dotplots.
    sc.pp.normalize_total(evidence_adata)
    sc.pp.log1p(evidence_adata)

    evidence_adata.obs[cluster_key] = evidence_adata.obs[cluster_key].astype("category")
    rep_key = _resolve_pass_rep_key(evidence_adata, cluster_key)
    sc.pp.neighbors(
        evidence_adata,
        use_rep=rep_key,
        random_state=MUSC_RANDOM_SEED,
        n_neighbors=MUSC_N_NEIGHBORS,
    )
    sc.tl.umap(evidence_adata, random_state=MUSC_RANDOM_SEED)

    old_figdir = sc.settings.figdir
    sc.settings.figdir = str(output_dir)
    try:
        sc.pl.umap(
            evidence_adata,
            color=cluster_key,
            legend_loc="on data",
            title=f"{pass_name} clusters",
            save=f"_{pass_name}_clusters_umap.pdf",
            show=False,
        )

        pass_markers = _filter_marker_dict(evidence_adata.var_names, MUSC_SUBTYPES)
        pass_markers = {label: genes for label, genes in pass_markers.items() if genes}
        sc.tl.rank_genes_groups(evidence_adata, groupby=cluster_key, method="wilcoxon")
        if pass_markers:
            sc.tl.dendrogram(evidence_adata, groupby=cluster_key)
            sc.pl.dotplot(
                evidence_adata,
                pass_markers,
                groupby=cluster_key,
                standard_scale="var",
                cmap="Reds",
                dendrogram=True,
                save=f"_{pass_name}_canonical_markers.pdf",
                show=False,
            )

        sc.pl.rank_genes_groups_dotplot(
            evidence_adata,
            n_genes=5,
            groupby=cluster_key,
            save=f"_{pass_name}_denovo_markers.pdf",
            show=False,
        )

        top_marker_tables = []
        for cluster in evidence_adata.obs[cluster_key].cat.categories:
            cluster_df = sc.get.rank_genes_groups_df(evidence_adata, group=cluster).head(10).copy()
            cluster_df.insert(0, "cluster", cluster)
            top_marker_tables.append(cluster_df)
        if top_marker_tables:
            pd.concat(top_marker_tables, ignore_index=True).to_csv(output_dir / f"{pass_name}_top_markers.csv", index=False)
    finally:
        sc.settings.figdir = old_figdir

    cluster_counts = evidence_adata.obs[cluster_key].astype(str).value_counts().sort_index()
    summary_df = pd.DataFrame({
        "pass_name": pass_name,
        "description": description,
        "cluster": cluster_counts.index.astype(str),
        "n_cells": cluster_counts.values.astype(int),
        "removed": [cluster in removed_clusters for cluster in cluster_counts.index.astype(str)],
    })
    summary_df.to_csv(output_dir / f"{pass_name}_removal_summary.csv", index=False)

    RUN_SUMMARY_ROWS.append({
        "pass_name": pass_name,
        "description": description,
        "cluster_key": cluster_key,
        "n_cells_before": int(evidence_adata.n_obs),
        "removed_clusters": removed_clusters,
        "n_removed": int(summary_df.loc[summary_df["removed"], "n_cells"].sum()),
        "n_cells_after": int(evidence_adata.n_obs - summary_df.loc[summary_df["removed"], "n_cells"].sum()),
        "output_dir": str(output_dir),
    })
    pd.DataFrame(RUN_SUMMARY_ROWS).to_csv(RUN_DIR / "run_filter_summary.csv", index=False)
    _write_run_manifest()

# %%
# ==============================================================================
# 1. INITIAL LOAD & QC
# ==============================================================================
set_global_reproducibility(MUSC_RANDOM_SEED)
_write_run_manifest()

print("--- Loading Data and Basic QC ---")
adata_atlas = sc.read_h5ad(ATLAS_PATH)
if "counts" not in adata_atlas.layers:
    raise ValueError("adata_atlas.layers['counts'] is required for scVI.")

adata_atlas = adata_atlas.copy()
adata_atlas.obs_names = adata_atlas.obs_names.astype(str)
adata_atlas.obs["donor_split_id"] = adata_atlas.obs["donor_split_id"].astype(str)
adata_atlas.obs["celltype_std"] = adata_atlas.obs["celltype_std"].astype(str)

keep_mask = adata_atlas.obs["celltype_std"].isin(MUSC_KEEP_LABELS)
adata_atlas = adata_atlas[keep_mask].copy()
adata_atlas.obs["celltype_std_original"] = adata_atlas.obs["celltype_std"].astype(str)
adata_atlas.obs["celltype_std"] = "MuSC"

var_names_upper = adata_atlas.var_names.str.upper()
noise_mask = (var_names_upper.str.startswith("MT-") | var_names_upper.str.startswith("RPL") | 
              var_names_upper.str.startswith("RPS") | var_names_upper.str.startswith("MT1") | 
              var_names_upper.str.startswith("MT2") | var_names_upper.str.startswith("HSP"))
if noise_mask.any():
    adata_atlas = adata_atlas[:, ~noise_mask].copy()

save_pass_snapshot(
    adata_atlas,
    DIR_INPUT,
    "input_after_qc",
    "MuSC-filtered atlas after initial QC and noise-gene removal.",
)

# %%
# ==============================================================================
# 2. PASS 1 & PASS 2: MACROSCOPIC CLEANUP
# ==============================================================================
print("--- Starting Pass 1 & 2: Identifying macroscopic garbage clusters ---")
# PASS 1
sc.pp.highly_variable_genes(adata_atlas, n_top_genes=MUSC_N_HVG, subset=False, flavor="seurat_v3")
adata_scvi_p1 = adata_atlas[:, adata_atlas.var["highly_variable"]].copy()
scvi.model.SCVI.setup_anndata(adata_scvi_p1, layer="counts", batch_key="donor_split_id")
scvi_model_p1 = scvi.model.SCVI(adata_scvi_p1, n_layers=MUSC_SCVI_N_LAYERS, n_latent=MUSC_SCVI_N_LATENT)
scvi_model_p1.train(max_epochs=MUSC_SCVI_MAX_EPOCHS, accelerator="cuda" if torch.cuda.is_available() else "cpu", devices=1)

adata_atlas.obsm["X_scVI_pass1"] = scvi_model_p1.get_latent_representation()
sc.pp.neighbors(adata_atlas, use_rep="X_scVI_pass1", random_state=MUSC_RANDOM_SEED, n_neighbors=MUSC_N_NEIGHBORS)
sc.tl.leiden(adata_atlas, resolution=MUSC_LEIDEN_RESOLUTION, key_added="leiden_pass1", random_state=MUSC_RANDOM_SEED)
save_pass_evidence(
    adata_atlas,
    cluster_key="leiden_pass1",
    output_dir=DIR_PASS1,
    pass_name="pass1",
    description="Initial macroscopic cleanup before removing obvious contaminant clusters.",
    removed_clusters=["4", "6"],
)
save_pass_snapshot(
    adata_atlas,
    DIR_PASS1,
    "pass1_before_filter",
    "Atlas state after pass1 clustering and before removing contaminant clusters.",
)

# %%

clean_mask_p1 = ~adata_atlas.obs["leiden_pass1"].isin(["6"])
adata_atlas = adata_atlas[clean_mask_p1].copy()
save_pass_snapshot(
    adata_atlas,
    DIR_PASS1,
    "pass1_after_filter",
    "Atlas state after removing pass1 contaminant clusters.",
)


# %%
# PASS 2
sc.pp.highly_variable_genes(adata_atlas, n_top_genes=MUSC_N_HVG, subset=False, flavor="seurat_v3")
adata_scvi_p2 = adata_atlas[:, adata_atlas.var["highly_variable"]].copy()
scvi.model.SCVI.setup_anndata(adata_scvi_p2, layer="counts", batch_key="donor_split_id")
scvi_model_p2 = scvi.model.SCVI(adata_scvi_p2, n_layers=MUSC_SCVI_N_LAYERS, n_latent=MUSC_SCVI_N_LATENT)
scvi_model_p2.train(max_epochs=MUSC_SCVI_MAX_EPOCHS, accelerator="cuda" if torch.cuda.is_available() else "cpu", devices=1)

adata_atlas.obsm["X_scVI_pass2"] = scvi_model_p2.get_latent_representation()
sc.pp.neighbors(adata_atlas, use_rep="X_scVI_pass2", random_state=MUSC_RANDOM_SEED, n_neighbors=MUSC_N_NEIGHBORS)
sc.tl.leiden(adata_atlas, resolution=MUSC_LEIDEN_RESOLUTION, key_added="leiden_pass2", random_state=MUSC_RANDOM_SEED)
save_pass_evidence(
    adata_atlas,
    cluster_key="leiden_pass2",
    output_dir=DIR_PASS2,
    pass_name="pass2",
    description="Final scVI manifold after pass1 cleanup, followed by one last cluster-level cleanup.",
    removed_clusters=["5"],
)
save_pass_snapshot(
    adata_atlas,
    DIR_PASS2,
    "pass2_before_filter",
    "Atlas state after final scVI clustering and before removing cluster 5.",
)

clean_mask_p2 = adata_atlas.obs["leiden_pass2"] != "5"
adata_atlas = adata_atlas[clean_mask_p2].copy()
save_pass_snapshot(
    adata_atlas,
    DIR_PASS2,
    "pass2_final_atlas",
    "Final atlas after pass1 cleanup, pass2 scVI manifold construction, and removal of cluster 5.",
)

adata_atlas.obsm["X_scVI"] = adata_atlas.obsm["X_scVI_pass2"].copy()
sc.pp.neighbors(adata_atlas, use_rep="X_scVI", random_state=MUSC_RANDOM_SEED, n_neighbors=MUSC_N_NEIGHBORS)
sc.tl.umap(adata_atlas, min_dist=MUSC_UMAP_MIN_DIST, random_state=MUSC_RANDOM_SEED)
adata_atlas.obs["leiden_final"] = adata_atlas.obs["leiden_pass2"].astype("category")

write_h5ad_safe(adata_atlas, MUSC_OUTPUT_DIR / "musc_atlas_purified.h5ad")

# %%
# ==============================================================================
# 4. PREPROCESSING & STANDARDIZATION FOR VISUALIZATION
# ==============================================================================
print("--- Standardizing and Preprocessing Data ---")
adata_atlas = sc.read_h5ad(MUSC_OUTPUT_DIR / "musc_atlas_purified.h5ad")

# Standardize to Mouse Title Case
adata_atlas.var_names = adata_atlas.var_names.str.capitalize()
if adata_atlas.raw is not None:
    raw_adata = sc.AnnData(X=adata_atlas.raw.X, var=adata_atlas.raw.var.copy())
    raw_adata.var_names = raw_adata.var_names.str.capitalize()
    adata_atlas.raw = raw_adata

adata_musc_plot = adata_atlas.copy()
sc.pp.normalize_total(adata_musc_plot)
sc.pp.log1p(adata_musc_plot)
adata_musc_plot.raw = adata_musc_plot # Freeze log-normalized
sc.pp.scale(adata_musc_plot, max_value=10) # Scale .X for plotting
adata_musc_plot.obs["leiden_final"] = adata_musc_plot.obs["leiden_final"].astype("category")

# Basic formatting config
sc.settings.set_figure_params(dpi=300, format="pdf", frameon=False, transparent=True)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
rcParams['pdf.fonttype'] = 42

# %%
# ==============================================================================
# 5. EXPLORATORY ANALYSIS (Find your new clusters!)
# ==============================================================================
print(f"--- Saving Exploratory Plots to {DIR_EXPLORATORY} ---")
sc.settings.figdir = str(DIR_EXPLORATORY)

# Basic UMAP and dendrogram of the final Leiden clusters
sc.pl.umap(adata_musc_plot, color="leiden_final", legend_loc="on data", save="_pure_leiden_umap.pdf", show=False)
sc.tl.rank_genes_groups(adata_musc_plot, groupby="leiden_final", method="wilcoxon")
sc.tl.dendrogram(adata_musc_plot, groupby="leiden_final")

# Canonical markers dotplot to map clusters manually
safe_markers = {
    subtype: [g.capitalize() for g in genes if g.capitalize() in adata_musc_plot.var_names] 
    for subtype, genes in MUSC_SUBTYPES.items()
}
safe_markers_flat = {k: v for k, v in safe_markers.items() if v}

sc.pl.dotplot(adata_musc_plot, safe_markers_flat, groupby="leiden_final", standard_scale="var", cmap="Reds", dendrogram=True, save="_canonical_markers_leiden.pdf", show=False)
sc.pl.rank_genes_groups_dotplot(adata_musc_plot, n_genes=5, groupby="leiden_final", save="_denovo_markers_leiden.pdf", show=False)

# Export Top Markers CSV
top_marker_tables = []
for cluster in adata_musc_plot.obs["leiden_final"].cat.categories:
    cluster_df = sc.get.rank_genes_groups_df(adata_musc_plot, group=cluster).head(10).copy()
    cluster_df.insert(0, "cluster", cluster)
    top_marker_tables.append(cluster_df)
pd.concat(top_marker_tables, ignore_index=True).to_csv(DIR_EXPLORATORY / "musc_top_markers_leiden_pure.csv", index=False)

# %%
# ==============================================================================
# 6. PHATE TRAJECTORY CALCULATION
# ==============================================================================
print("--- Calculating Final PHATE Trajectory ---")
phate_operator = phate.PHATE(knn=25, decay=40, t="auto", n_jobs=-1, random_state=MUSC_RANDOM_SEED)
adata_musc_plot.obsm["X_phate"] = phate_operator.fit_transform(adata_musc_plot.obsm["X_scVI"])

# %%
# ==============================================================================
# 7. APPLY BIOLOGICAL ANNOTATIONS & PALETTES
# ==============================================================================
# 🚨🚨🚨 STOP HERE AND CHECK THE EXPLORATORY FOLDER 🚨🚨🚨
# Look at '01_Exploratory/dotplot_canonical_markers_leiden.pdf' 
# Update these numbers to match the newly generated leiden_final clusters!
real_annotation_map = {
    "0": "C0_Quiescent",       
    "1": "C0_Quiescent",       
    "2": "C1_Activated",       
    "3": "C1_Activated",       
    "4": "C2_Proliferating",   
    "5": "C34_Differentiating",
}
# 🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨

# Apply map and color palette
adata_musc_plot.obs["annotation"] = adata_musc_plot.obs["leiden_final"].map(real_annotation_map).astype("category")

custom_colors = {
    "C0_Quiescent": "#557d52",       
    "C1_Activated": "#5f9f55",       
    "C2_Proliferating": "#f8e072",   
    "C34_Differentiating": "#e27e4c" 
}
categories = adata_musc_plot.obs["annotation"].cat.categories
palette = [custom_colors[cat] for cat in categories if cat in custom_colors]
adata_musc_plot.uns["annotation_colors"] = palette

write_h5ad_safe(adata_musc_plot, MUSC_OUTPUT_DIR / "musc_atlas_annotated.h5ad")
save_pass_snapshot(
    adata_musc_plot,
    DIR_PASS2,
    "pass2_annotated_atlas",
    "Final annotated atlas with publication-ready labels and embeddings.",
)

# %% 
# ==============================================================================
# 8. PUBLICATION-READY VISUALIZATION (Main Figures)
# ==============================================================================
print(f"--- Generating Main Publication Figures to {DIR_PUB_MAIN} ---")
sc.settings.figdir = str(DIR_PUB_MAIN)

# Figure 1: Annotated UMAP
sc.pl.umap(adata_musc_plot, color="annotation", legend_loc="on data", legend_fontsize=10, legend_fontoutline=2, title="MuSC Trajectory Atlas", save="_Fig1_Annotated_UMAP.pdf", show=False)

# Figure 2: Trajectory Continuum (PHATE)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
sc.pl.embedding(adata_musc_plot, basis="phate", color="annotation", ax=axs[0], show=False, title="PHATE Lineage")
sc.pl.embedding(adata_musc_plot, basis="phate", color="Myod1", color_map="magma", ax=axs[1], show=False, title="Myod1 (Activation)")
sc.pl.embedding(adata_musc_plot, basis="phate", color="Myog", color_map="magma", ax=axs[2], show=False, title="Myog (Differentiation)")
plt.tight_layout()
fig.savefig(f"{DIR_PUB_MAIN}/_Fig2_PHATE_Trajectory.pdf", dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Canonical Marker Dotplot (Grouped by annotation)
sc.pl.dotplot(adata_musc_plot, safe_markers_flat, groupby="annotation", standard_scale="var", cmap="Reds", dendrogram=False, save="_Fig3_Canonical_Dotplot.pdf", show=False)

# Figure 4: Key Stage Markers
key_genes = ["Pax7", "Myod1", "Mki67", "Myog"] 
valid_genes = [g for g in key_genes if g in adata_musc_plot.var_names]
if valid_genes:
    sc.pl.umap(adata_musc_plot, color=valid_genes, ncols=4, color_map="magma", vmin=0, vmax="p99", frameon=False, save="_Fig4_Key_Stage_Markers.pdf", show=False)

# %%
# ==============================================================================
# 9. INTEGRATION BENCHMARK (Supplementary Figures)
# ==============================================================================
print(f"--- Generating Benchmark Figures to {DIR_PUB_SUPP} ---")
sc.settings.figdir = str(DIR_PUB_SUPP)

# 
adata_musc_plot.obs["dataset_source"] = adata_musc_plot.obs["donor_split_id"].astype(str).str.split("::").str[0].astype("category")

# Step 9a: Compute Unintegrated PCA and its UMAP
sc.tl.pca(adata_musc_plot, svd_solver='arpack')
sc.pp.neighbors(adata_musc_plot, use_rep="X_pca", key_added="neighbors_pca", random_state=MUSC_RANDOM_SEED)
sc.tl.umap(adata_musc_plot, neighbors_key="neighbors_pca", random_state=MUSC_RANDOM_SEED)
adata_musc_plot.obsm["X_umap_pca"] = adata_musc_plot.obsm["X_umap"].copy()

# Step 9b: Restore Integrated scVI UMAP
sc.pp.neighbors(adata_musc_plot, use_rep="X_scVI", key_added="neighbors_scvi", random_state=MUSC_RANDOM_SEED)
sc.tl.umap(adata_musc_plot, neighbors_key="neighbors_scvi", random_state=MUSC_RANDOM_SEED)
adata_musc_plot.obsm["X_umap_scvi"] = adata_musc_plot.obsm["X_umap"].copy()

# Step 9c: Plot 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
sc.pl.embedding(adata_musc_plot, basis="X_umap_pca", color="dataset_source", ax=axs[0, 0], show=False, title="Unintegrated (PCA) - Source", frameon=False)
sc.pl.embedding(adata_musc_plot, basis="X_umap_pca", color="annotation", ax=axs[0, 1], show=False, title="Unintegrated (PCA) - Biology", frameon=False)
sc.pl.embedding(adata_musc_plot, basis="X_umap_scvi", color="dataset_source", ax=axs[1, 0], show=False, title="Integrated (scVI) - Source", frameon=False)
sc.pl.embedding(adata_musc_plot, basis="X_umap_scvi", color="annotation", ax=axs[1, 1], show=False, title="Integrated (scVI) - Biology", frameon=False)
plt.tight_layout()
fig.savefig(f"{DIR_PUB_SUPP}/_Fig5_Integration_Benchmark.pdf", dpi=300, bbox_inches='tight')
plt.close()

print("All tasks successfully completed. Workspace organized.")
