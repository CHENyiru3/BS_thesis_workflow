#!/usr/bin/env python3
"""
Map verification query cells into the MuSC reference scVI space, transfer
reference pseudotime, and score them with the local pseudotime clock.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("PYTHONHASHSEED", "43")

import anndata as ad
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import sparse as sp
from sklearn.neighbors import NearestNeighbors
import scvi
import umap

from path_config import ARTIFACT_ROOT, DATA_ROOT, MOUSE_VERIFICATION_DIR, PROJECT_ROOT

ROOT = PROJECT_ROOT
sys.path.insert(0, str(Path(os.environ.get("ABCLOCK_ROOT", str(ROOT / "abclock")))))
import metacells as abclock_metacells  # noqa: E402


REFERENCE_ATLAS_PATH = ARTIFACT_ROOT / "clock_artifacts" / "musc_annotation" / "musc_atlas_annotated.h5ad"
LOCAL_CLOCK_MODEL_PATH = (
    ARTIFACT_ROOT / "clock_artifacts" / "0310_results" / "expression_only_local_pseudotime" / "final_model.joblib"
)
LOCAL_CLOCK_TRAINING_METRICS_PATH = (
    ARTIFACT_ROOT / "clock_artifacts" / "0310_results" / "expression_only_local_pseudotime" / "training_metrics.json"
)
INPUT_DIR = DATA_ROOT / "external_verify" / "prepared"
MANIFEST_PATH = INPUT_DIR / "selected_samples_manifest.tsv"
OUTDIR = ARTIFACT_ROOT / "scvi_reference_verify_post_injury_local"
YOUNG_INJURY_H5AD = MOUSE_VERIFICATION_DIR / "Myo_Aged_SkM_mm10_v1-1_MuSC.h5ad"

REFERENCE_N_LATENT = 25
REFERENCE_MAX_EPOCHS = 200
QUERY_MAX_EPOCHS = 120
QUERY_NN_K = 30
QUERY_BOOTSTRAP_ITERS = 500
QUERY_METACELL_SIZE = 15
REFERENCE_RANDOM_SEED = 43
REFERENCE_UMAP_N_NEIGHBORS = 30
REFERENCE_UMAP_MIN_DIST = 0.3

YOUNG_COLOR = "#4F7A5A"
OLD_COLOR = "#C65D3B"
ISR_COLOR = "#3E6C8F"
NEUTRAL_COLOR = "#2F4858"
TIMEPOINT_COLOR_MAP = {
    "d0": "#4F7A5A",
    "d1": "#D17C48",
    "d2": "#C65D3B",
    "d3.5": "#8F4C7A",
    "d5": "#4C78A8",
    "d7": "#2F4858",
}

AGE_GROUP_COLOR_MAP = {
    "young": YOUNG_COLOR,
    "old": OLD_COLOR,
    "geriatric": "#7A4C5D",
}


def map_timepoint(value: object) -> str:
    token = str(value).strip().lower().replace("day", "").replace("dpi", "").replace("_", "")
    canonical = {
        "0": "d0",
        "0.0": "d0",
        "d0": "d0",
        "1": "d1",
        "1.0": "d1",
        "d1": "d1",
        "2": "d2",
        "2.0": "d2",
        "d2": "d2",
        "3.5": "d3.5",
        "d3.5": "d3.5",
        "5": "d5",
        "5.0": "d5",
        "d5": "d5",
        "7": "d7",
        "7.0": "d7",
        "d7": "d7",
    }
    return canonical.get(token, f"d{token}")


def map_age_group(value: object) -> str:
    s = str(value).strip().lower()
    if s.startswith("young"):
        return "young"
    if s.startswith("old"):
        return "old"
    if s.startswith("ger"):
        return "geriatric"
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shared-axis scVI reference verification.")
    parser.add_argument("--reference-atlas", default=str(REFERENCE_ATLAS_PATH), help="Reference MuSC atlas with counts and pseudotime.")
    parser.add_argument("--local-model", default=str(LOCAL_CLOCK_MODEL_PATH), help="Local pseudotime clock bundle.")
    parser.add_argument(
        "--training-metrics",
        default=str(LOCAL_CLOCK_TRAINING_METRICS_PATH),
        help="Training metrics JSON for deriving the held-out donor split in evaluation mode.",
    )
    parser.add_argument(
        "--query-source",
        choices=["prepared_external", "young_injury_h5ad"],
        default="young_injury_h5ad",
        help="Query source definition.",
    )
    parser.add_argument("--input-dir", default=str(INPUT_DIR), help="Prepared external sample directory.")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), help="Selected samples manifest for prepared_external mode.")
    parser.add_argument("--query-h5ad", default=str(YOUNG_INJURY_H5AD), help="Query H5AD for injury-timecourse mode.")
    parser.add_argument("--outdir", default=str(OUTDIR), help="Output directory.")
    parser.add_argument(
        "--mode",
        choices=["atlas_fixed", "evaluation", "final_reference"],
        default="atlas_fixed",
        help="atlas_fixed=reuse fixed atlas space/UMAP; evaluation=train-only reference; final_reference=full-atlas reference",
    )
    parser.add_argument("--reference-n-latent", type=int, default=REFERENCE_N_LATENT, help="Reference scVI latent dimensionality.")
    parser.add_argument("--reference-max-epochs", type=int, default=REFERENCE_MAX_EPOCHS, help="Reference scVI training epochs.")
    parser.add_argument("--query-max-epochs", type=int, default=QUERY_MAX_EPOCHS, help="Query surgery training epochs.")
    parser.add_argument("--query-nn-k", type=int, default=QUERY_NN_K, help="Reference neighbors for pseudotime transfer.")
    parser.add_argument("--bootstrap-iters", type=int, default=QUERY_BOOTSTRAP_ITERS, help="Bootstrap metacells per query sample.")
    parser.add_argument("--metacell-size", type=int, default=QUERY_METACELL_SIZE, help="Cells per query metacell.")
    parser.add_argument("--samples", default="", help="Comma-separated sample IDs to run; default runs all available samples.")
    parser.add_argument("--timepoints", default="0.0,1.0,2.0,3.5,5.0,7.0", help="Comma-separated timepoints for injury H5AD mode.")
    parser.add_argument("--age-groups", default="young,old,geriatric", help="Comma-separated age groups for injury H5AD mode.")
    parser.add_argument("--seed", type=int, default=REFERENCE_RANDOM_SEED, help="Random seed.")
    return parser.parse_args()


def _validate_paths(paths: dict[str, Path]) -> None:
    missing = [f"{label}: {path}" for label, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n- " + "\n- ".join(missing))


def _normalize_log1p_from_counts(matrix: sp.spmatrix | np.ndarray, target_sum: float = 1e4) -> sp.spmatrix | np.ndarray:
    if sp.issparse(matrix):
        work = matrix.tocsr(copy=True).astype(np.float32)
        totals = np.asarray(work.sum(axis=1)).ravel().astype(np.float32)
        scale = np.divide(
            np.full_like(totals, target_sum, dtype=np.float32),
            totals,
            out=np.zeros_like(totals, dtype=np.float32),
            where=totals > 0,
        )
        work = sp.diags(scale).dot(work).tocsr()
        work.data = np.log1p(work.data)
        return work

    work = np.asarray(matrix, dtype=np.float32).copy()
    totals = work.sum(axis=1).astype(np.float32)
    scale = np.divide(
        np.full_like(totals, target_sum, dtype=np.float32),
        totals,
        out=np.zeros_like(totals, dtype=np.float32),
        where=totals > 0,
    )
    work *= scale[:, None]
    np.log1p(work, out=work)
    return work


def _coerce_for_h5ad(adata: ad.AnnData) -> ad.AnnData:
    out = adata.copy()
    out.obs_names = pd.Index(out.obs_names.astype(str).tolist(), dtype=object)
    out.var_names = pd.Index(out.var_names.astype(str).tolist(), dtype=object)
    new_obs = {}
    for col in out.obs.columns:
        series = out.obs[col]
        if isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(series) or str(series.dtype).startswith("string"):
            new_obs[col] = pd.Series(series.astype(str).tolist(), index=out.obs.index, dtype=object)
        else:
            new_obs[col] = series.to_numpy()
    out.obs = pd.DataFrame(new_obs, index=out.obs.index)

    new_var = {}
    for col in out.var.columns:
        series = out.var[col]
        if isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(series) or str(series.dtype).startswith("string"):
            new_var[col] = pd.Series(series.astype(str).tolist(), index=out.var.index, dtype=object)
        else:
            new_var[col] = series.to_numpy()
    out.var = pd.DataFrame(new_var, index=out.var.index)
    return out


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def load_manifest(path: Path, selected_samples: str) -> pd.DataFrame:
    manifest = pd.read_csv(path, sep="\t")
    manifest["sample_id"] = manifest["sample_id"].astype(str)
    manifest["selected_file"] = manifest["selected_file"].astype(str)
    manifest["group"] = manifest["group"].astype(str)
    manifest["condition"] = manifest["condition"].astype(str)
    manifest["age_group_std"] = manifest["group"].map(lambda x: "young" if str(x).lower().startswith("adult") else "old")
    if selected_samples.strip():
        keep = {x.strip() for x in selected_samples.split(",") if x.strip()}
        manifest = manifest.loc[manifest["sample_id"].isin(keep)].copy()
    if manifest.empty:
        raise ValueError("No manifest rows selected.")
    return manifest.reset_index(drop=True)


def load_young_injury_query_manifest(query_h5ad: Path, *, selected_samples: str, timepoints: str, age_groups: str) -> pd.DataFrame:
    adata = ad.read_h5ad(query_h5ad, backed="r")
    try:
        obs = adata.obs.copy()
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    required = ["Sample.ID", "Age.Word", "Time.Point"]
    missing = [c for c in required if c not in obs.columns]
    if missing:
        raise ValueError(f"Injury H5AD missing required obs columns: {missing}")

    obs["Sample.ID"] = obs["Sample.ID"].astype(str)
    obs["Age.Word"] = obs["Age.Word"].map(map_age_group).astype(str)
    obs["Time.Point"] = pd.to_numeric(obs["Time.Point"], errors="coerce")
    keep_age_groups = {x.strip().lower() for x in age_groups.split(",") if x.strip()}
    keep_timepoints = {float(x.strip()) for x in timepoints.split(",") if x.strip()}
    obs = obs.loc[obs["Age.Word"].isin(keep_age_groups) & obs["Time.Point"].isin(keep_timepoints)].copy()
    samples = (
        obs.groupby(["Sample.ID", "Age.Word", "Time.Point"], sort=True)
        .size()
        .reset_index(name="n_cells")
        .rename(columns={"Sample.ID": "sample_id", "Age.Word": "age_group_std", "Time.Point": "timepoint"})
    )
    samples["group"] = samples["age_group_std"].map(
        {"young": "Young", "old": "Old", "geriatric": "Geriatric"}
    ).fillna("Unknown")
    samples["condition"] = samples["timepoint"].map(lambda x: "DMSO" if float(x) == 0.0 else "post_injury")
    samples["timepoint_std"] = samples["timepoint"].map(map_timepoint)
    samples["selected_file"] = str(query_h5ad)
    samples["query_mode"] = "injury_h5ad"
    if selected_samples.strip():
        keep = {x.strip() for x in selected_samples.split(",") if x.strip()}
        samples = samples.loc[samples["sample_id"].isin(keep)].copy()
    if samples.empty:
        raise ValueError("No injury samples selected after filtering.")
    return samples.reset_index(drop=True)


def load_reference_adata(path: Path) -> ad.AnnData:
    adata = ad.read_h5ad(path)
    required = ["annotation", "sample_id_std", "source"]
    missing = [c for c in required if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"Reference atlas missing required obs columns: {missing}")
    if "counts" not in adata.layers:
        raise ValueError("Reference atlas missing layers['counts'].")
    adata.obs["reference_annotation"] = adata.obs["annotation"].astype(str)
    if "pseudotime_monocle3" in adata.obs.columns:
        adata.obs["reference_pseudotime"] = pd.to_numeric(adata.obs["pseudotime_monocle3"], errors="coerce")
    else:
        cell_table = pd.read_csv(ARTIFACT_ROOT / "processed_adata" / "monocle3" / "musc_monocle3_cells.tsv", sep="\t")
        cell_table["cell_id"] = cell_table["cell_id"].astype(str)
        pt = pd.Series(pd.to_numeric(cell_table["pseudotime"], errors="coerce").to_numpy(dtype=float), index=cell_table["cell_id"])
        adata.obs["reference_pseudotime"] = pt.reindex(adata.obs_names.astype(str)).to_numpy(dtype=float)
    if not pd.Series(adata.obs["reference_pseudotime"]).notna().all():
        raise ValueError("Reference atlas contains missing/non-numeric reference pseudotime.")
    return adata


def subset_reference_for_mode(
    adata: ad.AnnData,
    *,
    mode: str,
    training_metrics_path: Path,
) -> tuple[ad.AnnData, dict[str, object]]:
    mode = str(mode)
    if mode == "atlas_fixed":
        return adata.copy(), {
            "reference_mode": mode,
            "n_reference_cells": int(adata.n_obs),
            "n_reference_donors": int(adata.obs["sample_id_std"].astype(str).nunique()),
            "excluded_test_donors": [],
        }
    if mode == "final_reference":
        return adata.copy(), {
            "reference_mode": mode,
            "n_reference_cells": int(adata.n_obs),
            "n_reference_donors": int(
                adata.obs["donor_split_id"].astype(str).nunique() if "donor_split_id" in adata.obs.columns else adata.obs["sample_id_std"].astype(str).nunique()
            ),
            "excluded_test_donors": [],
        }

    if "donor_split_id" not in adata.obs.columns:
        raise ValueError("Reference atlas missing donor_split_id required for evaluation-mode subsetting.")
    if not training_metrics_path.exists():
        raise FileNotFoundError(f"Training metrics file not found for evaluation mode: {training_metrics_path}")

    metrics = json.loads(training_metrics_path.read_text())
    test_donors = [str(x) for x in metrics.get("test_donors", [])]
    if not test_donors:
        raise ValueError("Training metrics JSON does not contain test_donors for evaluation mode.")

    keep_mask = ~adata.obs["donor_split_id"].astype(str).isin(set(test_donors))
    out = adata[keep_mask].copy()
    if out.n_obs == 0:
        raise ValueError("Evaluation-mode reference subset is empty after excluding test donors.")
    return out, {
        "reference_mode": mode,
        "n_reference_cells": int(out.n_obs),
        "n_reference_donors": int(out.obs["donor_split_id"].astype(str).nunique()),
        "excluded_test_donors": test_donors,
    }


def train_or_load_reference_scvi(
    adata_ref: ad.AnnData,
    outdir: Path,
    *,
    n_latent: int,
    max_epochs: int,
    seed: int,
) -> tuple[scvi.model.SCVI, ad.AnnData]:
    model_dir = outdir / "reference_scvi_model"
    latent_path = outdir / "reference_latent.tsv"
    meta_path = outdir / "reference_cell_metadata.tsv"

    scvi.settings.seed = int(seed)
    ref_work = adata_ref.copy()
    if model_dir.exists():
        scvi.model.SCVI.setup_anndata(ref_work, layer="counts", batch_key="source")
        ref_model = scvi.model.SCVI.load(model_dir, adata=ref_work)
        if latent_path.exists():
            latent_df = pd.read_csv(latent_path, sep="\t")
            latent_cols = [c for c in latent_df.columns if c != "cell_id"]
            latent_df = latent_df.set_index("cell_id").reindex(ref_work.obs_names.astype(str))
            ref_work.obsm["X_scvi_reference"] = latent_df[latent_cols].to_numpy(dtype=float)
        else:
            ref_work.obsm["X_scvi_reference"] = ref_model.get_latent_representation()
        return ref_model, ref_work

    scvi.model.SCVI.setup_anndata(ref_work, layer="counts", batch_key="source")
    ref_model = scvi.model.SCVI(ref_work, n_latent=int(n_latent), gene_likelihood="nb")
    ref_model.train(max_epochs=int(max_epochs), accelerator="auto", devices=1, plan_kwargs={"weight_decay": 0.0})
    ref_work.obsm["X_scvi_reference"] = ref_model.get_latent_representation()

    ref_model.save(model_dir, overwrite=True, save_anndata=False)
    latent_df = pd.DataFrame(ref_work.obsm["X_scvi_reference"], index=ref_work.obs_names.astype(str))
    latent_df.index.name = "cell_id"
    latent_df.to_csv(latent_path, sep="\t")
    ref_work.obs.assign(cell_id=ref_work.obs_names.astype(str)).to_csv(meta_path, sep="\t", index=False)
    return ref_model, ref_work


def fit_or_load_reference_umap(
    ref_adata: ad.AnnData,
    reference_dir: Path,
    *,
    mode: str,
    seed: int,
    n_neighbors: int = REFERENCE_UMAP_N_NEIGHBORS,
    min_dist: float = REFERENCE_UMAP_MIN_DIST,
) -> tuple[object, pd.DataFrame]:
    umap_path = reference_dir / "reference_umap.tsv"
    metadata_path = reference_dir / "reference_umap_metadata.json"

    if str(mode) == "atlas_fixed" and "X_umap" in ref_adata.obsm:
        coords = np.asarray(ref_adata.obsm["X_umap"], dtype=float)
        umap_df = pd.DataFrame(
            {
                "cell_id": ref_adata.obs_names.astype(str),
                "umap_1": coords[:, 0],
                "umap_2": coords[:, 1],
            }
        )
        umap_df.to_csv(umap_path, sep="\t", index=False)
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "reference_umap_source": "atlas_X_umap",
                    "query_projection_mode": "knn_interpolation_from_reference_latent",
                },
                handle,
                indent=2,
            )
        return None, umap_df

    if umap_path.exists():
        umap_df = pd.read_csv(umap_path, sep="\t")
        return None, umap_df

    reducer = umap.UMAP(
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric="euclidean",
        random_state=int(seed),
        transform_seed=int(seed),
    )
    coords = reducer.fit_transform(np.asarray(ref_adata.obsm["X_scvi_reference"], dtype=float))
    umap_df = pd.DataFrame(
        {
            "cell_id": ref_adata.obs_names.astype(str),
            "umap_1": coords[:, 0],
            "umap_2": coords[:, 1],
        }
    )
    umap_df.to_csv(umap_path, sep="\t", index=False)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "n_neighbors": int(n_neighbors),
                "min_dist": float(min_dist),
                "metric": "euclidean",
                "random_state": int(seed),
                "reference_umap_source": "refit_from_X_scvi_reference",
                "query_projection_mode": "knn_interpolation_from_reference_latent",
            },
            handle,
            indent=2,
        )
    return None, umap_df


def transform_query_to_reference_umap(
    query_latent: np.ndarray,
    reference_latent_df: pd.DataFrame,
    reference_umap_df: pd.DataFrame,
    *,
    n_neighbors: int = QUERY_NN_K,
) -> pd.DataFrame:
    merged = reference_latent_df.merge(reference_umap_df, on="cell_id", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("Reference latent and UMAP tables do not overlap on cell_id.")

    ref_latent = merged[["latent_1", "latent_2"]].copy()
    extra_latent_cols = [c for c in merged.columns if c.startswith("latent_") and c not in {"latent_1", "latent_2"}]
    latent_cols = ["latent_1", "latent_2"] + sorted(extra_latent_cols, key=lambda x: int(x.split("_")[1]))
    ref_latent = merged[latent_cols].to_numpy(dtype=float)
    ref_umap = merged[["umap_1", "umap_2"]].to_numpy(dtype=float)

    nn = NearestNeighbors(n_neighbors=min(int(n_neighbors), ref_latent.shape[0]), metric="euclidean")
    nn.fit(ref_latent)
    distances, indices = nn.kneighbors(np.asarray(query_latent, dtype=float))
    weights = 1.0 / np.clip(distances, 1e-8, None)
    weights = weights / weights.sum(axis=1, keepdims=True)
    coords = np.sum(weights[:, :, None] * ref_umap[indices], axis=1)
    return pd.DataFrame({"umap_1": coords[:, 0], "umap_2": coords[:, 1]})


def prepare_query_adata(path: Path, manifest_row: pd.Series, ref_var_names: pd.Index) -> ad.AnnData:
    query = ad.read_h5ad(path)
    if "counts" not in query.layers:
        raise ValueError(f"{path.name} missing layers['counts']")
    if "musc_like_keep" in query.obs.columns:
        keep = query.obs["musc_like_keep"].astype(bool).to_numpy()
        query = query[keep].copy()
    if "gene_symbol" in query.var.columns:
        query.var_names = pd.Index(query.var["gene_symbol"].astype(str))
    query.var_names_make_unique(join="__dup")
    overlap = ref_var_names.intersection(query.var_names)
    if overlap.empty:
        raise ValueError(f"No overlapping genes between query {path.name} and reference atlas.")

    counts = query.layers["counts"]
    if not sp.issparse(counts):
        counts = sp.csr_matrix(counts)
    else:
        counts = counts.tocsr()
    gene_to_idx = {str(g): i for i, g in enumerate(query.var_names.astype(str))}
    present_genes = [str(g) for g in ref_var_names if str(g) in gene_to_idx]
    missing_genes = [str(g) for g in ref_var_names if str(g) not in gene_to_idx]

    present_idx = [gene_to_idx[g] for g in present_genes]
    present_counts = counts[:, present_idx].tocoo()
    target_col_index = np.array([i for i, gene in enumerate(ref_var_names.astype(str)) if gene in gene_to_idx], dtype=np.int64)
    new_counts = sp.csr_matrix(
        (present_counts.data, (present_counts.row, target_col_index[present_counts.col])),
        shape=(query.n_obs, len(ref_var_names)),
        dtype=counts.dtype,
    )

    query = ad.AnnData(
        X=new_counts.copy(),
        obs=query.obs.copy(),
        var=pd.DataFrame(index=pd.Index(ref_var_names.astype(str))),
        layers={"counts": new_counts.copy()},
    )
    query.var_names = pd.Index(ref_var_names.astype(str))
    query.obs_names = pd.Index([f"{manifest_row['sample_id']}::{x}" for x in query.obs_names.astype(str)])
    query.obs["cell_id"] = query.obs_names.astype(str)
    query.obs["sample_id_std"] = str(manifest_row["sample_id"])
    query.obs["sample_label"] = str(manifest_row["sample_id"])
    query.obs["age_group_std"] = str(manifest_row["age_group_std"])
    query.obs["condition_std"] = str(manifest_row["condition"])
    query.obs["group_std"] = str(manifest_row["group"])
    query.obs["source"] = "external_verify_prepared"
    query.layers["data"] = _normalize_log1p_from_counts(query.layers["counts"])
    query.X = query.layers["counts"].copy()
    query.uns["reference_gene_overlap"] = {
        "n_reference_genes": int(len(ref_var_names)),
        "n_query_overlap_genes": int(len(present_genes)),
        "n_missing_reference_genes": int(len(missing_genes)),
    }
    return query


def prepare_query_adata_from_young_injury(query_h5ad: Path, manifest_row: pd.Series, ref_var_names: pd.Index) -> ad.AnnData:
    query = ad.read_h5ad(query_h5ad)
    required = ["Sample.ID", "Age.Word", "Time.Point"]
    missing = [c for c in required if c not in query.obs.columns]
    if missing:
        raise ValueError(f"Injury H5AD missing required obs columns: {missing}")
    if "counts" not in query.layers:
        raise ValueError("Young injury H5AD missing layers['counts'].")

    query.obs["Sample.ID"] = query.obs["Sample.ID"].astype(str)
    query.obs["Age.Word"] = query.obs["Age.Word"].astype(str)
    query.obs["Time.Point"] = pd.to_numeric(query.obs["Time.Point"], errors="coerce")
    mask = (
        query.obs["Sample.ID"].eq(str(manifest_row["sample_id"]))
        & query.obs["Age.Word"].map(map_age_group).eq(str(manifest_row["age_group_std"]))
        & query.obs["Time.Point"].eq(float(manifest_row["timepoint"]))
    )
    query = query[mask].copy()
    if query.n_obs == 0:
        raise ValueError(f"No cells found for injury sample {manifest_row['sample_id']}.")

    if "feature_name" in query.var.columns:
        query.var_names = pd.Index(query.var["feature_name"].astype(str))
    query.var_names_make_unique(join="__dup")
    overlap = ref_var_names.intersection(query.var_names)
    if overlap.empty:
        raise ValueError(f"No overlapping genes between injury query {manifest_row['sample_id']} and reference atlas.")

    counts = query.layers["counts"]
    if not sp.issparse(counts):
        counts = sp.csr_matrix(counts)
    else:
        counts = counts.tocsr()
    gene_to_idx = {str(g): i for i, g in enumerate(query.var_names.astype(str))}
    present_genes = [str(g) for g in ref_var_names if str(g) in gene_to_idx]
    missing_genes = [str(g) for g in ref_var_names if str(g) not in gene_to_idx]
    present_idx = [gene_to_idx[g] for g in present_genes]
    present_counts = counts[:, present_idx].tocoo()
    target_col_index = np.array([i for i, gene in enumerate(ref_var_names.astype(str)) if gene in gene_to_idx], dtype=np.int64)
    new_counts = sp.csr_matrix(
        (present_counts.data, (present_counts.row, target_col_index[present_counts.col])),
        shape=(query.n_obs, len(ref_var_names)),
        dtype=counts.dtype,
    )

    query = ad.AnnData(
        X=new_counts.copy(),
        obs=query.obs.copy(),
        var=pd.DataFrame(index=pd.Index(ref_var_names.astype(str))),
        layers={"counts": new_counts.copy()},
    )
    query.var_names = pd.Index(ref_var_names.astype(str))
    query.obs_names = pd.Index([f"{manifest_row['sample_id']}::{x}" for x in query.obs_names.astype(str)])
    query.obs["cell_id"] = query.obs_names.astype(str)
    query.obs["sample_id_std"] = str(manifest_row["sample_id"])
    query.obs["sample_label"] = str(manifest_row["sample_id"])
    query.obs["age_group_std"] = str(manifest_row["age_group_std"])
    query.obs["condition_std"] = str(manifest_row["condition"])
    query.obs["group_std"] = str(manifest_row["group"])
    query.obs["timepoint_std"] = str(manifest_row["timepoint_std"])
    query.obs["source"] = "young_injury_h5ad"
    query.layers["data"] = _normalize_log1p_from_counts(query.layers["counts"])
    query.X = query.layers["counts"].copy()
    query.uns["reference_gene_overlap"] = {
        "n_reference_genes": int(len(ref_var_names)),
        "n_query_overlap_genes": int(len(present_genes)),
        "n_missing_reference_genes": int(len(missing_genes)),
    }
    return query


def map_query_to_reference(
    query: ad.AnnData,
    ref_model: scvi.model.SCVI,
    *,
    max_epochs: int,
) -> tuple[scvi.model.SCVI, ad.AnnData]:
    query_work = query.copy()
    scvi.model.SCVI.prepare_query_anndata(query_work, ref_model)
    query_model = scvi.model.SCVI.load_query_data(query_work, ref_model)
    query_model.train(max_epochs=int(max_epochs), accelerator="auto", devices=1, plan_kwargs={"weight_decay": 0.0})
    query_work.obsm["X_scvi_mapped"] = query_model.get_latent_representation()
    return query_model, query_work


def transfer_reference_pseudotime(
    ref_adata: ad.AnnData,
    query_adata: ad.AnnData,
    *,
    n_neighbors: int,
) -> pd.DataFrame:
    ref_latent = np.asarray(ref_adata.obsm["X_scvi_reference"], dtype=float)
    query_latent = np.asarray(query_adata.obsm["X_scvi_mapped"], dtype=float)
    ref_pt = pd.to_numeric(ref_adata.obs["reference_pseudotime"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(ref_pt).all():
        raise ValueError("Reference pseudotime contains non-finite values.")

    nn = NearestNeighbors(n_neighbors=min(int(n_neighbors), ref_latent.shape[0]), metric="euclidean")
    nn.fit(ref_latent)
    distances, indices = nn.kneighbors(query_latent)

    weights = 1.0 / np.clip(distances, 1e-8, None)
    weights = weights / weights.sum(axis=1, keepdims=True)
    transferred_pt = np.sum(weights * ref_pt[indices], axis=1)
    mean_dist = distances.mean(axis=1)
    confidence = 1.0 / (1.0 + mean_dist)

    out = pd.DataFrame(
        {
            "cell_id": query_adata.obs_names.astype(str),
            "reference_pseudotime_transferred": transferred_pt,
            "reference_pseudotime_transfer_confidence": confidence,
            "reference_nn_mean_distance": mean_dist,
            "reference_nn_k": int(indices.shape[1]),
        }
    )
    out["reference_nearest_cell_id"] = [str(ref_adata.obs_names[idx_row[0]]) for idx_row in indices]
    return out


def generate_query_metacells(
    query_adata: ad.AnnData,
    *,
    bootstrap_iters: int,
    metacell_size: int,
    seed: int,
) -> ad.AnnData:
    work = query_adata.copy()
    work.obs["donor_bootstrap_id"] = work.obs["sample_id_std"].astype(str)
    work.obs["donor_split_id"] = work.obs["sample_id_std"].astype(str)
    work.obs["age_group_std"] = work.obs["age_group_std"].astype(str)
    work.obs["Age_group_binary_for_clock"] = work.obs["age_group_std"].replace({"geriatric": "old"}).astype(str)
    if "musc_best_subtype" in work.obs.columns:
        work.obs["celltype_std"] = work.obs["musc_best_subtype"].astype(str)
    else:
        work.obs["celltype_std"] = "MuSC"
    bs = abclock_metacells.generate_bootstrap_cells(
        work,
        n_cells_per_bin=int(metacell_size),
        n_iter=int(bootstrap_iters),
        donor_col="donor_bootstrap_id",
        age_col="Age_group_binary_for_clock",
        state_col="celltype_std",
        extra_meta_cols=["sample_id_std", "source", "condition_std", "group_std", "donor_split_id", "age_group_std"],
        aggregate_numeric_meta_cols=["reference_pseudotime_transferred", "reference_pseudotime_transfer_confidence"],
        balance_classes=False,
        random_state=int(seed),
    )
    if bs.n_obs == 0:
        age_counts = work.obs["age_group_std"].value_counts(dropna=False).to_dict()
        raise ValueError(
            "Bootstrap metacell generation returned no metacells. "
            f"Observed age_group_std counts: {age_counts}"
        )
    return bs


def preprocess_query_metacells_for_local_clock(
    metacells: ad.AnnData,
    gene_names: list[str],
) -> ad.AnnData:
    work = metacells.copy()
    counts = work.layers["counts"] if "counts" in work.layers else work.X
    work.layers["data"] = _normalize_log1p_from_counts(counts)
    work.X = work.layers["data"].copy()
    work.obs["pseudotime_monocle3_mean"] = pd.to_numeric(
        work.obs["reference_pseudotime_transferred_mean"],
        errors="coerce",
    )
    if not work.obs["pseudotime_monocle3_mean"].notna().all():
        raise ValueError("Query metacells contain missing transferred pseudotime.")
    keep = [gene for gene in gene_names if gene in work.var_names]
    if len(keep) != len(gene_names):
        missing = [gene for gene in gene_names if gene not in work.var_names]
        raise ValueError(f"Query metacells missing local-clock genes: {missing[:10]}")
    work = work[:, gene_names].copy()
    if "age_group_std" in work.obs.columns:
        work.obs["Age"] = work.obs["age_group_std"].astype(str)
    else:
        work.obs["Age"] = work.obs["Age"].astype(str)
    return work


def predict_local_clock(model_bundle: dict[str, object], adata: ad.AnnData) -> pd.DataFrame:
    if model_bundle.get("model_family") != "local_pseudotime_elasticnet":
        raise ValueError("Loaded model is not a local pseudotime clock bundle.")
    local = model_bundle["local_clock_family"]
    gene_names = [str(g) for g in model_bundle["gene_names"]]
    if list(adata.var_names.astype(str)) != gene_names:
        raise ValueError("Inference requires exact local-clock gene ordering.")

    pt = pd.to_numeric(adata.obs["pseudotime_monocle3_mean"], errors="coerce").to_numpy(dtype=float)
    centers = np.asarray(local["window_centers"], dtype=float)
    sigma = float(local["window_sigma"])
    raw = np.exp(-0.5 * ((pt[:, None] - centers[None, :]) / sigma) ** 2)
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    memberships = raw / row_sums

    X = np.asarray(adata.X.toarray() if sp.issparse(adata.X) else adata.X, dtype=float)
    prob_matrix = np.column_stack([np.asarray(clf.predict_proba(X)[:, 1], dtype=float) for clf in local["window_models"]])
    thresholds = np.asarray(local["window_thresholds"], dtype=float)
    p_old = np.sum(memberships * prob_matrix, axis=1)
    local_thr = np.sum(memberships * thresholds[None, :], axis=1)

    out = pd.DataFrame(
        {
            "metacell_id": adata.obs_names.astype(str),
            "sample_id_std": adata.obs["sample_id_std"].astype(str).to_numpy(),
            "age_group_std": adata.obs["Age"].astype(str).to_numpy(),
            "reference_pseudotime_transferred_mean": pt,
            "reference_pseudotime_transfer_confidence_mean": pd.to_numeric(
                adata.obs["reference_pseudotime_transfer_confidence_mean"], errors="coerce"
            ).to_numpy(dtype=float),
            "p_old": p_old,
            "local_threshold": local_thr,
            "pred_old_thr_local": (p_old >= local_thr).astype(int),
            "nearest_window_id": np.argmax(memberships, axis=1).astype(int),
        }
    )
    for idx in range(prob_matrix.shape[1]):
        out[f"window_{idx}_membership"] = memberships[:, idx]
        out[f"window_{idx}_p_old"] = prob_matrix[:, idx]
    return out


def summarize_query_curve(pred_df: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
    work = pred_df.copy()
    work["pt_bin"] = pd.qcut(work["reference_pseudotime_transferred_mean"], q=int(n_bins), labels=False, duplicates="drop")
    work = work.dropna(subset=["pt_bin"]).copy()
    work["pt_bin"] = work["pt_bin"].astype(int)
    out = (
        work.groupby("pt_bin")
        .agg(
            n_metacells=("p_old", "size"),
            mean_p_old=("p_old", "mean"),
            mean_threshold=("local_threshold", "mean"),
            mean_pseudotime=("reference_pseudotime_transferred_mean", "mean"),
            std_p_old=("p_old", "std"),
        )
        .reset_index()
        .sort_values("mean_pseudotime")
        .reset_index(drop=True)
    )
    out["std_p_old"] = out["std_p_old"].fillna(0.0)
    out["se_p_old"] = out["std_p_old"] / np.sqrt(out["n_metacells"].clip(lower=1))
    out["ci95_low"] = out["mean_p_old"] - 1.96 * out["se_p_old"]
    out["ci95_high"] = out["mean_p_old"] + 1.96 * out["se_p_old"]
    return out


def _sample_color(sample_id: str, meta: pd.DataFrame) -> str:
    row = meta.loc[meta["sample_id_std"] == sample_id].iloc[0]
    if "timepoint_std" in row.index and pd.notna(row["timepoint_std"]) and meta["age_group_std"].astype(str).nunique() == 1:
        return TIMEPOINT_COLOR_MAP.get(str(row["timepoint_std"]), NEUTRAL_COLOR)
    if str(row["age_group_std"]) in AGE_GROUP_COLOR_MAP:
        return AGE_GROUP_COLOR_MAP[str(row["age_group_std"])]
    if "isr" in str(row["condition_std"]).lower():
        return ISR_COLOR
    return OLD_COLOR


def plot_query_curve(curve_df: pd.DataFrame, outdir: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(curve_df["mean_pseudotime"], curve_df["mean_p_old"], color=NEUTRAL_COLOR, linewidth=2.8)
    ax.fill_between(curve_df["mean_pseudotime"], curve_df["ci95_low"], curve_df["ci95_high"], color=NEUTRAL_COLOR, alpha=0.18)
    ax.plot(curve_df["mean_pseudotime"], curve_df["mean_threshold"], color="#B56576", linewidth=1.8, linestyle="--")
    ax.set_xlabel("Transferred reference pseudotime")
    ax.set_ylabel("Aging score (p_old)")
    ax.set_ylim(0, 1.02)
    ax.set_title(title, fontweight="bold", pad=12)
    return _save_figure(fig, outdir / "01_aging_score_across_reference_pseudotime.pdf")


def plot_reference_umap_reference_only(reference_df: pd.DataFrame, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    sca = ax.scatter(
        reference_df["umap_1"],
        reference_df["umap_2"],
        c=reference_df["reference_pseudotime"],
        cmap="viridis",
        s=6,
        linewidths=0,
        alpha=0.8,
    )
    ax.set_xlabel("Reference scVI UMAP 1")
    ax.set_ylabel("Reference scVI UMAP 2")
    ax.set_title("Reference atlas scVI UMAP", fontweight="bold", pad=12)
    cbar = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Reference pseudotime")
    return _save_figure(fig, outdir / "00_reference_umap_reference_only.pdf")


def plot_reference_umap_overlay(reference_df: pd.DataFrame, query_df: pd.DataFrame, sample_id: str, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.scatter(reference_df["umap_1"], reference_df["umap_2"], s=5, alpha=0.08, color="#A0AEC0", linewidths=0)
    color_values = query_df["reference_pseudotime_transferred"]
    cmap = "viridis"
    if "timepoint_std" in query_df.columns:
        query_df = query_df.copy()
        query_df["plot_color"] = query_df["timepoint_std"].map(TIMEPOINT_COLOR_MAP).fillna(NEUTRAL_COLOR)
        ax.scatter(query_df["umap_1"], query_df["umap_2"], c=query_df["plot_color"], s=13, linewidths=0, alpha=0.9)
    else:
        sca = ax.scatter(query_df["umap_1"], query_df["umap_2"], c=color_values, cmap=cmap, s=13, linewidths=0, alpha=0.9)
        cbar = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Transferred pseudotime")
    ax.set_xlabel("Reference scVI UMAP 1")
    ax.set_ylabel("Reference scVI UMAP 2")
    ax.set_title(f"{sample_id} on reference scVI UMAP", fontweight="bold", pad=12)
    return _save_figure(fig, outdir / "01_reference_umap_query_overlay.pdf")


def plot_query_umap_by_transferred_pseudotime(query_df: pd.DataFrame, sample_id: str, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    sca = ax.scatter(
        query_df["umap_1"],
        query_df["umap_2"],
        c=query_df["reference_pseudotime_transferred"],
        cmap="viridis",
        s=13,
        linewidths=0,
        alpha=0.9,
    )
    ax.set_xlabel("Reference scVI UMAP 1")
    ax.set_ylabel("Reference scVI UMAP 2")
    ax.set_title(f"{sample_id} by transferred pseudotime", fontweight="bold", pad=12)
    cbar = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Transferred pseudotime")
    return _save_figure(fig, outdir / "02_query_umap_by_transferred_pseudotime.pdf")


def plot_cross_sample_summary(curves: pd.DataFrame, sample_meta: pd.DataFrame, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    if "timepoint_std" in sample_meta.columns and sample_meta["age_group_std"].astype(str).nunique() > 1:
        curves = curves.merge(sample_meta[["sample_id_std", "age_group_std"]], on="sample_id_std", how="left")
        age_labels = {"young": "Young", "old": "Old", "geriatric": "26mo"}
        for age_group, sub in curves.groupby("age_group_std", sort=False):
            color = AGE_GROUP_COLOR_MAP.get(str(age_group), NEUTRAL_COLOR)
            grouped = (
                sub.groupby("pt_bin", sort=True)
                .agg(
                    mean_pseudotime=("mean_pseudotime", "mean"),
                    mean_p_old=("mean_p_old", "mean"),
                    std_p_old=("mean_p_old", "std"),
                    n_samples=("sample_id_std", "nunique"),
                )
                .reset_index()
                .sort_values("mean_pseudotime")
            )
            grouped["std_p_old"] = grouped["std_p_old"].fillna(0.0)
            grouped["se_p_old"] = grouped["std_p_old"] / np.sqrt(grouped["n_samples"].clip(lower=1))
            grouped["ci95_low"] = grouped["mean_p_old"] - 1.96 * grouped["se_p_old"]
            grouped["ci95_high"] = grouped["mean_p_old"] + 1.96 * grouped["se_p_old"]
            ax.plot(
                grouped["mean_pseudotime"],
                grouped["mean_p_old"],
                linewidth=2.8,
                color=color,
                label=age_labels.get(str(age_group), str(age_group)),
            )
            ax.fill_between(grouped["mean_pseudotime"], grouped["ci95_low"], grouped["ci95_high"], color=color, alpha=0.14)
        ax.set_title("Post-injury local clock by age group on shared reference pseudotime", fontweight="bold", pad=12)
        ax.legend(title="Age group", frameon=False, loc="best")
    elif "timepoint_std" in sample_meta.columns:
        curves = curves.merge(sample_meta[["sample_id_std", "timepoint_std"]], on="sample_id_std", how="left")
        for timepoint, sub in curves.groupby("timepoint_std", sort=False):
            color = TIMEPOINT_COLOR_MAP.get(str(timepoint), NEUTRAL_COLOR)
            grouped = (
                sub.groupby("pt_bin", sort=True)
                .agg(
                    mean_pseudotime=("mean_pseudotime", "mean"),
                    mean_p_old=("mean_p_old", "mean"),
                    std_p_old=("mean_p_old", "std"),
                    n_samples=("sample_id_std", "nunique"),
                )
                .reset_index()
                .sort_values("mean_pseudotime")
            )
            grouped["std_p_old"] = grouped["std_p_old"].fillna(0.0)
            grouped["se_p_old"] = grouped["std_p_old"] / np.sqrt(grouped["n_samples"].clip(lower=1))
            grouped["ci95_low"] = grouped["mean_p_old"] - 1.96 * grouped["se_p_old"]
            grouped["ci95_high"] = grouped["mean_p_old"] + 1.96 * grouped["se_p_old"]
            ax.plot(grouped["mean_pseudotime"], grouped["mean_p_old"], linewidth=2.8, color=color, label=str(timepoint))
            ax.fill_between(grouped["mean_pseudotime"], grouped["ci95_low"], grouped["ci95_high"], color=color, alpha=0.14)
        ax.set_title("Young injury queries by timepoint on shared reference pseudotime", fontweight="bold", pad=12)
        ax.legend(title="Timepoint", frameon=False, loc="best")
    else:
        for sample_id, sub in curves.groupby("sample_id_std", sort=False):
            color = _sample_color(sample_id, sample_meta)
            meta = sample_meta.loc[sample_meta["sample_id_std"] == sample_id].iloc[0]
            label = f"{sample_id} ({meta['group_std']}, {meta['condition_std']})"
            sub = sub.sort_values("mean_pseudotime")
            ax.plot(sub["mean_pseudotime"], sub["mean_p_old"], linewidth=2.8, color=color, label=label)
        ax.set_title("Mapped query samples on shared reference pseudotime", fontweight="bold", pad=12)
        ax.legend(frameon=False, loc="best")
    ax.set_xlabel("Transferred reference pseudotime")
    ax.set_ylabel("Mean aging score (p_old)")
    ax.set_ylim(0, 1.02)
    return _save_figure(fig, outdir / "03_cross_sample_shared_reference_pseudotime.pdf")


def maybe_compare_previous(summary_dir: Path) -> None:
    previous = ARTIFACT_ROOT / "external_verify_prepared" / "summary" / "sample_summary.tsv"
    current = summary_dir / "sample_summary.tsv"
    if not previous.exists() or not current.exists():
        return
    prev_df = pd.read_csv(previous, sep="\t").rename(columns={"mean_p_old": "mean_p_old_separate"})
    cur_df = pd.read_csv(current, sep="\t").rename(columns={"mean_p_old": "mean_p_old_scvi_reference"})
    merged = cur_df.merge(prev_df[["sample_id_std", "mean_p_old_separate"]], on="sample_id_std", how="left")
    merged.to_csv(summary_dir / "comparison_to_separate_trajectory.tsv", sep="\t", index=False)


def compare_to_direct_injury_apply(summary_dir: Path) -> None:
    direct_path = ARTIFACT_ROOT / "clock_artifacts" / "0310_results" / "expression_only" / "supplementary_trajectory" / "sample_scores.tsv"
    current_path = summary_dir / "sample_summary.tsv"
    if not direct_path.exists() or not current_path.exists():
        return
    direct = pd.read_csv(direct_path, sep="\t")
    direct["sample_id_std"] = direct["sample_id_std"].astype(str)
    direct["timepoint_std"] = direct["timepoint_std"].astype(str)
    direct["age_group_std"] = direct["age_group_std"].astype(str)
    direct = direct[["sample_id_std", "timepoint_std", "age_group_std", "mean_p_old"]].rename(columns={"mean_p_old": "mean_p_old_direct"})
    current = pd.read_csv(current_path, sep="\t")
    if "timepoint_std" not in current.columns:
        return
    merged = current.merge(direct, on=["sample_id_std", "timepoint_std", "age_group_std"], how="left")
    merged.to_csv(summary_dir / "comparison_to_direct_apply.tsv", sep="\t", index=False)


def main() -> None:
    args = parse_args()

    reference_atlas = Path(args.reference_atlas)
    local_model_path = Path(args.local_model)
    training_metrics_path = Path(args.training_metrics)
    input_dir = Path(args.input_dir)
    manifest_path = Path(args.manifest)
    query_h5ad = Path(args.query_h5ad)
    outdir = Path(args.outdir)

    required = {
        "reference atlas": reference_atlas,
        "local model": local_model_path,
        "training metrics": training_metrics_path,
    }
    if str(args.query_source) == "prepared_external":
        required.update({"input dir": input_dir, "manifest": manifest_path})
    else:
        required.update({"query_h5ad": query_h5ad})
    _validate_paths(required)

    outdir.mkdir(parents=True, exist_ok=True)
    reference_dir = outdir / "reference" / str(args.mode)
    reference_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = outdir / "summary" / str(args.mode)
    summary_dir.mkdir(parents=True, exist_ok=True)

    if str(args.query_source) == "prepared_external":
        manifest = load_manifest(manifest_path, args.samples)
    else:
        manifest = load_young_injury_query_manifest(
            query_h5ad,
            selected_samples=args.samples,
            timepoints=str(args.timepoints),
            age_groups=str(args.age_groups),
        )
    model_bundle = joblib.load(local_model_path)
    local_gene_names = [str(g) for g in model_bundle["gene_names"]]

    print("[1/4] Loading reference atlas and training/loading reference scVI model")
    adata_ref_full = load_reference_adata(reference_atlas)
    adata_ref, ref_meta = subset_reference_for_mode(
        adata_ref_full,
        mode=str(args.mode),
        training_metrics_path=training_metrics_path,
    )
    ref_model, ref_saved = train_or_load_reference_scvi(
        adata_ref,
        reference_dir,
        n_latent=int(args.reference_n_latent),
        max_epochs=int(args.reference_max_epochs),
        seed=int(args.seed),
    )
    ref_latent = np.asarray(ref_saved.obsm["X_scvi_reference"], dtype=float)
    ref_latent_full_df = pd.DataFrame(ref_latent, index=ref_saved.obs_names.astype(str))
    ref_latent_full_df.columns = [f"latent_{i+1}" for i in range(ref_latent.shape[1])]
    ref_latent_full_df.index.name = "cell_id"
    ref_latent_full_df = ref_latent_full_df.reset_index()
    ref_latent_df = ref_latent_full_df[["cell_id", "latent_1", "latent_2"]].copy()
    ref_latent_df["reference_pseudotime"] = pd.to_numeric(ref_saved.obs["reference_pseudotime"], errors="coerce").to_numpy(dtype=float)
    ref_latent_df.to_csv(reference_dir / "reference_latent_2d.tsv", sep="\t", index=False)
    _, reference_umap_df = fit_or_load_reference_umap(
        ref_saved,
        reference_dir,
        mode=str(args.mode),
        seed=int(args.seed),
    )
    reference_umap_df = reference_umap_df.merge(
        ref_latent_df[["cell_id", "reference_pseudotime"]],
        on="cell_id",
        how="left",
    )
    plot_reference_umap_reference_only(reference_umap_df, reference_dir)
    with open(reference_dir / "reference_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "reference_atlas": str(reference_atlas),
                "training_metrics": str(training_metrics_path),
                **ref_meta,
                "n_latent": int(args.reference_n_latent),
                "reference_max_epochs": int(args.reference_max_epochs),
            },
            handle,
            indent=2,
        )

    sample_rows = []
    curve_frames = []
    meta_rows = []

    for _, row in manifest.iterrows():
        sample_id = str(row["sample_id"])
        if str(args.query_source) == "prepared_external":
            sample_file = input_dir / str(row["selected_file"])
            if not sample_file.exists():
                raise FileNotFoundError(f"Prepared sample file not found: {sample_file}")
        else:
            sample_file = query_h5ad

        sample_dir = outdir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        print(f"[2/4:{sample_id}] Preparing and mapping query")
        if str(args.query_source) == "prepared_external":
            query = prepare_query_adata(sample_file, row, ref_saved.var_names)
        else:
            query = prepare_query_adata_from_young_injury(query_h5ad, row, ref_saved.var_names)
        _, query_mapped = map_query_to_reference(query, ref_model, max_epochs=int(args.query_max_epochs))

        transfer_df = transfer_reference_pseudotime(ref_saved, query_mapped, n_neighbors=int(args.query_nn_k))
        query_mapped.obs = query_mapped.obs.merge(transfer_df, left_index=True, right_on="cell_id", how="left").set_index("cell_id")
        query_mapped.obs.index = query_mapped.obs.index.astype(str)

        query_latent = np.asarray(query_mapped.obsm["X_scvi_mapped"], dtype=float)
        query_latent_df = pd.DataFrame(
            {
                "cell_id": query_mapped.obs_names.astype(str),
                "latent_1": query_latent[:, 0],
                "latent_2": query_latent[:, 1],
                "reference_pseudotime_transferred": query_mapped.obs["reference_pseudotime_transferred"].to_numpy(dtype=float),
            }
        )
        query_latent_df.to_csv(sample_dir / "query_latent_2d.tsv", sep="\t", index=False)
        query_umap_df = transform_query_to_reference_umap(query_latent, ref_latent_full_df, reference_umap_df)
        query_umap_df.insert(0, "cell_id", query_mapped.obs_names.astype(str))
        query_umap_df["reference_pseudotime_transferred"] = query_mapped.obs["reference_pseudotime_transferred"].to_numpy(dtype=float)
        if "timepoint_std" in query_mapped.obs.columns:
            query_umap_df["timepoint_std"] = query_mapped.obs["timepoint_std"].astype(str).to_numpy()
        query_umap_df.to_csv(sample_dir / "query_umap_2d.tsv", sep="\t", index=False)
        transfer_df.to_csv(sample_dir / "query_cell_pseudotime_transfer.tsv", sep="\t", index=False)
        plot_reference_umap_overlay(reference_umap_df, query_umap_df, sample_id, sample_dir)
        plot_query_umap_by_transferred_pseudotime(query_umap_df, sample_id, sample_dir)

        print(f"[3/4:{sample_id}] Building mapped query metacells and scoring local clock")
        query_metacells = generate_query_metacells(
            query_mapped,
            bootstrap_iters=int(args.bootstrap_iters),
            metacell_size=int(args.metacell_size),
            seed=int(args.seed),
        )
        query_metacells = preprocess_query_metacells_for_local_clock(query_metacells, local_gene_names)
        pred_df = predict_local_clock(model_bundle, query_metacells)
        curve_df = summarize_query_curve(pred_df)
        curve_df["sample_id_std"] = sample_id
        pred_df.to_csv(sample_dir / "query_metacell_scores.tsv", sep="\t", index=False)
        curve_df.to_csv(sample_dir / "query_shared_axis_curve.tsv", sep="\t", index=False)
        plot_query_curve(curve_df, sample_dir, title=f"{sample_id}: aging score on shared reference pseudotime")

        sample_summary = {
            "sample_id_std": sample_id,
            "group_std": str(row["group"]),
            "condition_std": str(row["condition"]),
            "age_group_std": str(row["age_group_std"]),
            "n_query_cells": int(query_mapped.n_obs),
            "n_query_metacells": int(pred_df.shape[0]),
            "mean_p_old": float(pred_df["p_old"].mean()),
            "median_p_old": float(pred_df["p_old"].median()),
            "mean_transferred_pseudotime": float(pred_df["reference_pseudotime_transferred_mean"].mean()),
            "mean_transfer_confidence": float(pred_df["reference_pseudotime_transfer_confidence_mean"].mean()),
        }
        if "timepoint" in row.index:
            sample_summary["timepoint"] = float(row["timepoint"])
            sample_summary["timepoint_std"] = str(row["timepoint_std"])
        with open(sample_dir / "sample_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(sample_summary, handle, indent=2)

        sample_rows.append(sample_summary)
        curve_frames.append(curve_df)
        meta_rows.append(
            {
                "sample_id_std": sample_id,
                "group_std": str(row["group"]),
                "condition_std": str(row["condition"]),
                "age_group_std": str(row["age_group_std"]),
                **({"timepoint_std": str(row["timepoint_std"])} if "timepoint" in row.index else {}),
            }
        )

    print("[4/4] Writing cross-sample summary")
    sample_summary_df = pd.DataFrame(sample_rows).sort_values("sample_id_std").reset_index(drop=True)
    sample_summary_df.to_csv(summary_dir / "sample_summary.tsv", sep="\t", index=False)
    curve_summary_df = pd.concat(curve_frames, ignore_index=True)
    curve_summary_df.to_csv(summary_dir / "cross_sample_shared_axis_summary.tsv", sep="\t", index=False)
    sample_meta_df = pd.DataFrame(meta_rows).drop_duplicates().reset_index(drop=True)
    plot_cross_sample_summary(curve_summary_df, sample_meta_df, summary_dir)
    if str(args.query_source) == "prepared_external":
        maybe_compare_previous(summary_dir)
    else:
        compare_to_direct_injury_apply(summary_dir)

    with open(summary_dir / "summary_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "mode": str(args.mode),
                "query_source": str(args.query_source),
                "reference_atlas": str(reference_atlas),
                "training_metrics": str(training_metrics_path),
                "local_model": str(local_model_path),
                "input_dir": str(input_dir) if str(args.query_source) == "prepared_external" else None,
                "manifest": str(manifest_path) if str(args.query_source) == "prepared_external" else None,
                "query_h5ad": str(query_h5ad) if str(args.query_source) == "young_injury_h5ad" else None,
                "n_samples": int(sample_summary_df.shape[0]),
                "sample_ids": sample_summary_df["sample_id_std"].astype(str).tolist(),
            },
            handle,
            indent=2,
        )

    print("scVI reference verification complete.")
    print(f"Outputs: {outdir}")


if __name__ == "__main__":
    main()
