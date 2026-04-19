#!/usr/bin/env python3
"""
Generate standalone, high-resolution preprocessing report figures with large fonts.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import json

import anndata as ad
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import sparse

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import scanpy as sc

from path_config import ARTIFACT_ROOT, PROJECT_ROOT

ROOT = PROJECT_ROOT
FIG_DIR = ARTIFACT_ROOT / "figures" / "collection_preprocess"
PSEUDOTIME_FIG_DIR = ARTIFACT_ROOT / "figures" / "monocle3_pseudotime"
ATLAS_TRANSITION_FIG_DIR = ARTIFACT_ROOT / "figures" / "atlas_transition"
POST_MOUSE_ANALYSIS_DIR = ARTIFACT_ROOT / "post_mouse_analysis"
POST_MOUSE_CORE_FIG_DIR = ARTIFACT_ROOT / "figures" / "post_mouse_analysis_core"
POST_MOUSE_EXPLORATORY_FIG_DIR = ARTIFACT_ROOT / "figures" / "post_mouse_analysis_exploratory"
POST_HUMAN_ANALYSIS_DIR = ARTIFACT_ROOT / "human_post_analysis"
POST_HUMAN_CORE_FIG_DIR = ARTIFACT_ROOT / "figures" / "post_human_analysis_core"
POST_HUMAN_EXPLORATORY_FIG_DIR = ARTIFACT_ROOT / "figures" / "post_human_analysis_exploratory"
FIG6_HUMAN_DIR = ARTIFACT_ROOT / "figures" / "figure6_human_extension"
HUMAN_CLOCK_RESULTS_DIR = ARTIFACT_ROOT / "human_clock_outputs"
HUMAN_CLOCK_ATLAS_PATH = HUMAN_CLOCK_RESULTS_DIR / "human_musc_annotation" / "human_musc_small_atlas.h5ad"
HUMAN_SELECTED_DONOR_PATH = HUMAN_CLOCK_RESULTS_DIR / "human_musc_annotation" / "human_selected_donor_manifest.tsv"
MOUSE_CLOCK_RESULTS_DIR = ARTIFACT_ROOT / "clock_artifacts" / "0310_results" / "expression_only"
MOUSE_CLOCK_CORE_FIG_DIR = ARTIFACT_ROOT / "figures" / "mouse_clock_core"
MOUSE_CLOCK_EXPLORATORY_FIG_DIR = ARTIFACT_ROOT / "figures" / "mouse_clock_exploratory"
LOCAL_MOUSE_CLOCK_RESULTS_DIR = ARTIFACT_ROOT / "clock_artifacts" / "0310_results" / "expression_only_local_pseudotime"
LOCAL_MOUSE_CLOCK_SUPP_FIG_DIR = ARTIFACT_ROOT / "figures" / "supplementary_clock" / "pseudotime_aware_clock"
LOCAL_MOUSE_VERIFY_DIR = ARTIFACT_ROOT / "scvi_reference_verify_prepared" / "summary" / "evaluation"
OUTPUT_DIR = ARTIFACT_ROOT / "clock_artifacts" / "0310_results"
TRAINING_ATLAS_PATH = OUTPUT_DIR / "training_atlas_from_cleaned_annotated.h5ad"
ANNOTATED_ATLAS_PATH = ARTIFACT_ROOT / "clock_artifacts" / "musc_annotation" / "musc_atlas_annotated.h5ad"
MONOCLE3_CELL_PATH = ARTIFACT_ROOT / "processed_adata" / "monocle3" / "musc_monocle3_cells.tsv"
LINKAGE_MAP_PATH = ARTIFACT_ROOT / "clock_artifacts" / "donor_linkage_map.tsv"
MUSC_RUN_DIR = ARTIFACT_ROOT / "clock_artifacts" / "musc_annotation" / "runs" / "20260310_043522"
INPUT_AFTER_QC_PATH = MUSC_RUN_DIR / "00_input" / "input_after_qc.h5ad"
PASS1_BEFORE_FILTER_PATH = MUSC_RUN_DIR / "pass1" / "pass1_before_filter.h5ad"
PASS2_ANNOTATED_PATH = MUSC_RUN_DIR / "pass2" / "pass2_annotated_atlas.h5ad"
MSC_ENTROPY_RSCRIPT = Path(os.environ.get("R_SCRIPT_BIN", "Rscript"))

sys.path.insert(0, str(ROOT / "scripts"))
import mouse_workflow_core as mouse_core  # noqa: E402
sys.path.insert(0, str(Path(os.environ.get("ABCLOCK_ROOT", str(ROOT / "abclock")))))
import enrichment as abclock_enrichment  # noqa: E402

ANNOTATION_PALETTE = {
    "C0_Quiescent": "#4F7A5A",
    "C1_Activated": "#D7A84A",
    "C3_Differentiating/Proliferating": "#C65D3B",
}
PSEUDOTIME_CMAP = LinearSegmentedColormap.from_list(
    "musc_pseudotime",
    ["#EAF3FB", "#B8D5EA", "#74A9CF", "#2B7BBA", "#0B3C6D"],
)
SOURCE_DISPLAY_NAMES = {
    "GSE226907_wt": "GSE226907",
    "SKM_mouse_raw": "SKM mouse",
    "TabulaMuris_diaphragm_smartseq2": "TM diaphragm SS2",
    "TabulaMuris_limb_10x": "TM limb 10x",
    "TabulaMuris_limb_smartseq2": "TM limb SS2",
    "walter2024_main": "Walter 2024",
}
SOURCE_PALETTE = {
    "GSE226907_wt": "#1B9E77",
    "SKM_mouse_raw": "#D95F02",
    "TabulaMuris_diaphragm_smartseq2": "#7570B3",
    "TabulaMuris_limb_10x": "#E7298A",
    "TabulaMuris_limb_smartseq2": "#66A61E",
    "walter2024_main": "#E6AB02",
}
YOUNG_COLOR = "#4F7A5A"
OLD_COLOR = "#C65D3B"
SHARED_COLOR = "#7C8DA6"
NEUTRAL_COLOR = "#2F4858"
TIMEPOINT_COLOR_MAP = {
    "d0": "#4F7A5A",
    "d1": "#D17C48",
    "d2": "#C65D3B",
    "d3.5": "#8F4C7A",
    "d5": "#4C78A8",
    "d7": "#2F4858",
}
TIMEPOINT_ORDER = ["d0", "d1", "d2", "d3.5", "d5", "d7"]
STATE_DISPLAY_NAMES = {
    "C0_Quiescent": "Quiescent",
    "C1_Activated": "Activated",
    "C3_Differentiating/Proliferating": "Diff/Prolif",
}
HUMAN_PATTERN_PALETTE = {
    "C0_Mixed": "#7C8DA6",
    "C1_Mixed": "#4C78A8",
    "C2_Old_Enriched": "#C65D3B",
}


def _exploratory_type_dir(name: str) -> Path:
    return POST_MOUSE_EXPLORATORY_FIG_DIR / name


def _human_exploratory_type_dir(name: str) -> Path:
    return POST_HUMAN_EXPLORATORY_FIG_DIR / name


# ------------------------------------------------------------------------------
# DATA COLLECTION
# ------------------------------------------------------------------------------
def collect_raw_and_filtered_counts() -> pd.DataFrame:
    paths = mouse_core.MouseTrainingPaths()
    adatas = mouse_core.step_load_mouse_datasets(paths)
    adatas = mouse_core.step_harmonize_mouse_datasets(adatas, paths)

    rows = []
    for name, adata in adatas.items():
        filtered = mouse_core.filter_musc(adata, name)
        rows.append(
            {
                "source_name": name,
                "raw_cells": int(adata.n_obs),
                "musc_cells": int(filtered.n_obs),
            }
        )
    return pd.DataFrame(rows).sort_values("raw_cells", ascending=True)


def collect_training_atlas_composition() -> pd.DataFrame:
    # Use backed='r' to only load metadata, saving massive memory and time
    atlas = sc.read_h5ad(TRAINING_ATLAS_PATH, backed="r")
    comp = atlas.obs.groupby(["source", "Age_group_std"]).size().unstack(fill_value=0).reset_index()
    comp["donor_groups"] = comp["source"].map(atlas.obs.groupby("source")["donor_split_id"].nunique())
    comp["total"] = comp["old"] + comp["young"]
    if getattr(atlas, "file", None) is not None:
        atlas.file.close()
    return comp.sort_values("total", ascending=True)


def collect_linkage_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    linkage = pd.read_csv(LINKAGE_MAP_PATH, sep="\t")
    linked = linkage.loc[linkage["is_linked_across_sources"]].copy()
    return linkage, linked


def _decode_h5ad_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray([value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values], dtype=object)


def _read_h5ad_obs_column(obs_group: h5py.Group, column: str, n_obs: int) -> np.ndarray:
    if column not in obs_group:
        return np.full(n_obs, "unknown", dtype=object)

    obj = obs_group[column]
    if isinstance(obj, h5py.Dataset):
        return _decode_h5ad_strings(obj[()])

    categories = _decode_h5ad_strings(obj["categories"][()])
    codes = obj["codes"][()]
    return np.asarray([categories[int(code)] if code >= 0 else "unknown" for code in codes], dtype=object)


def collect_monocle3_overlay() -> tuple[pd.DataFrame, pd.DataFrame]:
    with h5py.File(ANNOTATED_ATLAS_PATH, "r") as handle:
        obs_group = handle["obs"]
        umap_coords = handle["obsm"]["X_umap"][()]
        cell_ids = _decode_h5ad_strings(obs_group["_index"][()])
        n_obs = len(cell_ids)
        annotation = _read_h5ad_obs_column(obs_group, "annotation", n_obs)
        source = _read_h5ad_obs_column(obs_group, "source", n_obs)
        pseudotime_atlas = np.asarray(
            pd.to_numeric(_read_h5ad_obs_column(obs_group, "pseudotime_monocle3", n_obs), errors="coerce"),
            dtype=float,
        )

    overlay = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "umap_1": umap_coords[:, 0],
            "umap_2": umap_coords[:, 1],
            "annotation_atlas": annotation,
            "source": source,
            "pseudotime_atlas": pseudotime_atlas,
        }
    )

    monocle = pd.read_csv(MONOCLE3_CELL_PATH, sep="\t")
    monocle["cell_id"] = monocle["cell_id"].astype(str)
    monocle["annotation"] = monocle["annotation"].astype(str)
    monocle["pseudotime"] = pd.to_numeric(monocle["pseudotime"], errors="coerce")
    merged = overlay.merge(monocle, on="cell_id", how="inner", validate="one_to_one")

    merged["annotation_final"] = merged["annotation"].where(merged["annotation"].notna(), merged["annotation_atlas"])
    merged["pseudotime_final"] = merged["pseudotime"].where(merged["pseudotime"].notna(), merged["pseudotime_atlas"])

    pt_min = float(merged["pseudotime_final"].min())
    pt_max = float(merged["pseudotime_final"].max())
    merged["pseudotime_0to1"] = (merged["pseudotime_final"] - pt_min) / max(pt_max - pt_min, 1e-9)
    merged["pseudotime_0to1"] = merged["pseudotime_0to1"].clip(0.0, 1.0)

    merged["pt_bin"] = pd.qcut(merged["pseudotime_0to1"], q=24, labels=False, duplicates="drop")
    backbone = (
        merged.groupby("pt_bin")
        .agg(
            n_cells=("cell_id", "size"),
            pseudotime_0to1=("pseudotime_0to1", "mean"),
            umap_1=("umap_1", "mean"),
            umap_2=("umap_2", "mean"),
        )
        .reset_index(drop=True)
        .sort_values("pseudotime_0to1")
    )
    backbone = backbone.loc[backbone["n_cells"] >= 20].copy()
    backbone["umap_1_smooth"] = backbone["umap_1"].rolling(window=3, center=True, min_periods=1).mean()
    backbone["umap_2_smooth"] = backbone["umap_2"].rolling(window=3, center=True, min_periods=1).mean()

    return merged, backbone


def _normalize_log1p_inplace(adata: sc.AnnData, target_sum: float = 1e4) -> None:
    X = adata.X
    if hasattr(X, "tocsr"):
        X = X.tocsr(copy=True)
        row_sums = np.asarray(X.sum(axis=1)).ravel().astype(float)
        scale = np.divide(float(target_sum), row_sums, out=np.zeros_like(row_sums), where=row_sums > 0)
        X = sparse.diags(scale) @ X
        X.data = np.log1p(X.data)
        adata.X = X
        return

    X = np.asarray(X, dtype=float)
    row_sums = X.sum(axis=1, keepdims=True)
    scale = np.divide(float(target_sum), row_sums, out=np.zeros_like(row_sums), where=row_sums > 0)
    adata.X = np.log1p(X * scale)


def collect_musc_transition_panels() -> dict[str, object]:
    input_adata = sc.read_h5ad(INPUT_AFTER_QC_PATH)
    if "counts" in input_adata.layers:
        input_adata.X = input_adata.layers["counts"].copy()
    _normalize_log1p_inplace(input_adata, target_sum=1e4)
    sc.pp.highly_variable_genes(input_adata, n_top_genes=min(2000, input_adata.n_vars), subset=True)
    sc.tl.pca(input_adata, svd_solver="arpack")
    input_panel = pd.DataFrame(
        {
            "x": input_adata.obsm["X_pca"][:, 0],
            "y": input_adata.obsm["X_pca"][:, 1],
            "source": input_adata.obs["donor_split_id"].astype(str).str.split("::").str[0].to_numpy(),
        }
    )

    with h5py.File(PASS1_BEFORE_FILTER_PATH, "r") as handle:
        obs_group = handle["obs"]
        latent = handle["obsm"]["X_scVI_pass1"][()]
        donor_split = _read_h5ad_obs_column(obs_group, "donor_split_id", latent.shape[0])
    pass1_umap = _compute_umap_with_r(latent, n_neighbors=30, min_dist=0.35, seed=42)
    pass1_panel = pd.DataFrame(
        {
            "x": pass1_umap[:, 0],
            "y": pass1_umap[:, 1],
            "source": pd.Series(donor_split).astype(str).str.split("::").str[0].to_numpy(),
        }
    )

    with h5py.File(PASS2_ANNOTATED_PATH, "r") as handle:
        obs_group = handle["obs"]
        umap_coords = handle["obsm"]["X_umap"][()]
        phate_coords = handle["obsm"]["X_phate"][()]
        annotation = _read_h5ad_obs_column(obs_group, "annotation", len(umap_coords))

    final_umap_panel = pd.DataFrame(
        {
            "x": umap_coords[:, 0],
            "y": umap_coords[:, 1],
            "annotation": annotation,
        }
    )
    final_phate_panel = pd.DataFrame(
        {
            "x": phate_coords[:, 0],
            "y": phate_coords[:, 1],
            "annotation": annotation,
        }
    )

    return {
        "input_umap": input_panel,
        "pass1_scvi_umap": pass1_panel,
        "final_scvi_umap": final_umap_panel,
        "final_phate": final_phate_panel,
        "counts": {
            "input_after_qc": int(input_panel.shape[0]),
            "pass1_before_filter": int(pass1_panel.shape[0]),
            "pass2_annotated": int(final_umap_panel.shape[0]),
            "final_phate": int(final_phate_panel.shape[0]),
        },
    }


def collect_post_mouse_summary() -> pd.DataFrame:
    summary_path = POST_MOUSE_ANALYSIS_DIR / "combined_state_summary.tsv"
    summary = pd.read_csv(summary_path, sep="\t")
    summary["state_display"] = summary["state"].map(STATE_DISPLAY_NAMES).fillna(summary["state"])
    return summary


def collect_post_mouse_state_tables(summary_df: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame | Path]]:
    state_tables: dict[str, dict[str, pd.DataFrame | Path]] = {}
    for row in summary_df.itertuples(index=False):
        state_dir = Path(row.state_dir)
        age_dir = state_dir / "age_comparison"
        state_tables[str(row.state)] = {
            "state_dir": state_dir,
            "go_old_up": pd.read_csv(age_dir / "go_old_up.tsv", sep="\t"),
            "go_young_up": pd.read_csv(age_dir / "go_young_up.tsv", sep="\t"),
            "go_old_regulon_targets": pd.read_csv(age_dir / "go_old_regulon_targets.tsv", sep="\t"),
            "go_young_regulon_targets": pd.read_csv(age_dir / "go_young_regulon_targets.tsv", sep="\t"),
            "regulon_overlap": pd.read_csv(age_dir / "regulon_overlap.tsv", sep="\t"),
            "markers": pd.read_csv(age_dir / "young_vs_old_markers.tsv", sep="\t"),
        }
    return state_tables


def collect_post_human_summary() -> pd.DataFrame:
    summary_path = POST_HUMAN_ANALYSIS_DIR / "post_analysis_summary.tsv"
    summary = pd.read_csv(summary_path, sep="\t")
    summary["analysis_label"] = "Human MuSC atlas"
    return summary


def collect_human_clock_results() -> dict[str, pd.DataFrame | dict]:
    with open(HUMAN_CLOCK_RESULTS_DIR / "showcase_training_metrics.json", "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    atlas = ad.read_h5ad(HUMAN_CLOCK_ATLAS_PATH)
    obs = atlas.obs.copy()
    umap = np.asarray(atlas.obsm["X_umap"], dtype=float)
    atlas_df = pd.DataFrame(
        {
            "umap_1": umap[:, 0],
            "umap_2": umap[:, 1],
            "sample_id_std": obs["sample_id_std"].astype(str).to_numpy(),
            "Age_group_std": obs["Age_group_std"].astype(str).to_numpy(),
            "age_range_std": obs["age_range_std"].astype(str).to_numpy(),
            "Sex_std": obs["Sex_std"].astype(str).to_numpy(),
            "age_pattern_annotation": obs["age_pattern_annotation"].astype(str).to_numpy(),
        }
    )
    return {
        "metrics": metrics,
        "showcase_split_results": pd.read_csv(HUMAN_CLOCK_RESULTS_DIR / "showcase_split_results.tsv", sep="\t"),
        "male_only_split_results": pd.read_csv(HUMAN_CLOCK_RESULTS_DIR / "male_only_split_results.tsv", sep="\t"),
        "showcase_donor_predictions": pd.read_csv(HUMAN_CLOCK_RESULTS_DIR / "showcase_donor_predictions.tsv", sep="\t"),
        "showcase_threshold_diagnostics": pd.read_csv(HUMAN_CLOCK_RESULTS_DIR / "showcase_threshold_diagnostics.tsv", sep="\t"),
        "gene_weights": pd.read_csv(HUMAN_CLOCK_RESULTS_DIR / "gene_weights.tsv", sep="\t"),
        "enrichment_aging": pd.read_csv(HUMAN_CLOCK_RESULTS_DIR / "enrichment_aging.tsv", sep="\t"),
        "enrichment_youth": pd.read_csv(HUMAN_CLOCK_RESULTS_DIR / "enrichment_youth.tsv", sep="\t"),
        "holdout_donor_summary": pd.read_csv(HUMAN_CLOCK_RESULTS_DIR / "showcase_holdout_donor_summary.tsv", sep="\t"),
        "selected_donor_manifest": pd.read_csv(HUMAN_SELECTED_DONOR_PATH, sep="\t"),
        "atlas": atlas_df,
    }


def collect_post_human_tables() -> dict[str, pd.DataFrame]:
    age_dir = POST_HUMAN_ANALYSIS_DIR / "age_comparison"
    return {
        "go_old_up": pd.read_csv(age_dir / "go_old_up.tsv", sep="\t"),
        "go_young_up": pd.read_csv(age_dir / "go_young_up.tsv", sep="\t"),
        "go_old_regulon_targets": pd.read_csv(age_dir / "go_old_regulon_targets.tsv", sep="\t"),
        "go_young_regulon_targets": pd.read_csv(age_dir / "go_young_regulon_targets.tsv", sep="\t"),
        "regulon_overlap": pd.read_csv(age_dir / "regulon_overlap.tsv", sep="\t"),
        "markers": pd.read_csv(age_dir / "young_vs_old_markers.tsv", sep="\t"),
    }


def collect_mouse_clock_results() -> dict[str, pd.DataFrame | dict]:
    return {
        "weights": pd.read_csv(MOUSE_CLOCK_RESULTS_DIR / "gene_weights.tsv", sep="\t"),
        "donor_predictions": pd.read_csv(MOUSE_CLOCK_RESULTS_DIR / "donor_predictions.tsv", sep="\t"),
        "holdout_predictions": pd.read_csv(MOUSE_CLOCK_RESULTS_DIR / "source_holdout_donor_predictions.tsv", sep="\t"),
        "threshold_compare": pd.read_csv(MOUSE_CLOCK_RESULTS_DIR / "source_holdout_threshold_setting_comparison.tsv", sep="\t"),
        "metrics": json.loads((MOUSE_CLOCK_RESULTS_DIR / "training_metrics.json").read_text()),
        "split": pd.read_csv(MOUSE_CLOCK_RESULTS_DIR / "split_diagnostics.tsv", sep="\t"),
    }


def collect_local_mouse_clock_results() -> dict[str, pd.DataFrame | dict]:
    return {
        "weights": pd.read_csv(LOCAL_MOUSE_CLOCK_RESULTS_DIR / "gene_weights.tsv", sep="\t"),
        "donor_predictions": pd.read_csv(LOCAL_MOUSE_CLOCK_RESULTS_DIR / "donor_predictions.tsv", sep="\t"),
        "holdout_predictions": pd.read_csv(LOCAL_MOUSE_CLOCK_RESULTS_DIR / "source_holdout_donor_predictions.tsv", sep="\t"),
        "thresholds": pd.read_csv(LOCAL_MOUSE_CLOCK_RESULTS_DIR / "source_holdout_thresholds.tsv", sep="\t"),
        "window_summary": pd.read_csv(LOCAL_MOUSE_CLOCK_RESULTS_DIR / "local_window_summary.tsv", sep="\t"),
        "window_summary_full": pd.read_csv(LOCAL_MOUSE_CLOCK_RESULTS_DIR / "local_window_summary_full.tsv", sep="\t"),
        "local_gene_weights": pd.read_csv(LOCAL_MOUSE_CLOCK_RESULTS_DIR / "local_gene_weights.tsv", sep="\t"),
        "metacell_predictions": pd.read_csv(LOCAL_MOUSE_CLOCK_RESULTS_DIR / "metacell_predictions.tsv", sep="\t"),
        "metrics": json.loads((LOCAL_MOUSE_CLOCK_RESULTS_DIR / "training_metrics.json").read_text()),
        "split": pd.read_csv(LOCAL_MOUSE_CLOCK_RESULTS_DIR / "split_diagnostics.tsv", sep="\t"),
    }


def collect_local_verification_results() -> dict[str, pd.DataFrame]:
    return {
        "sample_summary": pd.read_csv(LOCAL_MOUSE_VERIFY_DIR / "sample_summary.tsv", sep="\t"),
        "curve_summary": pd.read_csv(LOCAL_MOUSE_VERIFY_DIR / "cross_sample_shared_axis_summary.tsv", sep="\t"),
        "comparison_to_direct": pd.read_csv(LOCAL_MOUSE_VERIFY_DIR / "comparison_to_direct_apply.tsv", sep="\t"),
    }


def collect_mouse_clock_go_results(clock_results: dict[str, pd.DataFrame | dict], top_n: int = 100) -> dict[str, object]:
    weights = clock_results["weights"].copy()
    old_genes = weights.sort_values("Weight", ascending=False).head(top_n)["Gene"].astype(str).str.capitalize().tolist()
    young_genes = weights.sort_values("Weight", ascending=True).head(top_n)["Gene"].astype(str).str.capitalize().tolist()

    old_go = abclock_enrichment.run_enrichment_analysis(old_genes, gmt_path=str(mouse_core.POST_PATHS.gmt_path), organism="Mouse")
    young_go = abclock_enrichment.run_enrichment_analysis(young_genes, gmt_path=str(mouse_core.POST_PATHS.gmt_path), organism="Mouse")
    return {
        "old_genes": old_genes,
        "young_genes": young_genes,
        "old_go": old_go,
        "young_go": young_go,
        "top_n": int(top_n),
    }


def _clean_go_term(term: str) -> str:
    term = str(term)
    term = term.split(" (GO")[0]
    term = term.replace("GOBP_", "")
    term = term.replace("_", " ").strip()
    return term.title()


def _wrap_label(text: str, width: int = 28) -> str:
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        proposal = word if not current else f"{current} {word}"
        if len(proposal) <= width:
            current = proposal
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", transparent=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return path


def _compute_umap_with_r(latent: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.35, seed: int = 42) -> np.ndarray:
    rscript = str(MSC_ENTROPY_RSCRIPT if MSC_ENTROPY_RSCRIPT.exists() else "Rscript")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        latent_path = tmpdir_path / "latent.tsv"
        output_path = tmpdir_path / "umap.tsv"
        script_path = tmpdir_path / "run_umap.R"

        np.savetxt(latent_path, latent, delimiter="\t")
        script_path.write_text(
            textwrap.dedent(
                f"""
                suppressPackageStartupMessages(library(uwot))
                latent <- as.matrix(read.table("{latent_path}", sep="\\t", header=FALSE, check.names=FALSE))
                umap_coords <- uwot::umap(
                  latent,
                  n_neighbors = {int(n_neighbors)},
                  min_dist = {float(min_dist)},
                  metric = "euclidean",
                  n_threads = 1,
                  ret_model = FALSE,
                  verbose = FALSE,
                  seed = {int(seed)}
                )
                write.table(
                  umap_coords,
                  file = "{output_path}",
                  sep = "\\t",
                  row.names = FALSE,
                  col.names = FALSE,
                  quote = FALSE
                )
                """
            ),
            encoding="utf-8",
        )
        subprocess.run([rscript, str(script_path)], check=True, capture_output=True, text=True)
        return pd.read_csv(output_path, sep="\t", header=None).to_numpy(dtype=float)


def _top_terms(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    work = df.copy().head(top_n)
    if "Term" in work.columns:
        work["Term_clean"] = work["Term"].map(_clean_go_term).map(lambda x: _wrap_label(x, width=26))
    return work


def _parse_overlap_count(value: object) -> float:
    text = str(value)
    if "/" in text:
        left = text.split("/", 1)[0]
        try:
            return float(left)
        except ValueError:
            return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _scale_bubble_sizes(values: np.ndarray, *, global_min: float, global_max: float, min_size: float = 70.0, max_size: float = 260.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        return np.full(values.shape, min_size, dtype=float)
    if not np.isfinite(global_min) or not np.isfinite(global_max) or global_max <= global_min:
        return np.full(values.shape, (min_size + max_size) / 2.0, dtype=float)
    scaled = (values - global_min) / (global_max - global_min)
    scaled = np.clip(scaled, 0.0, 1.0)
    return min_size + scaled * (max_size - min_size)


# ------------------------------------------------------------------------------
# PUBLICATION PLOTTING (Standalone Large Format)
# ------------------------------------------------------------------------------
def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_raw_vs_retained(raw_df: pd.DataFrame) -> Path:
    fig_path = FIG_DIR / "01_raw_vs_musc_retained.pdf"

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(raw_df))

    ax.barh(y_pos, raw_df["raw_cells"], color="#E0E0E0", label="Total Sequenced Cells")
    ax.barh(y_pos, raw_df["musc_cells"], color="#3E6B48", label="Retained MuSCs")

    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(raw_df["source_name"])
    ax.set_xlabel("Cell Count (Log Scale)")
    ax.set_title("MuSC Distillation Efficiency", fontweight="bold", pad=20)

    ax.legend(frameon=True, edgecolor="black", loc="lower right")

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return fig_path


def plot_training_atlas_composition(atlas_df: pd.DataFrame) -> Path:
    fig_path = FIG_DIR / "02_atlas_age_composition.pdf"

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos_age = np.arange(len(atlas_df))

    ax.barh(y_pos_age, atlas_df["young"], color="#DCC48E", label="Young")
    ax.barh(y_pos_age, atlas_df["old"], left=atlas_df["young"], color="#B55D3D", label="Old")

    ax.set_yticks(y_pos_age)
    ax.set_yticklabels(atlas_df["source"])
    ax.set_xlabel("Number of MuSCs in Training Atlas")
    ax.set_title("Age Composition & Donor Counts", fontweight="bold", pad=20)

    ax.legend(frameon=True, edgecolor="black", loc="lower right")

    for y, total, donors in zip(y_pos_age, atlas_df["total"], atlas_df["donor_groups"]):
        ax.text(total + (atlas_df["total"].max() * 0.02), y, f"n={donors} donors", va="center", color="#333333", fontsize=12)

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return fig_path


def plot_linkage_summary(linkage: pd.DataFrame, linked: pd.DataFrame) -> Path:
    fig_path = FIG_DIR / "03_donor_linkage_donut.pdf"

    fig, ax = plt.subplots(figsize=(8, 8))
    sizes = [int(linked.shape[0]), int(linkage.shape[0] - linked.shape[0])]

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=["Cross-Source\nLinked", "Unlinked"],
        colors=["#5E8C61", "#EFEFEF"],
        startangle=90,
        autopct="%1.1f%%",
        pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
        textprops={"fontsize": 16},
    )
    plt.setp(autotexts, size=16, weight="bold", color="#333333")

    ax.text(0, 0, f"Total\nDonors\n{sum(sizes)}", ha="center", va="center", fontsize=18, fontweight="bold")
    ax.set_title("Sample Linkage Identity", fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return fig_path


def plot_monocle3_backbone(overlay: pd.DataFrame, backbone: pd.DataFrame) -> Path:
    fig_path = PSEUDOTIME_FIG_DIR / "01_monocle3_pseudotime_backbone_umap.pdf"

    fig, axs = plt.subplots(1, 3, figsize=(21, 7), gridspec_kw={"width_ratios": [1.0, 1.0, 1.08]})

    def format_umap_axis(ax: plt.Axes) -> None:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_frame_on(False)

    state_handles = []
    source_handles = []

    for ax in axs:
        ax.scatter(overlay["umap_1"], overlay["umap_2"], s=8, c="#D9D9D9", alpha=0.22, linewidths=0, zorder=1)
        format_umap_axis(ax)

    for label, color in ANNOTATION_PALETTE.items():
        sub = overlay.loc[overlay["annotation_final"] == label]
        if sub.empty:
            continue
        axs[0].scatter(sub["umap_1"], sub["umap_2"], s=10, c=color, alpha=0.9, linewidths=0, zorder=2)
        state_handles.append(
            Line2D([0], [0], marker="o", linestyle="", markersize=8, markerfacecolor=color, markeredgewidth=0, label=label)
        )
    axs[0].set_title("MuSC State UMAP", fontweight="bold", pad=16)

    source_order = [source for source in SOURCE_DISPLAY_NAMES if source in overlay["source"].unique()]
    for source in source_order:
        sub = overlay.loc[overlay["source"] == source]
        if sub.empty:
            continue
        axs[1].scatter(sub["umap_1"], sub["umap_2"], s=10, c=SOURCE_PALETTE[source], alpha=0.88, linewidths=0, zorder=2)
        source_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=8,
                markerfacecolor=SOURCE_PALETTE[source],
                markeredgewidth=0,
                label=SOURCE_DISPLAY_NAMES[source],
            )
        )
    axs[1].set_title("Study-Level UMAP", fontweight="bold", pad=16)

    axs[2].plot(backbone["umap_1_smooth"], backbone["umap_2_smooth"], color="white", linewidth=6, zorder=3)
    axs[2].plot(backbone["umap_1_smooth"], backbone["umap_2_smooth"], color="#1F2933", linewidth=3.2, zorder=4)

    sc_pt = axs[2].scatter(
        overlay["umap_1"],
        overlay["umap_2"],
        s=10,
        c=overlay["pseudotime_0to1"],
        cmap=PSEUDOTIME_CMAP,
        alpha=0.9,
        linewidths=0,
    )
    axs[2].set_title("Pseudotime with Trajectory Backbone", fontweight="bold", pad=16)

    start = backbone.iloc[0]
    end = backbone.iloc[-1]
    axs[2].annotate(
        "Start",
        xy=(start["umap_1_smooth"], start["umap_2_smooth"]),
        xytext=(-18, 12),
        textcoords="offset points",
        fontsize=12,
        fontweight="bold",
        color="#1F2933",
        arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#1F2933"),
    )
    axs[2].annotate(
        "Late",
        xy=(end["umap_1_smooth"], end["umap_2_smooth"]),
        xytext=(10, -16),
        textcoords="offset points",
        fontsize=12,
        fontweight="bold",
        color="#1F2933",
        arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#1F2933"),
    )

    axs[0].legend(
        handles=state_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.4,
    )
    axs[1].legend(
        handles=source_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.4,
    )

    cbar = fig.colorbar(sc_pt, ax=axs[2], fraction=0.046, pad=0.04)
    cbar.set_label("Pseudotime (0-1 scaled)", rotation=90, labelpad=12)

    fig.subplots_adjust(left=0.03, right=0.93, top=0.88, bottom=0.24, wspace=0.08)
    fig.savefig(fig_path, bbox_inches="tight", transparent=True)
    fig.savefig(fig_path.with_suffix(".png"), dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return fig_path


def plot_pseudotime_by_annotation(overlay: pd.DataFrame) -> Path:
    fig_path = PSEUDOTIME_FIG_DIR / "02_monocle3_pseudotime_by_state.pdf"
    order = (
        overlay.groupby("annotation_final")["pseudotime_0to1"]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    data = [overlay.loc[overlay["annotation_final"] == label, "pseudotime_0to1"].to_numpy() for label in order]
    parts = ax.violinplot(data, vert=False, showmeans=False, showmedians=False, showextrema=False)
    for body, label in zip(parts["bodies"], order):
        body.set_facecolor(ANNOTATION_PALETTE.get(label, "#999999"))
        body.set_edgecolor("white")
        body.set_alpha(0.9)

    for idx, label in enumerate(order, start=1):
        vals = overlay.loc[overlay["annotation_final"] == label, "pseudotime_0to1"].to_numpy()
        q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
        ax.plot([q1, q3], [idx, idx], color="#1F2933", lw=3, solid_capstyle="round")
        ax.scatter([med], [idx], color="white", edgecolor="#1F2933", s=40, zorder=3)

    ax.set_yticks(np.arange(1, len(order) + 1))
    ax.set_yticklabels(order)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Pseudotime (0-1 scaled)")
    ax.set_title("Pseudotime Progression by MuSC State", fontweight="bold", pad=16)

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight", transparent=True)
    fig.savefig(fig_path.with_suffix(".png"), dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return fig_path


def plot_musc_umap_transition(panels: dict[str, object]) -> list[Path]:
    counts = panels["counts"]
    outputs: list[Path] = []

    def _format_axis(ax: plt.Axes) -> None:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_frame_on(False)

    input_panel = panels["input_umap"]
    pass1_panel = panels["pass1_scvi_umap"]
    final_umap = panels["final_scvi_umap"]
    final_phate = panels["final_phate"]

    input_sources = [source for source in SOURCE_DISPLAY_NAMES if source in set(input_panel["source"])]
    source_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=SOURCE_PALETTE[source],
            markeredgewidth=0,
            label=SOURCE_DISPLAY_NAMES[source],
        )
        for source in input_sources
    ]

    fig, ax = plt.subplots(figsize=(6.0, 5.8))
    _format_axis(ax)
    for source in input_sources:
        sub = input_panel.loc[input_panel["source"] == source]
        ax.scatter(sub["x"], sub["y"], s=8, c=SOURCE_PALETTE[source], alpha=0.88, linewidths=0)
    ax.set_title(f"MuSC-Filtered PCA\nn = {counts['input_after_qc']:,}", fontweight="bold", pad=14)
    ax.legend(handles=source_handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, columnspacing=1.0, handletextpad=0.4)
    outputs.append(_save_figure(fig, ATLAS_TRANSITION_FIG_DIR / "01a_musc_filtered_pca.pdf"))

    pass1_sources = [source for source in SOURCE_DISPLAY_NAMES if source in set(pass1_panel["source"])]
    pass1_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=SOURCE_PALETTE[source],
            markeredgewidth=0,
            label=SOURCE_DISPLAY_NAMES[source],
        )
        for source in pass1_sources
    ]
    fig, ax = plt.subplots(figsize=(6.0, 5.8))
    _format_axis(ax)
    for source in pass1_sources:
        sub = pass1_panel.loc[pass1_panel["source"] == source]
        ax.scatter(sub["x"], sub["y"], s=8, c=SOURCE_PALETTE[source], alpha=0.88, linewidths=0)
    ax.set_title(f"Initial scVI UMAP\nn = {counts['pass1_before_filter']:,}", fontweight="bold", pad=14)
    ax.legend(handles=pass1_handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, columnspacing=1.0, handletextpad=0.4)
    outputs.append(_save_figure(fig, ATLAS_TRANSITION_FIG_DIR / "01b_initial_scvi_umap.pdf"))

    state_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=8, markerfacecolor=color, markeredgewidth=0, label=label)
        for label, color in ANNOTATION_PALETTE.items()
    ]
    fig, ax = plt.subplots(figsize=(6.0, 5.8))
    _format_axis(ax)
    for label, color in ANNOTATION_PALETTE.items():
        sub = final_umap.loc[final_umap["annotation"] == label]
        if sub.empty:
            continue
        ax.scatter(sub["x"], sub["y"], s=9, c=color, alpha=0.9, linewidths=0)
    ax.set_title(f"Filtered scVI UMAP\nn = {counts['pass2_annotated']:,}", fontweight="bold", pad=14)
    ax.legend(handles=state_handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, columnspacing=1.0, handletextpad=0.4)
    outputs.append(_save_figure(fig, ATLAS_TRANSITION_FIG_DIR / "01c_filtered_scvi_umap.pdf"))

    fig, ax = plt.subplots(figsize=(6.0, 5.8))
    _format_axis(ax)
    for label, color in ANNOTATION_PALETTE.items():
        sub = final_phate.loc[final_phate["annotation"] == label]
        if sub.empty:
            continue
        ax.scatter(sub["x"], sub["y"], s=9, c=color, alpha=0.9, linewidths=0)
    ax.set_title(f"Final PHATE Trajectory\nn = {counts['final_phate']:,}", fontweight="bold", pad=14)
    ax.legend(handles=state_handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, columnspacing=1.0, handletextpad=0.4)
    outputs.append(_save_figure(fig, ATLAS_TRANSITION_FIG_DIR / "01d_final_phate_trajectory.pdf"))

    return outputs


def plot_post_mouse_state_summary(summary_df: pd.DataFrame) -> Path:
    fig_path = POST_MOUSE_CORE_FIG_DIR / "01_state_summary_overview.pdf"
    order = summary_df.sort_values("young_cells", ascending=True).reset_index(drop=True)
    y = np.arange(len(order))

    fig, axs = plt.subplots(1, 3, figsize=(15, 6), gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]})

    axs[0].barh(y, order["young_cells"], color=YOUNG_COLOR, alpha=0.9, label="Young")
    axs[0].barh(y, order["old_cells"], left=order["young_cells"], color=OLD_COLOR, alpha=0.9, label="Old")
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(order["state_display"])
    axs[0].set_title("State Cell Counts", fontweight="bold", pad=12)
    axs[0].set_xlabel("Cells")
    axs[0].legend(frameon=False, loc="lower right")

    x = np.arange(len(order))
    width = 0.35
    axs[1].bar(x - width / 2, order["young_regulons"], width=width, color=YOUNG_COLOR, alpha=0.9, label="Young")
    axs[1].bar(x + width / 2, order["old_regulons"], width=width, color=OLD_COLOR, alpha=0.9, label="Old")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(order["state_display"], rotation=20, ha="right")
    axs[1].set_title("Regulon Counts", fontweight="bold", pad=12)
    axs[1].set_ylabel("Regulons")

    axs[2].bar(x, order["shared_tfs"], color=SHARED_COLOR, alpha=0.95, label="Shared TFs")
    axs[2].bar(x, order["young_only_tfs"], bottom=order["shared_tfs"], color=YOUNG_COLOR, alpha=0.9, label="Young-only TFs")
    axs[2].bar(
        x,
        order["old_only_tfs"],
        bottom=order["shared_tfs"] + order["young_only_tfs"],
        color=OLD_COLOR,
        alpha=0.9,
        label="Old-only TFs",
    )
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(order["state_display"], rotation=20, ha="right")
    axs[2].set_title("Regulatory Architecture", fontweight="bold", pad=12)
    axs[2].set_ylabel("TF categories")
    axs[2].legend(frameon=False, loc="upper right")

    fig.tight_layout()
    return _save_figure(fig, fig_path)


def _plot_go_overview(summary_df: pd.DataFrame, state_tables: dict[str, dict[str, pd.DataFrame | Path]], *, mode: str) -> Path:
    fig_path = POST_MOUSE_CORE_FIG_DIR / (
        "02_state_de_go_overview.pdf" if mode == "de" else "03_state_regulon_go_overview.pdf"
    )
    n_states = len(summary_df)
    fig, axs = plt.subplots(n_states, 2, figsize=(11.5, 4.3 * n_states), squeeze=False)
    all_overlap_values: list[float] = []
    for row in summary_df.itertuples(index=False):
        tables = state_tables[str(row.state)]
        old_df = tables["go_old_up"] if mode == "de" else tables["go_old_regulon_targets"]
        young_df = tables["go_young_up"] if mode == "de" else tables["go_young_regulon_targets"]
        for df in (old_df, young_df):
            top_df = _top_terms(df, top_n=5)
            if "Overlap" in top_df.columns and not top_df.empty:
                all_overlap_values.extend(top_df["Overlap"].map(_parse_overlap_count).tolist())
    finite_overlaps = np.asarray([value for value in all_overlap_values if np.isfinite(value)], dtype=float)
    global_overlap_min = float(finite_overlaps.min()) if finite_overlaps.size else 1.0
    global_overlap_max = float(finite_overlaps.max()) if finite_overlaps.size else 1.0

    for row_idx, row in enumerate(summary_df.itertuples(index=False)):
        tables = state_tables[str(row.state)]
        old_df = tables["go_old_up"] if mode == "de" else tables["go_old_regulon_targets"]
        young_df = tables["go_young_up"] if mode == "de" else tables["go_young_regulon_targets"]
        for col_idx, (df, color, title_suffix) in enumerate(
            [
                (old_df, OLD_COLOR, "Upregulated in old" if mode == "de" else "Old-enriched regulon targets"),
                (young_df, YOUNG_COLOR, "Upregulated in young" if mode == "de" else "Young-enriched regulon targets"),
            ]
        ):
            ax = axs[row_idx, col_idx]
            top_df = _top_terms(df, top_n=5)
            if top_df.empty:
                ax.text(0.5, 0.5, "No terms", ha="center", va="center")
                ax.set_axis_off()
                continue
            values = top_df["LogP"].to_numpy(dtype=float)
            overlap = top_df["Overlap"].map(_parse_overlap_count).to_numpy(dtype=float) if "Overlap" in top_df.columns else np.ones(len(top_df))
            ypos = np.arange(len(top_df))
            bubble_sizes = _scale_bubble_sizes(
                overlap,
                global_min=global_overlap_min,
                global_max=global_overlap_max,
                min_size=70.0,
                max_size=260.0,
            )
            ax.scatter(values, ypos, s=bubble_sizes, c=color, alpha=0.9, edgecolors="white", linewidths=0.8, zorder=3)
            ax.hlines(ypos, 0, values, color=color, alpha=0.35, linewidth=1.4, zorder=2)
            ax.set_yticks(ypos)
            ax.set_yticklabels(top_df["Term_clean"])
            ax.tick_params(axis="y", labelsize=11)
            ax.invert_yaxis()
            ax.set_xlabel("-log10 adj. P")
            ax.grid(axis="x", linestyle=":", alpha=0.25)
            xmin = max(float(values.min()) - 0.35, 0.0)
            xmax = float(values.max()) + 0.35
            ax.set_xlim(xmin, xmax)
            if row_idx == 0:
                ax.set_title(title_suffix, fontweight="bold", pad=12)
            if col_idx == 0:
                ax.set_ylabel(str(row.state_display), fontweight="bold", rotation=90, labelpad=22)
            if col_idx == 1:
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")

    suptitle = "State-Resolved Differential Expression GO Programs" if mode == "de" else "State-Resolved Regulon Target GO Programs"
    fig.suptitle(suptitle, fontweight="bold", y=0.995)
    legend_sizes = np.linspace(global_overlap_min, global_overlap_max, 3) if global_overlap_max > global_overlap_min else np.asarray([global_overlap_min] * 3)
    legend_handles = [
        plt.scatter(
            [],
            [],
            s=float(_scale_bubble_sizes(np.asarray([size]), global_min=global_overlap_min, global_max=global_overlap_max, min_size=70.0, max_size=260.0)[0]),
            color="#9AA5B1",
            alpha=0.7,
            edgecolors="white",
            linewidths=0.8,
            label=f"{int(round(size))} genes",
        )
        for size in legend_sizes
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        title="Overlap",
    )
    fig.subplots_adjust(left=0.15, right=0.97, top=0.92, bottom=0.11, hspace=0.42, wspace=0.18)
    return _save_figure(fig, fig_path)


def plot_post_mouse_de_go_overview(summary_df: pd.DataFrame, state_tables: dict[str, dict[str, pd.DataFrame | Path]]) -> Path:
    return _plot_go_overview(summary_df, state_tables, mode="de")


def plot_post_mouse_regulon_go_overview(summary_df: pd.DataFrame, state_tables: dict[str, dict[str, pd.DataFrame | Path]]) -> Path:
    return _plot_go_overview(summary_df, state_tables, mode="regulon")


def plot_post_mouse_regulon_architecture(summary_df: pd.DataFrame) -> Path:
    fig_path = POST_MOUSE_CORE_FIG_DIR / "04_regulon_architecture_summary.pdf"
    order = summary_df.sort_values("shared_tfs", ascending=True).reset_index(drop=True)
    x = np.arange(len(order))

    fig, axs = plt.subplots(1, 2, figsize=(13.8, 5.8))
    bars_left = [
        axs[0].bar(x - 0.18, order["young_unique_regulon_targets"], width=0.36, color=YOUNG_COLOR, alpha=0.9, label="Young-unique targets"),
        axs[0].bar(x + 0.18, order["old_unique_regulon_targets"], width=0.36, color=OLD_COLOR, alpha=0.9, label="Old-unique targets"),
    ]
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(order["state_display"], rotation=20, ha="right")
    axs[0].set_ylabel("Unique regulon target genes")
    axs[0].set_title("Age-Biased Regulon Target Burden", fontweight="bold", pad=12)

    line_young, = axs[1].plot(x, order["young_adjacencies"], marker="o", linewidth=2.5, color=YOUNG_COLOR, label="Young adjacencies")
    line_old, = axs[1].plot(x, order["old_adjacencies"], marker="o", linewidth=2.5, color=OLD_COLOR, label="Old adjacencies")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(order["state_display"], rotation=20, ha="right")
    axs[1].set_ylabel("GRN edges")
    axs[1].set_title("Network Density by State and Age", fontweight="bold", pad=12)

    legend_handles = [Patch(facecolor=YOUNG_COLOR), Patch(facecolor=OLD_COLOR), line_young, line_old]
    legend_labels = ["Young-unique targets", "Old-unique targets", "Young adjacencies", "Old adjacencies"]
    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.2, wspace=0.22)
    return _save_figure(fig, fig_path)


def plot_post_mouse_deg_mirror(summary_df: pd.DataFrame, state_tables: dict[str, dict[str, pd.DataFrame | Path]]) -> Path:
    fig_path = POST_MOUSE_CORE_FIG_DIR / "05_state_deg_mirror_overview.pdf"
    fig, axs = plt.subplots(len(summary_df), 1, figsize=(10.2, 4.3 * len(summary_df)), squeeze=False)

    for ax, row in zip(axs[:, 0], summary_df.itertuples(index=False)):
        markers_df = state_tables[str(row.state)]["markers"].copy()

        old_df = markers_df.loc[markers_df["direction"] == "old_up"].copy()
        young_df = markers_df.loc[markers_df["direction"] == "young_up"].copy()
        if "pvals_adj" in old_df.columns:
            old_df = old_df.loc[old_df["pvals_adj"].fillna(1.0) <= 0.05]
            young_df = young_df.loc[young_df["pvals_adj"].fillna(1.0) <= 0.05]

        old_df = old_df.sort_values("logfoldchanges", ascending=False).head(7).copy()
        young_df = young_df.sort_values("logfoldchanges", ascending=False).head(7).copy()

        old_pairs = [(str(gene), float(val)) for gene, val in zip(old_df["names"], old_df["logfoldchanges"])]
        young_pairs = [(str(gene), -float(val)) for gene, val in zip(young_df["names"], young_df["logfoldchanges"])]

        labels = [gene for gene, _ in reversed(young_pairs)] + [gene for gene, _ in old_pairs]
        values = [val for _, val in reversed(young_pairs)] + [val for _, val in old_pairs]
        colors = [YOUNG_COLOR] * len(young_pairs) + [OLD_COLOR] * len(old_pairs)
        ypos = np.arange(len(labels))

        ax.barh(ypos, values, color=colors, alpha=0.92)
        ax.axvline(0, color="#1F2933", linewidth=1.2)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_title(f"{row.state_display}: Top Differential Genes", fontweight="bold", pad=8)
        ax.set_xlabel("log fold change (young <- 0 -> old)")

        max_abs = max(abs(min(values, default=-1.0)), abs(max(values, default=1.0)))
        ax.set_xlim(-max_abs * 1.06, max_abs * 1.06)

        left_label_x = -max_abs * 1.01
        right_label_x = max_abs * 1.01
        ax.text(left_label_x, len(labels) - 0.4, "Young-up", color=YOUNG_COLOR, fontweight="bold", ha="left", va="bottom")
        ax.text(right_label_x, len(labels) - 0.4, "Old-up", color=OLD_COLOR, fontweight="bold", ha="right", va="bottom")

    fig.suptitle("State-Specific Differential Gene Programs", fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.93, bottom=0.06, hspace=0.48)
    return _save_figure(fig, fig_path)


def plot_state_marker_overview(state: str, state_display: str, markers_df: pd.DataFrame) -> Path:
    fig_path = _exploratory_type_dir("markers") / f"{state}__markers_overview.pdf"
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, direction, color, title in [
        (axs[0], "old_up", OLD_COLOR, "Old-up markers"),
        (axs[1], "young_up", YOUNG_COLOR, "Young-up markers"),
    ]:
        sub = markers_df.loc[markers_df["direction"] == direction].copy()
        if "pvals_adj" in sub.columns:
            sub = sub.loc[sub["pvals_adj"].fillna(1.0) <= 0.05]
        sub = sub.sort_values("logfoldchanges", ascending=False).head(10)
        ypos = np.arange(len(sub))
        ax.barh(ypos, sub["logfoldchanges"].to_numpy(dtype=float), color=color, alpha=0.92)
        ax.set_yticks(ypos)
        ax.set_yticklabels(sub["names"].astype(str))
        ax.invert_yaxis()
        ax.set_title(f"{state_display}: {title}", fontweight="bold", pad=10)
        ax.set_xlabel("log fold change")

    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_state_go_pair(state: str, state_display: str, old_df: pd.DataFrame, young_df: pd.DataFrame, outname: str, title_prefix: str) -> Path:
    folder = "de_go" if outname == "de_go_comparison" else "regulon_go"
    fig_path = _exploratory_type_dir(folder) / f"{state}__{outname}.pdf"
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 6.2))
    for ax, df, color, suffix in [
        (
            axs[0],
            old_df,
            OLD_COLOR,
            "Upregulated in old" if outname == "de_go_comparison" else "Old-enriched regulon targets",
        ),
        (
            axs[1],
            young_df,
            YOUNG_COLOR,
            "Upregulated in young" if outname == "de_go_comparison" else "Young-enriched regulon targets",
        ),
    ]:
        top_df = _top_terms(df, top_n=10)
        ypos = np.arange(len(top_df))
        ax.barh(ypos, top_df["LogP"].to_numpy(dtype=float), color=color, alpha=0.92)
        ax.set_yticks(ypos)
        ax.set_yticklabels(top_df["Term_clean"])
        ax.invert_yaxis()
        ax.set_xlabel("-log10 adj. P")
        ax.set_title(f"{state_display}: {suffix}", fontweight="bold", pad=10)
    fig.suptitle(title_prefix, fontweight="bold", y=1.02)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.08, wspace=0.18)
    return _save_figure(fig, fig_path)


def plot_state_tf_overlap(state: str, state_display: str, regulon_df: pd.DataFrame) -> Path:
    fig_path = _exploratory_type_dir("tf_overlap") / f"{state}__tf_overlap_summary.pdf"
    counts = {
        "Shared TFs": int(((~regulon_df["young_only_regulon"]) & (~regulon_df["old_only_regulon"])).sum()),
        "Young-only TFs": int(regulon_df["young_only_regulon"].sum()),
        "Old-only TFs": int(regulon_df["old_only_regulon"].sum()),
    }
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [SHARED_COLOR, YOUNG_COLOR, OLD_COLOR]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    bars = ax.bar(labels, values, color=colors, alpha=0.92)
    ax.set_title(f"{state_display}: TF Overlap Structure", fontweight="bold", pad=12)
    ax.set_ylabel("Number of TFs")
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.02, str(value), ha="center", va="bottom")
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_state_regulon_target_burden(state: str, state_display: str, summary_row: pd.Series) -> Path:
    fig_path = _exploratory_type_dir("target_burden") / f"{state}__regulon_target_burden.pdf"
    labels = ["Young-unique", "Old-unique"]
    values = [
        float(summary_row["young_unique_regulon_targets"]),
        float(summary_row["old_unique_regulon_targets"]),
    ]
    colors = [YOUNG_COLOR, OLD_COLOR]
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    bars = ax.bar(labels, values, color=colors, alpha=0.92)
    ax.set_title(f"{state_display}: Unique Regulon Targets", fontweight="bold", pad=12)
    ax.set_ylabel("Genes")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.02, f"{int(value)}", ha="center", va="bottom")
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_all_state_marker_overview(summary_df: pd.DataFrame, state_tables: dict[str, dict[str, pd.DataFrame | Path]]) -> Path:
    fig_path = _exploratory_type_dir("markers") / "00_all_states__markers_overview.pdf"
    fig, axs = plt.subplots(len(summary_df), 2, figsize=(13.5, 4.4 * len(summary_df)), squeeze=False)

    for row_idx, row in enumerate(summary_df.itertuples(index=False)):
        markers_df = state_tables[str(row.state)]["markers"]
        for col_idx, (direction, color, title) in enumerate(
            [("old_up", OLD_COLOR, "Old-up"), ("young_up", YOUNG_COLOR, "Young-up")]
        ):
            ax = axs[row_idx, col_idx]
            sub = markers_df.loc[markers_df["direction"] == direction].copy()
            if "pvals_adj" in sub.columns:
                sub = sub.loc[sub["pvals_adj"].fillna(1.0) <= 0.05]
            sub = sub.sort_values("logfoldchanges", ascending=False).head(8)
            ypos = np.arange(len(sub))
            ax.barh(ypos, sub["logfoldchanges"].to_numpy(dtype=float), color=color, alpha=0.92)
            ax.set_yticks(ypos)
            ax.set_yticklabels(sub["names"].astype(str))
            ax.invert_yaxis()
            ax.set_xlabel("log fold change")
            ax.set_title(f"{row.state_display}: {title}", fontweight="bold", pad=8)

    fig.tight_layout(h_pad=2.2, w_pad=2.2)
    return _save_figure(fig, fig_path)


def plot_all_state_tf_overlap(summary_df: pd.DataFrame, state_tables: dict[str, dict[str, pd.DataFrame | Path]]) -> Path:
    fig_path = _exploratory_type_dir("tf_overlap") / "00_all_states__tf_overlap_summary.pdf"
    fig, axs = plt.subplots(1, len(summary_df), figsize=(5.0 * len(summary_df), 4.8), squeeze=False)

    for ax, row in zip(axs[0], summary_df.itertuples(index=False)):
        regulon_df = state_tables[str(row.state)]["regulon_overlap"]
        counts = {
            "Shared": int(((~regulon_df["young_only_regulon"]) & (~regulon_df["old_only_regulon"])).sum()),
            "Young-only": int(regulon_df["young_only_regulon"].sum()),
            "Old-only": int(regulon_df["old_only_regulon"].sum()),
        }
        labels = list(counts.keys())
        values = list(counts.values())
        colors = [SHARED_COLOR, YOUNG_COLOR, OLD_COLOR]
        bars = ax.bar(labels, values, color=colors, alpha=0.92)
        ax.set_title(str(row.state_display), fontweight="bold", pad=10)
        ax.tick_params(axis="x", rotation=18)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.03, str(value), ha="center", va="bottom")

    axs[0][0].set_ylabel("Number of TFs")
    fig.tight_layout(w_pad=2.2)
    return _save_figure(fig, fig_path)


def plot_all_state_target_burden(summary_df: pd.DataFrame) -> Path:
    fig_path = _exploratory_type_dir("target_burden") / "00_all_states__regulon_target_burden.pdf"
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    order = summary_df.sort_values("state_display").reset_index(drop=True)
    x = np.arange(len(order))
    ax.bar(x - 0.18, order["young_unique_regulon_targets"], width=0.36, color=YOUNG_COLOR, alpha=0.92, label="Young-unique")
    ax.bar(x + 0.18, order["old_unique_regulon_targets"], width=0.36, color=OLD_COLOR, alpha=0.92, label="Old-unique")
    ax.set_xticks(x)
    ax.set_xticklabels(order["state_display"], rotation=18, ha="right")
    ax.set_ylabel("Unique regulon target genes")
    ax.set_title("Age-Biased Regulon Target Burden Across States", fontweight="bold", pad=12)
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_post_human_summary(summary_df: pd.DataFrame) -> Path:
    fig_path = POST_HUMAN_CORE_FIG_DIR / "01_human_summary_overview.pdf"
    row = summary_df.iloc[0]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))

    axs[0].bar(["Young", "Old"], [row["young_cells"], row["old_cells"]], color=[YOUNG_COLOR, OLD_COLOR], alpha=0.92)
    axs[0].set_title("Human Cell Counts", fontweight="bold", pad=12)
    axs[0].set_ylabel("Cells")

    axs[1].bar(["Young", "Old"], [row["young_regulons"], row["old_regulons"]], color=[YOUNG_COLOR, OLD_COLOR], alpha=0.92)
    axs[1].set_title("Human Regulon Counts", fontweight="bold", pad=12)
    axs[1].set_ylabel("Regulons")

    axs[2].bar(
        ["Shared TFs", "Young-only TFs", "Old-only TFs"],
        [row["shared_tfs"], row["young_only_tfs"], row["old_only_tfs"]],
        color=[SHARED_COLOR, YOUNG_COLOR, OLD_COLOR],
        alpha=0.92,
    )
    axs[2].set_title("Human Regulatory Architecture", fontweight="bold", pad=12)
    axs[2].tick_params(axis="x", rotation=18)
    axs[2].set_ylabel("TFs")

    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_post_human_go_pair(old_df: pd.DataFrame, young_df: pd.DataFrame, outdir: Path, filename: str, title_prefix: str) -> Path:
    fig_path = outdir / filename
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 6.2))
    for ax, df, color, suffix in [
        (axs[0], old_df, OLD_COLOR, "Old-biased"),
        (axs[1], young_df, YOUNG_COLOR, "Young-biased"),
    ]:
        top_df = _top_terms(df, top_n=10)
        ypos = np.arange(len(top_df))
        ax.barh(ypos, top_df["LogP"].to_numpy(dtype=float), color=color, alpha=0.92)
        ax.set_yticks(ypos)
        ax.set_yticklabels(top_df["Term_clean"])
        ax.invert_yaxis()
        ax.set_xlabel("-log10 adj. P")
        ax.set_title(f"Human MuSC: {suffix}", fontweight="bold", pad=10)
    fig.suptitle(title_prefix, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_post_human_marker_overview(markers_df: pd.DataFrame) -> Path:
    fig_path = _human_exploratory_type_dir("markers") / "00_human__markers_overview.pdf"
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 6.0))
    for ax, direction, color, title in [
        (axs[0], "old_up", OLD_COLOR, "Old-up markers"),
        (axs[1], "young_up", YOUNG_COLOR, "Young-up markers"),
    ]:
        sub = markers_df.loc[markers_df["direction"] == direction].copy()
        if "pvals_adj" in sub.columns:
            sub = sub.loc[sub["pvals_adj"].fillna(1.0) <= 0.05]
        sub = sub.sort_values("logfoldchanges", ascending=False).head(12)
        ypos = np.arange(len(sub))
        ax.barh(ypos, sub["logfoldchanges"].to_numpy(dtype=float), color=color, alpha=0.92)
        ax.set_yticks(ypos)
        ax.set_yticklabels(sub["names"].astype(str))
        ax.invert_yaxis()
        ax.set_xlabel("log fold change")
        ax.set_title(f"Human MuSC: {title}", fontweight="bold", pad=10)
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_post_human_tf_overlap(regulon_df: pd.DataFrame) -> Path:
    fig_path = _human_exploratory_type_dir("tf_overlap") / "00_human__tf_overlap_summary.pdf"
    counts = {
        "Shared TFs": int(((~regulon_df["young_only_regulon"]) & (~regulon_df["old_only_regulon"])).sum()),
        "Young-only TFs": int(regulon_df["young_only_regulon"].sum()),
        "Old-only TFs": int(regulon_df["old_only_regulon"].sum()),
    }
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [SHARED_COLOR, YOUNG_COLOR, OLD_COLOR]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    bars = ax.bar(labels, values, color=colors, alpha=0.92)
    ax.set_title("Human MuSC TF Overlap Structure", fontweight="bold", pad=12)
    ax.set_ylabel("TFs")
    ax.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.03, str(value), ha="center", va="bottom")
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_post_human_target_burden(summary_df: pd.DataFrame) -> Path:
    fig_path = _human_exploratory_type_dir("target_burden") / "00_human__regulon_target_burden.pdf"
    row = summary_df.iloc[0]
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    bars = ax.bar(
        ["Young-unique", "Old-unique"],
        [row["young_unique_regulon_targets"], row["old_unique_regulon_targets"]],
        color=[YOUNG_COLOR, OLD_COLOR],
        alpha=0.92,
    )
    ax.set_title("Human Age-Biased Regulon Targets", fontweight="bold", pad=12)
    ax.set_ylabel("Genes")
    for bar, value in zip(bars, [row["young_unique_regulon_targets"], row["old_unique_regulon_targets"]]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(row["young_unique_regulon_targets"], row["old_unique_regulon_targets"]) * 0.03, f"{int(value)}", ha="center", va="bottom")
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_human_reference_atlas(clock_results: dict[str, pd.DataFrame | dict]) -> Path:
    fig_path = FIG6_HUMAN_DIR / "01_human_reference_atlas.pdf"
    atlas = clock_results["atlas"].copy()

    donor_order = atlas.groupby("sample_id_std").size().sort_values(ascending=False).index.astype(str).tolist()
    donor_palette = plt.get_cmap("tab10")
    donor_colors = {donor: donor_palette(i % 10) for i, donor in enumerate(donor_order)}

    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.8))
    for age, color in [("young", YOUNG_COLOR), ("old", OLD_COLOR)]:
        sub = atlas.loc[atlas["Age_group_std"].astype(str) == age]
        axs[0].scatter(sub["umap_1"], sub["umap_2"], s=9, alpha=0.7, linewidths=0, color=color, label=age.capitalize())
    axs[0].set_title("Human MuSC reference by age", fontweight="bold", pad=10, fontsize=15)
    axs[0].legend(frameon=False, loc="best", fontsize=11, title_fontsize=12)

    for donor in donor_order:
        sub = atlas.loc[atlas["sample_id_std"].astype(str) == donor]
        axs[1].scatter(sub["umap_1"], sub["umap_2"], s=8, alpha=0.72, linewidths=0, color=donor_colors[donor], label=donor)
    axs[1].set_title("Human MuSC reference by donor", fontweight="bold", pad=10, fontsize=15)
    axs[1].legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", ncol=1, fontsize=9, title_fontsize=10)

    for pattern, color in HUMAN_PATTERN_PALETTE.items():
        sub = atlas.loc[atlas["age_pattern_annotation"].astype(str) == pattern]
        if sub.empty:
            continue
        axs[2].scatter(sub["umap_1"], sub["umap_2"], s=9, alpha=0.72, linewidths=0, color=color, label=pattern.replace("_", " "))
    axs[2].set_title("Human MuSC age-pattern states", fontweight="bold", pad=10, fontsize=15)
    axs[2].legend(frameon=False, loc="best", fontsize=10, title_fontsize=11)

    for ax in axs:
        ax.set_xlabel("UMAP 1", fontsize=13)
        ax.set_ylabel("UMAP 2", fontsize=13)
        ax.tick_params(labelsize=11)
    fig.suptitle("Human MuSC Reference Atlas", fontweight="bold", y=1.02, fontsize=17)
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_human_selected_donor_cohort(clock_results: dict[str, pd.DataFrame | dict]) -> Path:
    fig_path = FIG6_HUMAN_DIR / "02_human_selected_donor_cohort.pdf"
    manifest = clock_results["selected_donor_manifest"].copy()
    manifest["sample_id_std"] = manifest["sample_id_std"].astype(str)
    manifest["age_group_binary"] = manifest["age_group_binary"].astype(str)
    manifest = manifest.sort_values(["age_group_binary", "selection_rank_within_age", "n_cells_selected"], ascending=[True, True, False]).reset_index(drop=True)
    manifest["x"] = np.arange(manifest.shape[0])
    colors = [YOUNG_COLOR if x == "young" else OLD_COLOR for x in manifest["age_group_binary"].astype(str)]

    fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.8))
    axs[0].bar(manifest["x"], manifest["n_cells_raw_musc"], color=colors, alpha=0.9)
    axs[0].set_xticks(manifest["x"])
    axs[0].set_xticklabels(manifest["sample_id_std"], rotation=35, ha="right")
    axs[0].set_ylabel("Raw MuSC cells", fontsize=13)
    axs[0].set_title("Retained male donors", fontweight="bold", pad=10, fontsize=15)

    axs[1].bar(manifest["x"], manifest["n_cells_selected"], color=colors, alpha=0.9)
    axs[1].set_xticks(manifest["x"])
    axs[1].set_xticklabels(manifest["sample_id_std"], rotation=35, ha="right")
    axs[1].set_ylabel("Selected cells", fontsize=13)
    axs[1].set_title("Cells contributed to human reference", fontweight="bold", pad=10, fontsize=15)
    for ax in axs:
        ax.tick_params(labelsize=11)

    legend_handles = [
        Patch(facecolor=YOUNG_COLOR, edgecolor="none", label="Young"),
        Patch(facecolor=OLD_COLOR, edgecolor="none", label="Old"),
    ]
    fig.legend(handles=legend_handles, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), fontsize=11)
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_human_clock_gene_weights(clock_results: dict[str, pd.DataFrame | dict]) -> Path:
    fig_path = FIG6_HUMAN_DIR / "03_human_clock_gene_weights.pdf"
    weights = clock_results["gene_weights"].copy()
    top_old = weights.sort_values("Weight", ascending=False).head(15).copy()
    top_young = weights.sort_values("Weight", ascending=True).head(15).copy()
    top_old["Weight_plot"] = top_old["Weight"]
    top_young["Weight_plot"] = top_young["Weight"].abs()

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 6.2))
    for ax, df, color, title in [
        (axs[0], top_old.sort_values("Weight_plot"), OLD_COLOR, "Old-associated genes"),
        (axs[1], top_young.sort_values("Weight_plot"), YOUNG_COLOR, "Young-associated genes"),
    ]:
        ypos = np.arange(len(df))
        ax.hlines(ypos, 0, df["Weight_plot"], color=color, linewidth=2.4)
        ax.scatter(df["Weight_plot"], ypos, color=color, s=44, zorder=3)
        ax.set_yticks(ypos)
        ax.set_yticklabels(df["Gene"].astype(str))
        ax.set_xlabel("|Coefficient|", fontsize=13)
        ax.set_title(title, fontweight="bold", pad=10, fontsize=15)
        ax.tick_params(labelsize=11)

    fig.suptitle("Human Elastic-Net Clock Gene Weights", fontweight="bold", y=1.02, fontsize=17)
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_human_clock_showcase_donor_scores(clock_results: dict[str, pd.DataFrame | dict]) -> Path:
    fig_path = FIG6_HUMAN_DIR / "04_human_clock_showcase_donor_scores.pdf"
    donor = clock_results["showcase_donor_predictions"].copy().sort_values("mean_p_old").reset_index(drop=True)
    threshold = float(clock_results["metrics"]["selected_threshold"])
    donor["x"] = np.arange(donor.shape[0])
    colors = np.where(donor["age_group"].astype(str) == "old", OLD_COLOR, YOUNG_COLOR)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.scatter(donor["x"], donor["mean_p_old"], s=95, c=colors, edgecolors="white", linewidths=0.9, zorder=3)
    ax.axhline(threshold, color="#1F2933", linestyle="--", linewidth=1.8, label=f"Train-opt = {threshold:.3f}")
    ax.set_xticks(donor["x"])
    ax.set_xticklabels(donor["donor_id"].astype(str))
    ax.set_ylabel("Mean donor p_old", fontsize=13)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Human showcase holdout donors", fontweight="bold", pad=12, fontsize=15)
    legend_handles = [
        Patch(facecolor=YOUNG_COLOR, edgecolor="none", label="Young donor"),
        Patch(facecolor=OLD_COLOR, edgecolor="none", label="Old donor"),
    ]
    legend = ax.legend(handles=legend_handles, frameon=False, loc="upper left", fontsize=11)
    ax.add_artist(legend)
    ax.legend(frameon=False, loc="lower right", fontsize=11)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_human_clock_threshold_behavior(clock_results: dict[str, pd.DataFrame | dict]) -> Path:
    fig_path = FIG6_HUMAN_DIR / "05_human_clock_threshold_behavior.pdf"
    threshold_df = clock_results["showcase_threshold_diagnostics"].copy().sort_values("threshold")
    train_threshold = float(clock_results["metrics"]["selected_threshold"])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(threshold_df["threshold"], threshold_df["balanced_accuracy"], color="#8F4C7A", linewidth=2.4, marker="o")
    ax.axvline(0.5, color="#6B7280", linestyle=":", linewidth=1.3, label="0.5")
    ax.axvline(train_threshold, color="#2B6CB0", linestyle="--", linewidth=1.6, label=f"Train-opt = {train_threshold:.3f}")
    ax.set_xlabel("Threshold", fontsize=13)
    ax.set_ylabel("Balanced accuracy", fontsize=13)
    ax.set_ylim(0.45, 1.02)
    ax.set_title("Human showcase threshold behavior", fontweight="bold", pad=12, fontsize=15)
    ax.legend(frameon=False, loc="lower left", fontsize=11)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_human_clock_metric_summary(clock_results: dict[str, pd.DataFrame | dict]) -> Path:
    fig_path = FIG6_HUMAN_DIR / "06_human_clock_metric_summary.pdf"
    split_results = clock_results["male_only_split_results"].copy()
    showcase = clock_results["showcase_split_results"].copy().iloc[0]

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axs[0].scatter(
        split_results["holdout_metacell_auc"],
        split_results["holdout_metacell_balanced_accuracy_thr_opt"],
        s=55,
        alpha=0.8,
        c=np.where(split_results["dominance_checks_passed"].astype(bool), "#2B6CB0", "#9CA3AF"),
        edgecolors="white",
        linewidths=0.6,
    )
    axs[0].scatter(
        [float(showcase["holdout_metacell_auc"])],
        [float(showcase["holdout_metacell_balanced_accuracy_thr_opt"])],
        s=120,
        c="#C05621",
        edgecolors="white",
        linewidths=1.0,
        zorder=4,
    )
    axs[0].set_xlabel("Holdout metacell AUC", fontsize=13)
    axs[0].set_ylabel("Holdout metacell balanced accuracy", fontsize=13)
    axs[0].set_xlim(-0.02, 1.02)
    axs[0].set_ylim(-0.02, 1.02)
    axs[0].set_title("All donor-disjoint candidate splits", fontweight="bold", pad=10, fontsize=15)
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="#2B6CB0", label="Dominance checks passed", markersize=8),
        Line2D([0], [0], marker="o", linestyle="", color="#9CA3AF", label="Other candidate split", markersize=8),
        Line2D([0], [0], marker="o", linestyle="", color="#C05621", label="Selected showcase split", markersize=9),
    ]
    axs[0].legend(handles=legend_handles, frameon=False, loc="lower left", fontsize=10)
    axs[0].tick_params(labelsize=11)

    metric_labels = ["Train AUC", "Holdout\nMetacell AUC", "Holdout\nMetacell BA", "Holdout\nDonor BA"]
    metric_values = [
        float(showcase["train_auc_train_only"]),
        float(showcase["holdout_metacell_auc"]),
        float(showcase["holdout_metacell_balanced_accuracy_thr_opt"]),
        float(showcase["holdout_donor_balanced_accuracy_thr_opt"]),
    ]
    axs[1].bar(metric_labels, metric_values, color=["#4C78A8", "#C05621", "#C05621", "#C05621"], alpha=0.92)
    axs[1].set_ylim(0, 1.05)
    axs[1].set_title("Selected showcase split performance", fontweight="bold", pad=10, fontsize=15)
    axs[1].tick_params(axis="x", rotation=15, labelsize=11)
    axs[1].tick_params(axis="y", labelsize=11)
    for idx, value in enumerate(metric_values):
        axs[1].text(idx, value + 0.03, f"{value:.3f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_human_clock_go_overview(clock_results: dict[str, pd.DataFrame | dict]) -> Path:
    fig_path = FIG6_HUMAN_DIR / "07_human_clock_go_overview.pdf"
    old_go = _top_terms(clock_results["enrichment_aging"], top_n=10)
    young_go = _top_terms(clock_results["enrichment_youth"], top_n=10)

    fig, axs = plt.subplots(1, 2, figsize=(13.5, 6.4))
    for ax, df, color, title in [
        (axs[0], old_go, OLD_COLOR, "Old-biased clock genes"),
        (axs[1], young_go, YOUNG_COLOR, "Young-biased clock genes"),
    ]:
        ypos = np.arange(len(df))
        ax.barh(ypos, df["LogP"].to_numpy(dtype=float), color=color, alpha=0.92)
        ax.set_yticks(ypos)
        ax.set_yticklabels(df["Term_clean"])
        ax.invert_yaxis()
        ax.set_xlabel("-log10 adj. P", fontsize=13)
        ax.set_title(title, fontweight="bold", pad=10, fontsize=15)
        ax.tick_params(labelsize=11)
    fig.suptitle("GO Programs Encoded by Human Clock Genes", fontweight="bold", y=1.02, fontsize=17)
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def _short_donor_label(donor_id: str) -> str:
    donor_id = str(donor_id)
    return donor_id.split("::")[-1].replace("Old_", "O_").replace("Young_", "Y_")


def plot_mouse_clock_gene_weights(clock_results: dict[str, pd.DataFrame | dict]) -> Path:
    fig_path = MOUSE_CLOCK_CORE_FIG_DIR / "01_mouse_clock_gene_weights.pdf"
    weights = clock_results["weights"].copy()
    metrics = clock_results["metrics"]

    top_old = weights.sort_values("Weight", ascending=False).head(15).copy()
    top_young = weights.sort_values("Weight", ascending=True).head(15).copy()
    top_old["Weight_plot"] = top_old["Weight"]
    top_young["Weight_plot"] = top_young["Weight"].abs()

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 6.2))
    for ax, df, color, title in [
        (axs[0], top_old.sort_values("Weight_plot"), OLD_COLOR, "Old-associated genes"),
        (axs[1], top_young.sort_values("Weight_plot"), YOUNG_COLOR, "Young-associated genes"),
    ]:
        ypos = np.arange(len(df))
        ax.hlines(ypos, 0, df["Weight_plot"], color=color, linewidth=2.4)
        ax.scatter(df["Weight_plot"], ypos, color=color, s=44, zorder=3)
        ax.set_yticks(ypos)
        ax.set_yticklabels(df["Gene"].astype(str))
        ax.set_xlabel("|Coefficient|")
        ax.set_title(title, fontweight="bold", pad=10)

    fig.suptitle(
        f"Elastic-Net Clock Gene Weights (n = {metrics['n_training_features']} features)",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_mouse_clock_donor_scores(clock_results: dict[str, pd.DataFrame | dict]) -> Path:
    fig_path = MOUSE_CLOCK_CORE_FIG_DIR / "02_mouse_clock_donor_scores.pdf"
    donor = clock_results["donor_predictions"].copy()
    threshold = float(clock_results["metrics"]["selected_threshold"])

    donor["x_group"] = donor["age_group"].map({"young": 0, "old": 1}).fillna(-1).astype(float)
    donor["x_jitter"] = donor.groupby("age_group").cumcount().to_numpy()
    donor["x"] = donor["x_group"] + np.linspace(-0.12, 0.12, len(donor))

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    for source in SOURCE_DISPLAY_NAMES:
        sub = donor.loc[donor["source"] == source]
        if sub.empty:
            continue
        ax.scatter(
            sub["x"],
            sub["mean_p_old"],
            s=70,
            c=SOURCE_PALETTE[source],
            alpha=0.92,
            edgecolors="white",
            linewidths=0.7,
            label=SOURCE_DISPLAY_NAMES[source],
        )

    ax.axhline(threshold, color="#1F2933", linestyle="--", linewidth=2, label=f"Train-opt = {threshold:.3f}")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Young donors", "Old donors"])
    ax.set_ylabel("Mean donor p_old")
    ax.set_title("Donor-Level Score Separation", fontweight="bold", pad=12)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_mouse_clock_threshold_transfer(clock_results: dict[str, pd.DataFrame | dict]) -> Path:
    fig_path = MOUSE_CLOCK_CORE_FIG_DIR / "03_mouse_clock_threshold_transfer.pdf"
    holdout = clock_results["holdout_predictions"].copy().sort_values("mean_p_old").reset_index(drop=True)
    threshold_df = clock_results["threshold_compare"].copy()
    holdout["label"] = holdout["donor_id"].astype(str).map(_short_donor_label)
    holdout["y"] = np.arange(len(holdout))

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    colors = np.where(holdout["age_group"].astype(str) == "old", OLD_COLOR, YOUNG_COLOR)
    ax.scatter(holdout["mean_p_old"], holdout["y"], c=colors, s=78, edgecolors="white", linewidths=0.8, zorder=3)

    for _, row in threshold_df.iterrows():
        color = {
            "fixed_0p5": "#6B7280",
            "train_split_optimal": "#2B6CB0",
            "source_recalibrated": "#C05621",
        }.get(str(row["threshold_setting"]), "#444444")
        label = {
            "fixed_0p5": "0.5",
            "train_split_optimal": "Train-opt",
            "source_recalibrated": "Source-recal",
        }.get(str(row["threshold_setting"]), str(row["threshold_setting"]))
        ax.axvline(float(row["threshold"]), color=color, linestyle="--", linewidth=2, label=f"{label}: {float(row['threshold']):.3f}")

    ax.set_yticks(holdout["y"])
    ax.set_yticklabels(holdout["label"])
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Mean donor p_old")
    ax.set_title("Threshold Transfer on SKM Mouse Holdout", fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_mouse_clock_metric_summary(clock_results: dict[str, pd.DataFrame | dict]) -> Path:
    fig_path = MOUSE_CLOCK_CORE_FIG_DIR / "04_mouse_clock_metric_summary.pdf"
    metrics = clock_results["metrics"]
    threshold_df = clock_results["threshold_compare"].copy()

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
    metric_labels = ["AUC", "Balanced\nAccuracy"]
    internal = [float(metrics["test_auc"]), float(metrics["test_balanced_accuracy_mean"])]
    external = [float(metrics["article_holdout_auc"]), float(metrics["article_holdout_balanced_accuracy"])]
    x = np.arange(len(metric_labels))
    axs[0].bar(x - 0.18, internal, width=0.36, color="#2B6CB0", alpha=0.92, label="Donor split")
    axs[0].bar(x + 0.18, external, width=0.36, color="#C05621", alpha=0.92, label="Source holdout")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(metric_labels)
    axs[0].set_ylim(0, 1.05)
    axs[0].set_title("Internal vs External Benchmark", fontweight="bold", pad=10)
    axs[0].legend(frameon=False)

    x2 = np.arange(len(threshold_df))
    axs[1].bar(x2, threshold_df["balanced_accuracy"].to_numpy(dtype=float), color=["#6B7280", "#2B6CB0", "#C05621"], alpha=0.92)
    axs[1].set_xticks(x2)
    axs[1].set_xticklabels(["0.5", "Train-opt", "Source-recal"])
    axs[1].set_ylim(0, 1.05)
    axs[1].set_ylabel("Balanced accuracy")
    axs[1].set_title("Threshold Sensitivity on Source Holdout", fontweight="bold", pad=10)
    for idx, row in enumerate(threshold_df.itertuples(index=False)):
        axs[1].text(idx, float(row.balanced_accuracy) + 0.03, f"{float(row.threshold):.3f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_mouse_clock_weight_distribution(clock_results: dict[str, pd.DataFrame | dict]) -> Path:
    fig_path = MOUSE_CLOCK_EXPLORATORY_FIG_DIR / "01_mouse_clock_weight_distribution.pdf"
    weights = clock_results["weights"]["Weight"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.hist(weights, bins=60, color=SHARED_COLOR, alpha=0.9, edgecolor="white")
    ax.axvline(0, color="#1F2933", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Gene count")
    ax.set_title("Distribution of Clock Coefficients", fontweight="bold", pad=12)
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_mouse_clock_sparsity_summary(clock_results: dict[str, pd.DataFrame | dict]) -> Path:
    fig_path = MOUSE_CLOCK_EXPLORATORY_FIG_DIR / "02_mouse_clock_sparsity_summary.pdf"
    w = clock_results["weights"]["Weight"].to_numpy(dtype=float)
    counts = {
        "Positive": int((w > 0).sum()),
        "Zero": int((w == 0).sum()),
        "Negative": int((w < 0).sum()),
    }
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    bars = ax.bar(list(counts.keys()), list(counts.values()), color=[OLD_COLOR, SHARED_COLOR, YOUNG_COLOR], alpha=0.92)
    ax.set_ylabel("Features")
    ax.set_title("Elastic-Net Sparsity Pattern", fontweight="bold", pad=12)
    for bar, value in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(counts.values()) * 0.02, str(value), ha="center", va="bottom")
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_mouse_clock_holdout_ranked_donors(clock_results: dict[str, pd.DataFrame | dict]) -> Path:
    fig_path = MOUSE_CLOCK_EXPLORATORY_FIG_DIR / "03_mouse_clock_holdout_ranked_donors.pdf"
    holdout = clock_results["holdout_predictions"].copy().sort_values("mean_p_old").reset_index(drop=True)
    holdout["label"] = holdout["donor_id"].astype(str).map(_short_donor_label)
    holdout["y"] = np.arange(len(holdout))
    colors = np.where(holdout["age_group"].astype(str) == "old", OLD_COLOR, YOUNG_COLOR)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.scatter(holdout["mean_p_old"], holdout["y"], c=colors, s=78, edgecolors="white", linewidths=0.8)
    ax.set_yticks(holdout["y"])
    ax.set_yticklabels(holdout["label"])
    ax.set_xlabel("Mean donor p_old")
    ax.set_title("Ranked Source-Holdout Donor Scores", fontweight="bold", pad=12)
    ax.set_xlim(-0.02, 1.02)
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_mouse_clock_go_overview(clock_go: dict[str, object]) -> Path:
    fig_path = MOUSE_CLOCK_CORE_FIG_DIR / "05_mouse_clock_go_overview.pdf"
    old_go = _top_terms(clock_go["old_go"], top_n=8)
    young_go = _top_terms(clock_go["young_go"], top_n=8)
    old_go.to_csv(MOUSE_CLOCK_CORE_FIG_DIR / "mouse_clock_go_old_biased.tsv", sep="\t", index=False)
    young_go.to_csv(MOUSE_CLOCK_CORE_FIG_DIR / "mouse_clock_go_young_biased.tsv", sep="\t", index=False)

    overlap_values = []
    for df in (old_go, young_go):
        if "Overlap" in df.columns:
            overlap_values.extend(df["Overlap"].map(_parse_overlap_count).tolist())
    finite_overlap = np.asarray([x for x in overlap_values if np.isfinite(x)], dtype=float)
    overlap_min = float(finite_overlap.min()) if finite_overlap.size else 1.0
    overlap_max = float(finite_overlap.max()) if finite_overlap.size else 1.0

    fig, axs = plt.subplots(1, 2, figsize=(10.0, 6.0))
    for ax, df, color, title in [
        (axs[0], old_go, OLD_COLOR, "Old-biased clock genes"),
        (axs[1], young_go, YOUNG_COLOR, "Young-biased clock genes"),
    ]:
        values = df["LogP"].to_numpy(dtype=float)
        overlap = df["Overlap"].map(_parse_overlap_count).to_numpy(dtype=float) if "Overlap" in df.columns else np.ones(len(df))
        sizes = _scale_bubble_sizes(overlap, global_min=overlap_min, global_max=overlap_max, min_size=70.0, max_size=260.0)
        ypos = np.arange(len(df))
        ax.scatter(values, ypos, s=sizes, c=color, alpha=0.9, edgecolors="white", linewidths=0.8, zorder=3)
        ax.hlines(ypos, values.min() - 0.2, values, color=color, alpha=0.35, linewidth=1.4, zorder=2)
        ax.set_yticks(ypos)
        ax.set_yticklabels(df["Term_clean"])
        ax.tick_params(axis="y", labelsize=11)
        ax.invert_yaxis()
        ax.set_xlabel("-log10 adj. P")
        ax.set_title(title, fontweight="bold", pad=10)
        xmin = max(float(values.min()) - 0.35, 0.0)
        xmax = float(values.max()) + 0.35
        ax.set_xlim(xmin, xmax)
        ax.grid(axis="x", linestyle=":", alpha=0.25)

    legend_sizes = np.linspace(overlap_min, overlap_max, 3) if overlap_max > overlap_min else np.asarray([overlap_min] * 3)
    legend_handles = [
        plt.scatter(
            [],
            [],
            s=float(_scale_bubble_sizes(np.asarray([size]), global_min=overlap_min, global_max=overlap_max, min_size=70.0, max_size=260.0)[0]),
            color="#9AA5B1",
            alpha=0.7,
            edgecolors="white",
            linewidths=0.8,
            label=f"{int(round(size))} genes",
        )
        for size in legend_sizes
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005), title="Overlap")
    fig.suptitle(f"GO Programs Encoded by Clock Genes (top {clock_go['top_n']} per direction)", fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.9, bottom=0.18, wspace=0.42)
    return _save_figure(fig, fig_path)


def plot_mouse_clock_go_driver_genes(clock_go: dict[str, object]) -> Path:
    fig_path = MOUSE_CLOCK_CORE_FIG_DIR / "06_mouse_clock_go_driver_genes.pdf"
    rows = []
    for direction, df, color in [
        ("Old-biased", clock_go["old_go"], OLD_COLOR),
        ("Young-biased", clock_go["young_go"], YOUNG_COLOR),
    ]:
        top_df = df.head(3).copy()
        for _, row in top_df.iterrows():
            genes = [gene.strip() for gene in str(row.get("Genes", "")).split(";") if gene.strip()]
            for gene in genes[:8]:
                rows.append(
                    {
                        "direction": direction,
                        "term": _clean_go_term(str(row["Term"])),
                        "gene": gene,
                        "color": color,
                    }
                )
    driver_df = pd.DataFrame(rows)
    driver_df.to_csv(MOUSE_CLOCK_CORE_FIG_DIR / "mouse_clock_go_driver_genes.tsv", sep="\t", index=False)

    fig, axs = plt.subplots(2, 1, figsize=(7.3, 7.1))
    for ax, direction, color in [
        (axs[0], "Old-biased", OLD_COLOR),
        (axs[1], "Young-biased", YOUNG_COLOR),
    ]:
        sub = driver_df.loc[driver_df["direction"] == direction].copy()
        if sub.empty:
            ax.text(0.5, 0.5, "No driver genes", ha="center", va="center")
            ax.set_axis_off()
            continue
        sub["term_wrap"] = sub["term"].map(lambda x: _wrap_label(x, width=24))
        terms = list(dict.fromkeys(sub["term_wrap"].tolist()))
        genes = list(dict.fromkeys(sub["gene"].tolist()))
        term_to_y = {term: idx for idx, term in enumerate(terms)}
        gene_to_x = {gene: idx for idx, gene in enumerate(genes)}
        ax.scatter(
            [gene_to_x[g] for g in sub["gene"]],
            [term_to_y[t] for t in sub["term_wrap"]],
            s=65,
            c=color,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.7,
        )
        ax.set_xticks(range(len(genes)))
        ax.set_xticklabels(genes, rotation=65, ha="right")
        ax.set_yticks(range(len(terms)))
        ax.set_yticklabels(terms)
        ax.set_title(f"{direction} clock drivers", fontweight="bold", pad=10)
        ax.invert_yaxis()
    fig.suptitle("Leading Clock Genes Driving Enriched GO Terms", fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.22, right=0.96, top=0.91, bottom=0.08, hspace=0.58)
    return _save_figure(fig, fig_path)


def _local_top_gene_table(local_clock_results: dict[str, pd.DataFrame | dict], top_n: int = 15) -> pd.DataFrame:
    weights = local_clock_results["weights"].copy()
    weights["abs_weight"] = weights["Weight"].abs()
    return weights.sort_values("abs_weight", ascending=False).head(top_n).copy()


def _local_gene_change_table(local_clock_results: dict[str, pd.DataFrame | dict]) -> pd.DataFrame:
    local_weights = local_clock_results["local_gene_weights"].copy()
    summary = (
        local_weights.groupby("Gene")
        .agg(
            min_weight=("Weight", "min"),
            max_weight=("Weight", "max"),
            mean_weight=("Weight", "mean"),
            std_weight=("Weight", "std"),
        )
        .reset_index()
    )
    summary["std_weight"] = summary["std_weight"].fillna(0.0)
    summary["weight_range"] = summary["max_weight"] - summary["min_weight"]
    return summary.sort_values(["weight_range", "std_weight", "Gene"], ascending=[False, False, True]).reset_index(drop=True)


def plot_local_clock_gene_weights(local_clock_results: dict[str, pd.DataFrame | dict]) -> Path:
    fig_path = LOCAL_MOUSE_CLOCK_SUPP_FIG_DIR / "01_local_clock_gene_weights.pdf"
    weights = local_clock_results["weights"].copy()
    metrics = local_clock_results["metrics"]

    top_old = weights.sort_values("Weight", ascending=False).head(15).copy()
    top_young = weights.sort_values("Weight", ascending=True).head(15).copy()
    top_old["Weight_plot"] = top_old["Weight"]
    top_young["Weight_plot"] = top_young["Weight"].abs()

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 6.2))
    for ax, df, color, title in [
        (axs[0], top_old.sort_values("Weight_plot"), OLD_COLOR, "Old-associated genes"),
        (axs[1], top_young.sort_values("Weight_plot"), YOUNG_COLOR, "Young-associated genes"),
    ]:
        ypos = np.arange(len(df))
        ax.hlines(ypos, 0, df["Weight_plot"], color=color, linewidth=2.4)
        ax.scatter(df["Weight_plot"], ypos, color=color, s=44, zorder=3)
        ax.set_yticks(ypos)
        ax.set_yticklabels(df["Gene"].astype(str))
        ax.set_xlabel("|Mean coefficient|")
        ax.set_title(title, fontweight="bold", pad=10)

    fig.suptitle(
        f"Local Pseudotime Clock Gene Weights (n = {metrics['n_training_features']} features)",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_local_clock_donor_scores(local_clock_results: dict[str, pd.DataFrame | dict]) -> Path:
    fig_path = LOCAL_MOUSE_CLOCK_SUPP_FIG_DIR / "02_local_clock_donor_scores.pdf"
    donor = local_clock_results["donor_predictions"].copy()
    threshold = float(local_clock_results["metrics"]["selected_threshold"])

    donor["x_group"] = donor["age_group"].map({"young": 0, "old": 1}).fillna(-1).astype(float)
    donor["x"] = donor["x_group"] + np.linspace(-0.12, 0.12, len(donor))

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    for source in SOURCE_DISPLAY_NAMES:
        sub = donor.loc[donor["source"] == source]
        if sub.empty:
            continue
        ax.scatter(
            sub["x"],
            sub["mean_p_old"],
            s=70,
            c=SOURCE_PALETTE[source],
            alpha=0.92,
            edgecolors="white",
            linewidths=0.7,
            label=SOURCE_DISPLAY_NAMES[source],
        )

    ax.axhline(threshold, color="#1F2933", linestyle="--", linewidth=2, label=f"Global train-opt = {threshold:.3f}")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Young donors", "Old donors"])
    ax.set_ylabel("Mean donor p_old")
    ax.set_title("Local Clock Donor-Level Score Separation", fontweight="bold", pad=12)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_local_clock_threshold_behavior(local_clock_results: dict[str, pd.DataFrame | dict]) -> Path:
    threshold_df = local_clock_results["thresholds"].copy()
    window_summary = local_clock_results["window_summary"].copy().sort_values("window_id").reset_index(drop=True)
    train_threshold = float(local_clock_results["metrics"]["selected_threshold"])
    fig_path = LOCAL_MOUSE_CLOCK_SUPP_FIG_DIR / "03_local_clock_threshold_behavior.pdf"

    rows = []
    for row in window_summary.itertuples(index=False):
        sub = threshold_df.loc[threshold_df["window_id"].astype(int) == int(row.window_id)].copy()
        if sub.empty:
            continue
        selected = sub.loc[np.isclose(sub["threshold"].astype(float), float(row.selected_threshold), atol=1e-9)].copy()
        if selected.empty:
            nearest_idx = (sub["threshold"].astype(float) - float(row.selected_threshold)).abs().idxmin()
            selected = sub.loc[[nearest_idx]].copy()
        selected_row = selected.iloc[0]
        rows.append(
            {
                "window_id": int(row.window_id),
                "window_label": f"W{int(row.window_id)}",
                "center_pseudotime": float(row.center_pseudotime),
                "selected_threshold": float(row.selected_threshold),
                "selected_balanced_accuracy": float(selected_row["balanced_accuracy"]),
            }
        )
    summary_df = pd.DataFrame(rows).sort_values("center_pseudotime").reset_index(drop=True)

    fig, axs = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True)

    axs[0].plot(
        summary_df["center_pseudotime"],
        summary_df["selected_balanced_accuracy"],
        color="#8F4C7A",
        linewidth=2.4,
        marker="o",
        markersize=6,
    )
    for row in summary_df.itertuples(index=False):
        axs[0].text(
            float(row.center_pseudotime),
            float(row.selected_balanced_accuracy) + 0.015,
            str(row.window_label),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axs[0].set_ylabel("Balanced accuracy")
    axs[0].set_ylim(0.5, 1.02)
    axs[0].set_title("Selected-Threshold Accuracy by Pseudotime Window", fontweight="bold", pad=10)
    axs[0].grid(axis="y", linestyle=":", alpha=0.22)

    axs[1].plot(
        summary_df["center_pseudotime"],
        summary_df["selected_threshold"],
        color="#C05621",
        linewidth=2.4,
        marker="o",
        markersize=6,
    )
    for row in summary_df.itertuples(index=False):
        axs[1].text(
            float(row.center_pseudotime),
            float(row.selected_threshold) + 0.03,
            str(row.window_label),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axs[1].axhline(0.5, color="#6B7280", linestyle=":", linewidth=1.3, label="0.5")
    axs[1].axhline(train_threshold, color="#2B6CB0", linestyle="--", linewidth=1.5, label=f"Global train-opt = {train_threshold:.3f}")
    axs[1].set_xlabel("Window center pseudotime")
    axs[1].set_ylabel("Selected threshold")
    axs[1].set_ylim(0.0, 1.02)
    axs[1].set_title("Selected Threshold by Pseudotime Window", fontweight="bold", pad=10)
    axs[1].grid(axis="y", linestyle=":", alpha=0.22)
    axs[1].legend(frameon=False, loc="lower right")

    fig.suptitle("Local Threshold Behavior Summary", fontweight="bold", y=0.995)
    fig.tight_layout()
    _save_figure(fig, fig_path)

    diag_path = LOCAL_MOUSE_CLOCK_SUPP_FIG_DIR / "03b_local_clock_threshold_scan_diagnostics.pdf"

    window_ids = window_summary["window_id"].astype(int).tolist()
    n_windows = len(window_ids)
    fig, axes = plt.subplots(n_windows, 1, figsize=(8.4, 2.15 * n_windows), sharex=True)
    if n_windows == 1:
        axes = [axes]

    for ax, window_id in zip(axes, window_ids):
        sub = threshold_df.loc[threshold_df["window_id"].astype(int) == int(window_id)].copy()
        sub = sub.sort_values("threshold")
        row = window_summary.loc[window_summary["window_id"].astype(int) == int(window_id)].iloc[0]
        selected_threshold = float(row["selected_threshold"])
        center = float(row["center_pseudotime"])

        if sub.empty:
            ax.text(0.5, 0.5, "No threshold table", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        ax.plot(
            sub["threshold"],
            sub["balanced_accuracy"],
            color="#8F4C7A",
            linewidth=2.2,
        )
        ax.axvline(0.5, color="#6B7280", linestyle=":", linewidth=1.3)
        ax.axvline(train_threshold, color="#2B6CB0", linestyle="--", linewidth=1.5)
        ax.axvline(selected_threshold, color="#C05621", linestyle="--", linewidth=1.8)

        peak_idx = int(sub["balanced_accuracy"].astype(float).idxmax())
        peak = sub.loc[peak_idx]
        ax.scatter(
            [float(peak["threshold"])],
            [float(peak["balanced_accuracy"])],
            color="#8F4C7A",
            s=34,
            zorder=3,
        )
        ax.text(
            0.02,
            0.88,
            f"W{window_id}  center={center:.2f}  sel={selected_threshold:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            ha="left",
            va="top",
        )
        ax.set_ylim(0.45, 1.02)
        ax.set_ylabel("Bal. acc.")
        ax.grid(axis="y", linestyle=":", alpha=0.22)

    axes[-1].set_xlabel("Threshold")
    fig.suptitle("Window-Specific Threshold Scan Diagnostics", fontweight="bold", y=0.995)
    legend_handles = [
        Line2D([0], [0], color="#8F4C7A", linewidth=2.2, label="Balanced accuracy"),
        Line2D([0], [0], color="#6B7280", linestyle=":", linewidth=1.3, label="0.5"),
        Line2D([0], [0], color="#2B6CB0", linestyle="--", linewidth=1.5, label=f"Global train-opt = {train_threshold:.3f}"),
        Line2D([0], [0], color="#C05621", linestyle="--", linewidth=1.8, label="Window selected"),
    ]
    fig.legend(handles=legend_handles, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=4)
    fig.subplots_adjust(top=0.92, bottom=0.12, hspace=0.28)
    _save_figure(fig, diag_path)
    return fig_path


def plot_local_clock_metric_summary(local_clock_results: dict[str, pd.DataFrame | dict]) -> Path:
    fig_path = LOCAL_MOUSE_CLOCK_SUPP_FIG_DIR / "04_local_clock_metric_summary.pdf"
    metrics = local_clock_results["metrics"]
    window_summary = local_clock_results["window_summary"].copy().sort_values("center_pseudotime")

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
    metric_labels = ["AUC", "Balanced\nAccuracy"]
    internal = [float(metrics["test_auc"]), float(metrics["test_balanced_accuracy_mean"])]
    external = [float(metrics["article_holdout_auc"]), float(metrics["article_holdout_balanced_accuracy"])]
    x = np.arange(len(metric_labels))
    axs[0].bar(x - 0.18, internal, width=0.36, color="#2B6CB0", alpha=0.92, label="Donor split")
    axs[0].bar(x + 0.18, external, width=0.36, color="#C05621", alpha=0.92, label="Source holdout")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(metric_labels)
    axs[0].set_ylim(0, 1.05)
    axs[0].set_title("Internal vs External Benchmark", fontweight="bold", pad=10)
    axs[0].legend(frameon=False)

    axs[1].plot(
        window_summary["center_pseudotime"],
        window_summary["selected_threshold"],
        color="#8F4C7A",
        linewidth=2.6,
        marker="o",
    )
    axs[1].axhline(float(metrics["selected_threshold"]), color="#1F2933", linestyle="--", linewidth=1.8)
    axs[1].set_xlabel("Window center pseudotime")
    axs[1].set_ylabel("Selected threshold")
    axs[1].set_ylim(0, 1.02)
    axs[1].set_title("Local Thresholds Across Pseudotime Windows", fontweight="bold", pad=10)

    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_local_clock_window_summary(local_clock_results: dict[str, pd.DataFrame | dict]) -> Path:
    fig_path = LOCAL_MOUSE_CLOCK_SUPP_FIG_DIR / "05_local_clock_window_summary.pdf"
    window_summary = local_clock_results["window_summary"].copy().sort_values("center_pseudotime").reset_index(drop=True)
    window_summary["window_label"] = [f"W{i}" for i in window_summary["window_id"]]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap("magma")
    for idx, row in window_summary.iterrows():
        y = idx
        color = cmap(norm(float(row["selected_threshold"])))
        ax.hlines(y, float(row["support_min"]), float(row["support_max"]), color=color, linewidth=5.2, alpha=0.85)
        ax.scatter(
            float(row["center_pseudotime"]),
            y,
            s=40 + float(row["n_nearest_assigned"]) / 70.0,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        ax.text(float(row["support_max"]) + 0.18, y, f"n={int(row['n_nearest_assigned'])}", va="center", fontsize=10)

    ax.set_yticks(np.arange(window_summary.shape[0]))
    ax.set_yticklabels(window_summary["window_label"])
    ax.set_xlabel("Reference pseudotime")
    ax.set_ylabel("Local window")
    ax.set_title("Local Window Definition Across Pseudotime", fontweight="bold", pad=12)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Selected threshold")
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_local_clock_gene_dynamics(local_clock_results: dict[str, pd.DataFrame | dict], top_n: int = 25) -> Path:
    fig_path = LOCAL_MOUSE_CLOCK_SUPP_FIG_DIR / "06_local_clock_gene_dynamics_heatmap.pdf"
    local_weights = local_clock_results["local_gene_weights"].copy()
    gene_change = _local_gene_change_table(local_clock_results).head(top_n)
    pivot = (
        local_weights.loc[local_weights["Gene"].isin(gene_change["Gene"])]
        .pivot_table(index="Gene", columns="center_pseudotime", values="Weight", aggfunc="mean")
        .reindex(gene_change["Gene"].astype(str).tolist())
    )

    vmax = float(np.nanmax(np.abs(pivot.to_numpy(dtype=float)))) if not pivot.empty else 1.0
    vmax = max(vmax, 1e-6)
    fig, ax = plt.subplots(figsize=(10.2, 7.8))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.astype(str))
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([f"{float(x):.1f}" for x in pivot.columns], rotation=40, ha="right")
    ax.set_xlabel("Window center pseudotime")
    ax.set_title("Genes With the Largest Coefficient Changes Across Windows", fontweight="bold", pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Coefficient")
    gene_change.head(top_n).to_csv(LOCAL_MOUSE_CLOCK_SUPP_FIG_DIR / "local_clock_top_changing_genes.tsv", sep="\t", index=False)
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_local_clock_top5_changes(local_clock_results: dict[str, pd.DataFrame | dict]) -> Path:
    fig_path = LOCAL_MOUSE_CLOCK_SUPP_FIG_DIR / "07_local_clock_top5_gene_changes.pdf"
    gene_change = _local_gene_change_table(local_clock_results).head(5)
    local_weights = local_clock_results["local_gene_weights"].copy()
    top5 = local_weights.loc[local_weights["Gene"].isin(gene_change["Gene"])].copy()
    centers = np.sort(top5["center_pseudotime"].astype(float).unique())

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    palette = plt.get_cmap("tab10")
    for idx, gene in enumerate(gene_change["Gene"].astype(str)):
        sub = top5.loc[top5["Gene"].astype(str) == gene].sort_values("center_pseudotime")
        ax.plot(
            sub["center_pseudotime"],
            sub["Weight"],
            marker="o",
            linewidth=2.4,
            color=palette(idx),
            label=gene,
        )
    ax.axhline(0, color="#1F2933", linestyle="--", linewidth=1.3)
    ax.set_xticks(centers)
    ax.set_xlabel("Window center pseudotime")
    ax.set_ylabel("Coefficient")
    ax.set_title("Top 5 Genes With the Largest Window-to-Window Changes", fontweight="bold", pad=12)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_local_clock_pseudotime_behavior(local_clock_results: dict[str, pd.DataFrame | dict], n_bins: int = 24) -> Path:
    fig_path = LOCAL_MOUSE_CLOCK_SUPP_FIG_DIR / "08_local_clock_pseudotime_behavior.pdf"
    metacells = local_clock_results["metacell_predictions"].copy()
    metacells["pseudotime_metacell"] = pd.to_numeric(metacells["pseudotime_metacell"], errors="coerce")
    metacells["p_old"] = pd.to_numeric(metacells["p_old"], errors="coerce")
    metacells["local_threshold"] = pd.to_numeric(metacells["local_threshold"], errors="coerce")
    metacells = metacells.dropna(subset=["pseudotime_metacell", "p_old", "local_threshold"]).copy()
    metacells["pt_bin"] = pd.qcut(metacells["pseudotime_metacell"], q=int(n_bins), labels=False, duplicates="drop")
    summary = (
        metacells.groupby("pt_bin")
        .agg(
            mean_pseudotime=("pseudotime_metacell", "mean"),
            mean_p_old=("p_old", "mean"),
            mean_local_threshold=("local_threshold", "mean"),
            std_p_old=("p_old", "std"),
            n_metacells=("p_old", "size"),
        )
        .reset_index()
        .sort_values("mean_pseudotime")
    )
    summary["std_p_old"] = summary["std_p_old"].fillna(0.0)
    summary["se_p_old"] = summary["std_p_old"] / np.sqrt(summary["n_metacells"].clip(lower=1))
    summary["ci95_low"] = summary["mean_p_old"] - 1.96 * summary["se_p_old"]
    summary["ci95_high"] = summary["mean_p_old"] + 1.96 * summary["se_p_old"]
    summary.to_csv(LOCAL_MOUSE_CLOCK_SUPP_FIG_DIR / "local_clock_pseudotime_behavior.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(summary["mean_pseudotime"], summary["mean_p_old"], color=NEUTRAL_COLOR, linewidth=2.8, label="Mean p_old")
    ax.fill_between(summary["mean_pseudotime"], summary["ci95_low"], summary["ci95_high"], color=NEUTRAL_COLOR, alpha=0.16)
    ax.plot(summary["mean_pseudotime"], summary["mean_local_threshold"], color="#B56576", linewidth=1.9, linestyle="--", label="Mean local threshold")
    ax.set_xlabel("Atlas metacell pseudotime")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title("Local Clock Behavior Across Atlas Pseudotime", fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_local_clock_verification_pseudotime(local_verification_results: dict[str, pd.DataFrame]) -> Path:
    fig_path = LOCAL_MOUSE_CLOCK_SUPP_FIG_DIR / "09_verification_p_old_across_pseudotime.pdf"
    sample_summary = local_verification_results["sample_summary"].copy()
    curve_summary = local_verification_results["curve_summary"].copy()

    sample_summary["timepoint_std"] = sample_summary["timepoint_std"].astype(str)
    curve_summary = curve_summary.merge(
        sample_summary[["sample_id_std", "timepoint_std"]].drop_duplicates(),
        on="sample_id_std",
        how="left",
        validate="many_to_one",
    )
    grouped_curve = (
        curve_summary.groupby(["timepoint_std", "pt_bin"], as_index=False)
        .agg(
            mean_pseudotime=("mean_pseudotime", "mean"),
            mean_p_old=("mean_p_old", "mean"),
            std_p_old=("mean_p_old", "std"),
            n_samples=("sample_id_std", "nunique"),
        )
    )
    grouped_curve["std_p_old"] = grouped_curve["std_p_old"].fillna(0.0)
    grouped_curve["se_p_old"] = grouped_curve["std_p_old"] / np.sqrt(grouped_curve["n_samples"].clip(lower=1))
    grouped_curve["ci95_low"] = grouped_curve["mean_p_old"] - 1.96 * grouped_curve["se_p_old"]
    grouped_curve["ci95_high"] = grouped_curve["mean_p_old"] + 1.96 * grouped_curve["se_p_old"]
    grouped_curve["timepoint_std"] = pd.Categorical(grouped_curve["timepoint_std"], categories=TIMEPOINT_ORDER, ordered=True)
    grouped_curve = grouped_curve.sort_values(["timepoint_std", "mean_pseudotime"]).reset_index(drop=True)
    grouped_curve.to_csv(LOCAL_MOUSE_CLOCK_SUPP_FIG_DIR / "09_verification_p_old_across_pseudotime.tsv", sep="\t", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(12.0, 5.0))

    for timepoint in TIMEPOINT_ORDER:
        sub = grouped_curve.loc[grouped_curve["timepoint_std"] == timepoint].copy()
        if sub.empty:
            continue
        color = TIMEPOINT_COLOR_MAP.get(timepoint, NEUTRAL_COLOR)
        axs[0].plot(
            sub["mean_pseudotime"],
            sub["mean_p_old"],
            color=color,
            linewidth=2.6,
            label=timepoint,
        )
        axs[0].fill_between(sub["mean_pseudotime"], sub["ci95_low"], sub["ci95_high"], color=color, alpha=0.12)
    axs[0].set_xlabel("Transferred reference pseudotime")
    axs[0].set_ylabel("Mean p_old")
    axs[0].set_ylim(0, 1.02)
    axs[0].set_title("Verification curves across shared pseudotime", fontweight="bold", pad=10)
    axs[0].legend(title="Timepoint", frameon=False, loc="best")

    sample_summary["timepoint_std"] = pd.Categorical(sample_summary["timepoint_std"], categories=TIMEPOINT_ORDER, ordered=True)
    sample_summary = sample_summary.sort_values(["timepoint_std", "sample_id_std"]).reset_index(drop=True)
    time_means = (
        sample_summary.groupby("timepoint_std", as_index=False)
        .agg(mean_p_old=("mean_p_old", "mean"))
        .sort_values("timepoint_std")
    )
    x_map = {tp: i for i, tp in enumerate(TIMEPOINT_ORDER)}
    for timepoint in TIMEPOINT_ORDER:
        sub = sample_summary.loc[sample_summary["timepoint_std"] == timepoint].copy()
        if sub.empty:
            continue
        color = TIMEPOINT_COLOR_MAP.get(timepoint, NEUTRAL_COLOR)
        x = float(x_map[timepoint])
        offsets = np.linspace(-0.12, 0.12, max(1, len(sub)))
        axs[1].scatter(
            np.full(len(sub), x) + offsets[: len(sub)],
            sub["mean_p_old"],
            s=34,
            color=color,
            alpha=0.45,
            linewidths=0,
        )
    axs[1].plot(
        [x_map[str(tp)] for tp in time_means["timepoint_std"].astype(str)],
        time_means["mean_p_old"],
        color=NEUTRAL_COLOR,
        linewidth=2.6,
        marker="o",
        markersize=5.5,
    )
    axs[1].set_xticks(range(len(TIMEPOINT_ORDER)))
    axs[1].set_xticklabels(TIMEPOINT_ORDER)
    axs[1].set_xlabel("Post-injury time")
    axs[1].set_ylabel("Sample mean p_old")
    axs[1].set_ylim(0, 1.02)
    axs[1].set_title("Young verification timecourse summary", fontweight="bold", pad=10)

    fig.suptitle("Local Clock Verification Robustness on Post-injury Data", fontweight="bold", y=0.995)
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def plot_local_clock_train_test_pseudotime(local_clock_results: dict[str, pd.DataFrame | dict], n_bins: int = 24) -> Path:
    fig_path = LOCAL_MOUSE_CLOCK_SUPP_FIG_DIR / "10_train_test_pseudotime_behavior.pdf"
    metacells = local_clock_results["metacell_predictions"].copy()
    metacells["pseudotime_metacell"] = pd.to_numeric(metacells["pseudotime_metacell"], errors="coerce")
    metacells["p_old"] = pd.to_numeric(metacells["p_old"], errors="coerce")
    metacells["local_threshold"] = pd.to_numeric(metacells["local_threshold"], errors="coerce")
    metacells["split_set"] = metacells["split_set"].astype(str)
    metacells["donor_split_id"] = metacells["donor_split_id"].astype(str)
    metacells["Age_group_std"] = metacells["Age_group_std"].astype(str)
    metacells = metacells.dropna(subset=["pseudotime_metacell", "p_old", "local_threshold"]).copy()
    metacells["pt_bin"] = pd.qcut(metacells["pseudotime_metacell"], q=int(n_bins), labels=False, duplicates="drop")
    metacells = metacells.dropna(subset=["pt_bin"]).copy()
    metacells["pt_bin"] = metacells["pt_bin"].astype(int)

    split_summary = (
        metacells.groupby(["split_set", "pt_bin"], as_index=False)
        .agg(
            mean_pseudotime=("pseudotime_metacell", "mean"),
            mean_p_old=("p_old", "mean"),
            mean_local_threshold=("local_threshold", "mean"),
            std_p_old=("p_old", "std"),
            n_metacells=("p_old", "size"),
        )
    )
    split_summary["std_p_old"] = split_summary["std_p_old"].fillna(0.0)
    split_summary["se_p_old"] = split_summary["std_p_old"] / np.sqrt(split_summary["n_metacells"].clip(lower=1))
    split_summary["ci95_low"] = split_summary["mean_p_old"] - 1.96 * split_summary["se_p_old"]
    split_summary["ci95_high"] = split_summary["mean_p_old"] + 1.96 * split_summary["se_p_old"]

    donor_summary = (
        metacells.loc[metacells["split_set"] == "test"]
        .groupby(["donor_split_id", "Age_group_std", "pt_bin"], as_index=False)
        .agg(
            mean_pseudotime=("pseudotime_metacell", "mean"),
            mean_p_old=("p_old", "mean"),
            n_metacells=("p_old", "size"),
        )
    )
    donor_summary = donor_summary.loc[donor_summary["n_metacells"] >= 5].copy()
    split_summary.to_csv(LOCAL_MOUSE_CLOCK_SUPP_FIG_DIR / "10_train_test_pseudotime_behavior.tsv", sep="\t", index=False)
    donor_summary.to_csv(LOCAL_MOUSE_CLOCK_SUPP_FIG_DIR / "10_test_donor_pseudotime_behavior.tsv", sep="\t", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(13.0, 5.2))

    split_colors = {"train": "#2B6CB0", "test": "#C05621"}
    for split in ["train", "test"]:
        sub = split_summary.loc[split_summary["split_set"] == split].copy().sort_values("mean_pseudotime")
        if sub.empty:
            continue
        color = split_colors[split]
        axs[0].plot(sub["mean_pseudotime"], sub["mean_p_old"], color=color, linewidth=2.8)
        axs[0].fill_between(sub["mean_pseudotime"], sub["ci95_low"], sub["ci95_high"], color=color, alpha=0.12)
        axs[0].plot(
            sub["mean_pseudotime"],
            sub["mean_local_threshold"],
            color=color,
            linewidth=1.6,
            linestyle="--",
            alpha=0.9,
        )
    axs[0].set_xlabel("Atlas metacell pseudotime")
    axs[0].set_ylabel("Score")
    axs[0].set_ylim(0, 1.02)
    axs[0].set_title("Train/Test Local Clock Behavior Across Pseudotime", fontweight="bold", pad=10)
    split_handles = [
        Line2D([0], [0], color=split_colors["train"], linewidth=2.8, label="Train"),
        Line2D([0], [0], color=split_colors["test"], linewidth=2.8, label="Test"),
    ]
    metric_handles = [
        Line2D([0], [0], color="#374151", linewidth=2.2, linestyle="-", label="Mean p_old"),
        Line2D([0], [0], color="#374151", linewidth=1.6, linestyle="--", label="Mean local threshold"),
    ]
    legend_split = axs[0].legend(handles=split_handles, title="Split", frameon=False, loc="upper left")
    axs[0].add_artist(legend_split)
    axs[0].legend(handles=metric_handles, title="Metric", frameon=False, loc="lower right")

    test_donors = donor_summary["donor_split_id"].drop_duplicates().tolist()
    for donor_id in test_donors:
        sub = donor_summary.loc[donor_summary["donor_split_id"] == donor_id].copy().sort_values("mean_pseudotime")
        if sub.empty:
            continue
        age = str(sub["Age_group_std"].iloc[0]).lower()
        color = YOUNG_COLOR if age == "young" else OLD_COLOR
        axs[1].plot(
            sub["mean_pseudotime"],
            sub["mean_p_old"],
            color=color,
            linewidth=1.8,
            alpha=0.8,
        )
        end = sub.iloc[-1]
        axs[1].text(
            float(end["mean_pseudotime"]) + 0.05,
            float(end["mean_p_old"]),
            _short_donor_label(donor_id),
            fontsize=8.5,
            color=color,
            va="center",
        )
    legend_handles = [
        Line2D([0], [0], color=YOUNG_COLOR, linewidth=2.0, label="Young test donors"),
        Line2D([0], [0], color=OLD_COLOR, linewidth=2.0, label="Old test donors"),
    ]
    axs[1].set_xlabel("Atlas metacell pseudotime")
    axs[1].set_ylabel("Mean p_old")
    axs[1].set_ylim(0, 1.02)
    axs[1].set_title("Held-out Test Donors Across Pseudotime", fontweight="bold", pad=10)
    axs[1].legend(handles=legend_handles, title="Held-out age group", frameon=False, loc="best")

    fig.suptitle("Local Clock Train/Test Structure Across Pseudotime", fontweight="bold", y=0.995)
    fig.tight_layout()
    return _save_figure(fig, fig_path)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PSEUDOTIME_FIG_DIR.mkdir(parents=True, exist_ok=True)
    ATLAS_TRANSITION_FIG_DIR.mkdir(parents=True, exist_ok=True)
    MOUSE_CLOCK_CORE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    MOUSE_CLOCK_EXPLORATORY_FIG_DIR.mkdir(parents=True, exist_ok=True)
    FIG6_HUMAN_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_MOUSE_CLOCK_SUPP_FIG_DIR.mkdir(parents=True, exist_ok=True)
    POST_MOUSE_CORE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    POST_MOUSE_EXPLORATORY_FIG_DIR.mkdir(parents=True, exist_ok=True)
    POST_HUMAN_CORE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    POST_HUMAN_EXPLORATORY_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for folder in ["markers", "de_go", "regulon_go", "tf_overlap", "target_burden"]:
        _exploratory_type_dir(folder).mkdir(parents=True, exist_ok=True)
        _human_exploratory_type_dir(folder).mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    print("--- Collecting Artifacts ---")
    raw_df = collect_raw_and_filtered_counts()
    atlas_df = collect_training_atlas_composition()
    linkage, linked = collect_linkage_summary()
    transition_panels = collect_musc_transition_panels()
    overlay, backbone = collect_monocle3_overlay()
    post_summary = collect_post_mouse_summary()
    post_state_tables = collect_post_mouse_state_tables(post_summary)
    human_summary = collect_post_human_summary()
    human_tables = collect_post_human_tables()
    mouse_clock = collect_mouse_clock_results()
    mouse_clock_go = collect_mouse_clock_go_results(mouse_clock, top_n=100)
    local_mouse_clock = collect_local_mouse_clock_results()
    local_verification = collect_local_verification_results()
    human_clock = collect_human_clock_results()

    print("--- Generating Individual PDFs ---")
    outputs = [
        plot_raw_vs_retained(raw_df),
        plot_training_atlas_composition(atlas_df),
        plot_linkage_summary(linkage, linked),
        plot_monocle3_backbone(overlay, backbone),
        plot_pseudotime_by_annotation(overlay),
        plot_mouse_clock_gene_weights(mouse_clock),
        plot_mouse_clock_donor_scores(mouse_clock),
        plot_mouse_clock_threshold_transfer(mouse_clock),
        plot_mouse_clock_metric_summary(mouse_clock),
        plot_mouse_clock_go_overview(mouse_clock_go),
        plot_mouse_clock_go_driver_genes(mouse_clock_go),
        plot_mouse_clock_weight_distribution(mouse_clock),
        plot_mouse_clock_sparsity_summary(mouse_clock),
        plot_mouse_clock_holdout_ranked_donors(mouse_clock),
        plot_local_clock_gene_weights(local_mouse_clock),
        plot_local_clock_donor_scores(local_mouse_clock),
        plot_local_clock_threshold_behavior(local_mouse_clock),
        plot_local_clock_metric_summary(local_mouse_clock),
        plot_local_clock_window_summary(local_mouse_clock),
        plot_local_clock_gene_dynamics(local_mouse_clock),
        plot_local_clock_top5_changes(local_mouse_clock),
        plot_local_clock_pseudotime_behavior(local_mouse_clock),
        plot_local_clock_verification_pseudotime(local_verification),
        plot_local_clock_train_test_pseudotime(local_mouse_clock),
        plot_human_reference_atlas(human_clock),
        plot_human_selected_donor_cohort(human_clock),
        plot_human_clock_gene_weights(human_clock),
        plot_human_clock_showcase_donor_scores(human_clock),
        plot_human_clock_threshold_behavior(human_clock),
        plot_human_clock_metric_summary(human_clock),
        plot_human_clock_go_overview(human_clock),
        plot_post_mouse_state_summary(post_summary),
        plot_post_mouse_de_go_overview(post_summary, post_state_tables),
        plot_post_mouse_regulon_go_overview(post_summary, post_state_tables),
        plot_post_mouse_regulon_architecture(post_summary),
        plot_post_mouse_deg_mirror(post_summary, post_state_tables),
        plot_all_state_marker_overview(post_summary, post_state_tables),
        plot_all_state_tf_overlap(post_summary, post_state_tables),
        plot_all_state_target_burden(post_summary),
        plot_post_human_summary(human_summary),
        plot_post_human_go_pair(
            human_tables["go_old_up"],
            human_tables["go_young_up"],
            POST_HUMAN_CORE_FIG_DIR,
            "02_human_de_go_overview.pdf",
            "Human MuSC DE GO comparison",
        ),
        plot_post_human_go_pair(
            human_tables["go_old_regulon_targets"],
            human_tables["go_young_regulon_targets"],
            POST_HUMAN_CORE_FIG_DIR,
            "03_human_regulon_go_overview.pdf",
            "Human MuSC regulon target GO comparison",
        ),
        plot_post_human_marker_overview(human_tables["markers"]),
        plot_post_human_tf_overlap(human_tables["regulon_overlap"]),
        plot_post_human_target_burden(human_summary),
    ]
    outputs.extend(plot_musc_umap_transition(transition_panels))

    for row in post_summary.itertuples(index=False):
        state = str(row.state).replace("/", "_")
        state_display = str(row.state_display)
        tables = post_state_tables[str(row.state)]
        outputs.extend(
            [
                plot_state_marker_overview(state, state_display, tables["markers"]),
                plot_state_go_pair(state, state_display, tables["go_old_up"], tables["go_young_up"], "de_go_comparison", "DE GO comparison"),
                plot_state_go_pair(
                    state,
                    state_display,
                    tables["go_old_regulon_targets"],
                    tables["go_young_regulon_targets"],
                    "regulon_go_comparison",
                    "Regulon target GO comparison",
                ),
                plot_state_tf_overlap(state, state_display, tables["regulon_overlap"]),
                plot_state_regulon_target_burden(state, state_display, pd.Series(row._asdict())),
            ]
        )

    print("Success! Large-format figures saved to:")
    for path in outputs:
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
