#!/usr/bin/env python3
"""
State-stratified young-vs-old MuSC post-analysis (DE, GO, and GRN).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys

import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse import issparse

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import scanpy as sc

from path_config import ARTIFACT_ROOT, CISTARGET_DIR, PROJECT_ROOT

ROOT = PROJECT_ROOT
sys.path.insert(0, str(Path(os.environ.get("ABCLOCK_ROOT", str(ROOT / "abclock")))))

from abclock import enrichment as abclock_enrichment  # noqa: E402
from abclock import grn as abclock_grn  # noqa: E402


ANNOTATED_ATLAS_PATH = ARTIFACT_ROOT / "clock_artifacts" / "musc_annotation" / "musc_atlas_annotated.h5ad"
FALLBACK_TRAINING_ATLAS_PATH = ARTIFACT_ROOT / "clock_artifacts" / "0310_results" / "training_atlas_from_cleaned_annotated.h5ad"
OUTDIR = ARTIFACT_ROOT / "post_mouse_analysis"

CISTARGET_ROOT = CISTARGET_DIR
DB_ROOT = CISTARGET_ROOT / "databases" / "mouse"
MOTIF_ANNO_PATH = CISTARGET_ROOT / "motif2tf" / "motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl"
TF_LIST_PATH = CISTARGET_ROOT / "tf_lists" / "allTFs_mm.txt"
GO_GMT_PATH = CISTARGET_ROOT / "m5.go.bp.v2024.1.Mm.symbols.gmt"

N_PYSCENIC_HVG = 5000
DE_TOP_GENES = 150
GRN_NUM_WORKERS = 8
GRN_N_STEPS = 300
GRN_MAX_EDGES = 50000
MIN_CELLS_PER_AGE_FOR_GRN = 200
MIN_CELLS_PER_AGE_FOR_DE = 50

REQUIRED_OBS_COLS = ("annotation", "Age_group_std")
AGE_ORDER = ("young", "old")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _load_atlas() -> tuple[sc.AnnData, str]:
    atlas_path = ANNOTATED_ATLAS_PATH
    try:
        adata = sc.read_h5ad(atlas_path)
        source_label = "annotated_atlas"
    except Exception as exc:
        if not FALLBACK_TRAINING_ATLAS_PATH.exists():
            raise RuntimeError(
                f"Failed to read annotated atlas at {atlas_path}. "
                "Fallback training atlas is also unavailable."
            ) from exc
        adata = sc.read_h5ad(FALLBACK_TRAINING_ATLAS_PATH)
        source_label = "training_atlas_fallback"

    missing = [col for col in REQUIRED_OBS_COLS if col not in adata.obs.columns]
    if missing:
        raise ValueError(f"Atlas missing required obs columns: {missing}")
    if "counts" not in adata.layers:
        raise ValueError("Atlas must contain raw counts in .layers['counts'].")

    adata.obs["annotation"] = adata.obs["annotation"].astype(str)
    adata.obs["Age_group_std"] = adata.obs["Age_group_std"].astype(str)
    return adata, source_label


def _make_grn_ready_adata(adata: sc.AnnData) -> sc.AnnData:
    work = adata.copy()
    work.X = work.layers["counts"].copy()
    sc.pp.filter_genes(work, min_counts=50)
    mt_genes = [gene for gene in work.var_names if str(gene).lower().startswith("mt-")]
    if mt_genes:
        work = work[:, ~work.var_names.isin(mt_genes)].copy()
    _normalize_log1p_inplace(work, target_sum=1e4)
    n_hvg = min(int(N_PYSCENIC_HVG), int(work.n_vars))
    if n_hvg > 0 and work.n_vars > n_hvg:
        sc.pp.highly_variable_genes(work, n_top_genes=n_hvg, subset=True)
    work.var_names = work.var_names.str.capitalize()
    return work


def _make_de_ready_adata(adata: sc.AnnData) -> sc.AnnData:
    work = adata.copy()
    work.X = work.layers["counts"].copy()
    _normalize_log1p_inplace(work, target_sum=1e4)
    return work


def _normalize_log1p_inplace(adata: sc.AnnData, target_sum: float = 1e4) -> None:
    X = adata.X
    if issparse(X):
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
    X = X * scale
    adata.X = np.log1p(X)


def _regulons_to_summary(regulons: list) -> pd.DataFrame:
    rows = []
    for regulon in regulons:
        tf = getattr(regulon, "transcription_factor", "") or str(getattr(regulon, "name", "unknown")).split("_", 1)[0]
        genes = sorted(set(map(str, getattr(regulon, "genes", []) or [])))
        rows.append(
            {
                "regulon": str(getattr(regulon, "name", "unknown")),
                "tf": str(tf),
                "n_genes": int(len(genes)),
                "genes": ";".join(genes),
            }
        )
    return pd.DataFrame(rows).sort_values(["tf", "regulon"]).reset_index(drop=True)


def _run_one_grn(age_adata: sc.AnnData, age_label: str, outdir: Path, tf_list: list[str], db_paths: list[Path]) -> dict[str, object]:
    grn_input = _make_grn_ready_adata(age_adata)
    abclock_grn_result = abclock_grn.run_regdiffusion_pyscenic_pipeline(
        adata=grn_input,
        tf_list=tf_list,
        db_paths=db_paths,
        motif_anno_path=str(MOTIF_ANNO_PATH),
        n_steps=GRN_N_STEPS,
        num_workers=GRN_NUM_WORKERS,
        max_edges=GRN_MAX_EDGES,
    )

    adj = abclock_grn_result.get("adjacencies", pd.DataFrame())
    regulons = abclock_grn_result.get("regulons", [])
    auc_mtx = abclock_grn_result.get("auc_mtx", pd.DataFrame())
    regulon_df = _regulons_to_summary(regulons)

    if isinstance(adj, pd.DataFrame) and not adj.empty:
        adj.to_csv(outdir / "grn_adjacencies.tsv", sep="\t", index=False)
    if hasattr(auc_mtx, "to_csv") and not auc_mtx.empty:
        auc_mtx.to_csv(outdir / "grn_auc_matrix.tsv", sep="\t")
    regulon_df.to_csv(outdir / "grn_regulons_summary.tsv", sep="\t", index=False)

    metadata = {
        "age_label": str(age_label),
        "n_cells": int(age_adata.n_obs),
        "n_genes_input": int(grn_input.n_vars),
        "n_regulons": int(regulon_df.shape[0]),
        "n_adjacencies": int(adj.shape[0]) if isinstance(adj, pd.DataFrame) else 0,
    }
    with open(outdir / "grn_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "adjacencies": adj,
        "regulons": regulons,
        "regulon_df": regulon_df,
        "auc_mtx": auc_mtx,
        "summary": metadata,
    }


def _extract_top_markers(marker_df: pd.DataFrame, top_n: int) -> list[str]:
    if marker_df.empty:
        return []
    work = marker_df.copy()
    if "pvals_adj" in work.columns:
        work = work.loc[work["pvals_adj"].fillna(1.0) <= 0.05]
    if "logfoldchanges" in work.columns:
        work = work.loc[work["logfoldchanges"].fillna(0.0) > 0]
    genes = work["names"].astype(str).head(top_n).tolist()
    return list(dict.fromkeys(genes))


def _run_state_de_and_go(state_name: str, adata_state: sc.AnnData, outdir: Path) -> dict[str, object]:
    de_adata = _make_de_ready_adata(adata_state)
    if "Age_group_std" not in de_adata.obs.columns:
        raise ValueError("State AnnData missing Age_group_std for DE analysis.")

    age_counts = de_adata.obs["Age_group_std"].astype(str).value_counts().to_dict()
    if age_counts.get("young", 0) < MIN_CELLS_PER_AGE_FOR_DE or age_counts.get("old", 0) < MIN_CELLS_PER_AGE_FOR_DE:
        return {
            "state": state_name,
            "age_counts": age_counts,
            "skipped": True,
            "skip_reason": "insufficient_cells_for_de",
        }

    sc.tl.rank_genes_groups(
        de_adata,
        groupby="Age_group_std",
        groups=["old"],
        reference="young",
        method="wilcoxon",
    )
    old_vs_young = sc.get.rank_genes_groups_df(de_adata, group="old").copy()
    old_vs_young["direction"] = "old_up"

    sc.tl.rank_genes_groups(
        de_adata,
        groupby="Age_group_std",
        groups=["young"],
        reference="old",
        method="wilcoxon",
    )
    young_vs_old = sc.get.rank_genes_groups_df(de_adata, group="young").copy()
    young_vs_old["direction"] = "young_up"

    marker_df = pd.concat([old_vs_young, young_vs_old], ignore_index=True)
    marker_df.to_csv(outdir / "young_vs_old_markers.tsv", sep="\t", index=False)

    old_genes = _extract_top_markers(old_vs_young, DE_TOP_GENES)
    young_genes = _extract_top_markers(young_vs_old, DE_TOP_GENES)

    old_go = abclock_enrichment.run_enrichment_analysis(old_genes, gmt_path=str(GO_GMT_PATH), organism="Mouse")
    young_go = abclock_enrichment.run_enrichment_analysis(young_genes, gmt_path=str(GO_GMT_PATH), organism="Mouse")
    old_go.to_csv(outdir / "go_old_up.tsv", sep="\t", index=False)
    young_go.to_csv(outdir / "go_young_up.tsv", sep="\t", index=False)

    if not old_go.empty or not young_go.empty:
        fig = abclock_enrichment.plot_age_enrichment_comparison(
            old_go,
            young_go,
            top_n=10,
            figsize=(12, 8),
            aging_color="#C65D3B",
            youth_color="#4F7A5A",
        )
        if fig is not None:
            fig.savefig(outdir / "go_age_comparison.png", dpi=300, bbox_inches="tight")

    return {
        "state": state_name,
        "age_counts": age_counts,
        "skipped": False,
        "old_up_genes": old_genes,
        "young_up_genes": young_genes,
        "n_old_go_terms": int(old_go.shape[0]),
        "n_young_go_terms": int(young_go.shape[0]),
    }


def _compare_regulons_between_ages(
    state_name: str,
    young_regulon_df: pd.DataFrame,
    old_regulon_df: pd.DataFrame,
    outdir: Path,
) -> dict[str, object]:
    def to_map(reg_df: pd.DataFrame) -> dict[str, set[str]]:
        mapping: dict[str, set[str]] = {}
        for _, row in reg_df.iterrows():
            tf = str(row["tf"])
            genes = set(filter(None, str(row["genes"]).split(";")))
            mapping[tf] = genes
        return mapping

    young_map = to_map(young_regulon_df)
    old_map = to_map(old_regulon_df)
    all_tfs = sorted(set(young_map) | set(old_map))

    rows = []
    young_unique_target_union: set[str] = set()
    old_unique_target_union: set[str] = set()
    for tf in all_tfs:
        young_targets = young_map.get(tf, set())
        old_targets = old_map.get(tf, set())
        shared = young_targets & old_targets
        young_unique = young_targets - old_targets
        old_unique = old_targets - young_targets
        denom = len(young_targets | old_targets)
        rows.append(
            {
                "state": state_name,
                "tf": tf,
                "young_targets": len(young_targets),
                "old_targets": len(old_targets),
                "shared_targets": len(shared),
                "young_unique_targets": len(young_unique),
                "old_unique_targets": len(old_unique),
                "jaccard": float(len(shared) / denom) if denom else 0.0,
                "young_only_regulon": tf in young_map and tf not in old_map,
                "old_only_regulon": tf in old_map and tf not in young_map,
                "young_unique_gene_list": ";".join(sorted(young_unique)),
                "old_unique_gene_list": ";".join(sorted(old_unique)),
            }
        )
        young_unique_target_union.update(young_unique)
        old_unique_target_union.update(old_unique)

    comparison_df = pd.DataFrame(rows).sort_values(
        ["old_only_regulon", "young_only_regulon", "old_unique_targets", "young_unique_targets", "tf"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    comparison_df.to_csv(outdir / "regulon_overlap.tsv", sep="\t", index=False)

    young_targets_df = pd.DataFrame({"gene": sorted(young_unique_target_union)})
    old_targets_df = pd.DataFrame({"gene": sorted(old_unique_target_union)})
    young_targets_df.to_csv(outdir / "young_biased_regulon_targets.tsv", sep="\t", index=False)
    old_targets_df.to_csv(outdir / "old_biased_regulon_targets.tsv", sep="\t", index=False)

    young_go = abclock_enrichment.run_enrichment_analysis(
        young_targets_df["gene"].astype(str).tolist(),
        gmt_path=str(GO_GMT_PATH),
        organism="Mouse",
    ) if not young_targets_df.empty else pd.DataFrame()
    old_go = abclock_enrichment.run_enrichment_analysis(
        old_targets_df["gene"].astype(str).tolist(),
        gmt_path=str(GO_GMT_PATH),
        organism="Mouse",
    ) if not old_targets_df.empty else pd.DataFrame()

    young_go.to_csv(outdir / "go_young_regulon_targets.tsv", sep="\t", index=False)
    old_go.to_csv(outdir / "go_old_regulon_targets.tsv", sep="\t", index=False)

    if not old_go.empty or not young_go.empty:
        fig = abclock_enrichment.plot_age_enrichment_comparison(
            old_go,
            young_go,
            top_n=10,
            figsize=(12, 8),
            aging_color="#C65D3B",
            youth_color="#4F7A5A",
        )
        if fig is not None:
            fig.savefig(outdir / "go_regulon_target_comparison.png", dpi=300, bbox_inches="tight")

    return {
        "state": state_name,
        "n_shared_tfs": int(sum((tf in young_map) and (tf in old_map) for tf in all_tfs)),
        "n_young_only_tfs": int(sum((tf in young_map) and (tf not in old_map) for tf in all_tfs)),
        "n_old_only_tfs": int(sum((tf in old_map) and (tf not in young_map) for tf in all_tfs)),
        "n_young_unique_regulon_targets": int(len(young_unique_target_union)),
        "n_old_unique_regulon_targets": int(len(old_unique_target_union)),
        "n_young_regulon_go_terms": int(young_go.shape[0]),
        "n_old_regulon_go_terms": int(old_go.shape[0]),
    }


def main() -> None:
    _require_file(MOTIF_ANNO_PATH, "mouse motif annotation")
    _require_file(TF_LIST_PATH, "mouse TF list")
    _require_file(GO_GMT_PATH, "mouse GO GMT")
    db_paths = sorted(DB_ROOT.glob("*.feather"))
    if not db_paths:
        raise FileNotFoundError(f"No mouse cisTarget databases found in {DB_ROOT}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    adata, atlas_source = _load_atlas()
    tf_list = pd.read_csv(TF_LIST_PATH, header=None)[0].astype(str).tolist()

    state_age_counts = (
        adata.obs.groupby(["annotation", "Age_group_std"]).size().reset_index(name="n_cells")
        .sort_values(["annotation", "Age_group_std"])
        .reset_index(drop=True)
    )
    state_age_counts.to_csv(OUTDIR / "state_age_counts.tsv", sep="\t", index=False)

    summary_rows = []
    state_labels = sorted(map(str, adata.obs["annotation"].astype(str).unique()))
    for state_name in state_labels:
        state_key = _safe_name(state_name)
        state_dir = OUTDIR / state_key
        young_dir = state_dir / "young"
        old_dir = state_dir / "old"
        compare_dir = state_dir / "age_comparison"
        young_dir.mkdir(parents=True, exist_ok=True)
        old_dir.mkdir(parents=True, exist_ok=True)
        compare_dir.mkdir(parents=True, exist_ok=True)

        adata_state = adata[adata.obs["annotation"].astype(str) == state_name].copy()
        age_counts = adata_state.obs["Age_group_std"].astype(str).value_counts().to_dict()

        state_summary: dict[str, object] = {
            "state": state_name,
            "state_dir": str(state_dir),
            "atlas_source": atlas_source,
            "young_cells": int(age_counts.get("young", 0)),
            "old_cells": int(age_counts.get("old", 0)),
            "grn_skipped": False,
            "de_skipped": False,
        }

        de_result = _run_state_de_and_go(state_name, adata_state, compare_dir)
        state_summary["de_skipped"] = bool(de_result.get("skipped", False))
        state_summary["de_skip_reason"] = de_result.get("skip_reason")
        state_summary["n_old_go_terms"] = de_result.get("n_old_go_terms")
        state_summary["n_young_go_terms"] = de_result.get("n_young_go_terms")

        if age_counts.get("young", 0) < MIN_CELLS_PER_AGE_FOR_GRN or age_counts.get("old", 0) < MIN_CELLS_PER_AGE_FOR_GRN:
            state_summary["grn_skipped"] = True
            state_summary["grn_skip_reason"] = "insufficient_cells_for_grn"
            summary_rows.append(state_summary)
            with open(state_dir / "state_summary.json", "w", encoding="utf-8") as f:
                json.dump(state_summary, f, indent=2)
            continue

        young_state = adata_state[adata_state.obs["Age_group_std"].astype(str) == "young"].copy()
        old_state = adata_state[adata_state.obs["Age_group_std"].astype(str) == "old"].copy()

        young_grn = _run_one_grn(young_state, "young", young_dir, tf_list, db_paths)
        old_grn = _run_one_grn(old_state, "old", old_dir, tf_list, db_paths)
        regulon_compare = _compare_regulons_between_ages(
            state_name,
            young_grn["regulon_df"],
            old_grn["regulon_df"],
            compare_dir,
        )

        state_summary.update(
            {
                "young_regulons": young_grn["summary"]["n_regulons"],
                "old_regulons": old_grn["summary"]["n_regulons"],
                "young_adjacencies": young_grn["summary"]["n_adjacencies"],
                "old_adjacencies": old_grn["summary"]["n_adjacencies"],
                "shared_tfs": regulon_compare["n_shared_tfs"],
                "young_only_tfs": regulon_compare["n_young_only_tfs"],
                "old_only_tfs": regulon_compare["n_old_only_tfs"],
                "young_unique_regulon_targets": regulon_compare["n_young_unique_regulon_targets"],
                "old_unique_regulon_targets": regulon_compare["n_old_unique_regulon_targets"],
                "young_regulon_go_terms": regulon_compare["n_young_regulon_go_terms"],
                "old_regulon_go_terms": regulon_compare["n_old_regulon_go_terms"],
            }
        )

        with open(state_dir / "state_summary.json", "w", encoding="utf-8") as f:
            json.dump(state_summary, f, indent=2)
        summary_rows.append(state_summary)

    summary_df = pd.DataFrame(summary_rows).sort_values("state").reset_index(drop=True)
    summary_df.to_csv(OUTDIR / "combined_state_summary.tsv", sep="\t", index=False)
    with open(OUTDIR / "combined_state_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2)

    print(f"State-stratified post-analysis complete: {OUTDIR}")


if __name__ == "__main__":
    main()
