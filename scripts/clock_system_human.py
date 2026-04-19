#!/usr/bin/env python3
"""
Human MuSC proof-of-concept age-state classifier pipeline.

This run evaluates all eligible donor-disjoint holdouts, picks one showcase
split with quality gates, and then builds downstream artifacts from the
showcase training branch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import warnings

import joblib
from matplotlib import pyplot as plt
import pandas as pd

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import scanpy as sc

from path_config import ARTIFACT_ROOT, CELL_CYCLE_GENES_PATH, CISTARGET_DIR, HUMAN_TRAINING_DIR, PROJECT_ROOT

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT / "scripts"))

import human_clock_core as human_core  # noqa: E402


ROOT = PROJECT_ROOT
DATA_PATH = HUMAN_TRAINING_DIR / "SKM_human_raw_cells2nuclei_2023-06-22.h5ad"
OUTPUT_DIR = ARTIFACT_ROOT / "human_clock_outputs"

HUMAN_DB_DIR = CISTARGET_DIR / "databases" / "human"
HUMAN_MOTIF_ANNO_PATH = CISTARGET_DIR / "motif2tf" / "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl"
HUMAN_TF_LIST_PATH = CISTARGET_DIR / "tf_lists" / "allTFs_hg38.txt"
HUMAN_GMT_PATH = CISTARGET_DIR / "c5.go.bp.v2024.1.Hs.symbols.gmt"

MIN_DONOR_RAW_MUSC_CELLS = 50
BOOTSTRAP_ITERS = 1000
N_CELLS_PER_BIN = 25
N_TRAINING_HVG = 1500
N_PYSCENIC_HVG = 5000
MIN_GENE_COUNTS = 50
MODEL_C = 0.1
MODEL_L1_RATIO = 0.5
MODEL_MAX_ITER = 5000
GRN_NUM_WORKERS = 8
MIN_DONOR_AUC = 0.75
MIN_DONOR_BALANCED_ACCURACY = 0.70

SEX_CONFOUND_GENE_BLACKLIST = [
    "XIST",
    "TSIX",
    "JPX",
    "XACT",
    "RPS4Y1",
    "DDX3Y",
    "UTY",
    "KDM5D",
    "EIF1AY",
    "ZFY",
    "TMSB4Y",
    "NLGN4Y",
    "USP9Y",
]

MAX_TOP_GENE_ABS_WEIGHT_FRACTION = 0.10
MAX_TOP5_GENE_ABS_WEIGHT_FRACTION = 0.40


def validate_required_paths() -> None:
    db_paths = sorted(HUMAN_DB_DIR.glob("*.feather"))
    human_core.validate_required_paths(
        {
            "human input h5ad": DATA_PATH,
            "cell cycle genes": CELL_CYCLE_GENES_PATH,
            "human motif annotation": HUMAN_MOTIF_ANNO_PATH,
            "human TF list": HUMAN_TF_LIST_PATH,
            "human GO BP GMT": HUMAN_GMT_PATH,
            **({"human cisTarget feather databases": HUMAN_DB_DIR} if not db_paths else {}),
        }
    )


def save_full_split_outputs(
    donor_summary: pd.DataFrame,
    selection_manifest: dict[str, object],
    split_specs: list[dict[str, str]],
    split_results: pd.DataFrame,
    donor_predictions: pd.DataFrame,
    threshold_diagnostics: pd.DataFrame,
    dominance_diagnostics: pd.DataFrame,
    coefficient_stability: pd.DataFrame,
    quality_gate: dict[str, object],
    showcase_selection: dict[str, object],
) -> dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_results_path = OUTPUT_DIR / "male_only_split_results.tsv"
    donor_predictions_path = OUTPUT_DIR / "male_only_donor_predictions.tsv"
    split_manifest_path = OUTPUT_DIR / "male_only_split_manifest.json"
    holdout_summary_path = OUTPUT_DIR / "holdout_donor_summary.tsv"
    dominance_path = OUTPUT_DIR / "feature_dominance_diagnostics.tsv"
    threshold_path = OUTPUT_DIR / "threshold_diagnostics.tsv"
    stability_path = OUTPUT_DIR / "coefficient_stability.tsv"

    split_results.to_csv(split_results_path, sep="\t", index=False)
    donor_predictions.to_csv(donor_predictions_path, sep="\t", index=False)
    threshold_diagnostics.to_csv(threshold_path, sep="\t", index=False)
    donor_summary.to_csv(holdout_summary_path, sep="\t", index=False)
    dominance_diagnostics.to_csv(dominance_path, sep="\t", index=False)
    coefficient_stability.to_csv(stability_path, sep="\t", index=False)

    manifest = {
        **selection_manifest,
        "n_splits": int(len(split_specs)),
        "split_specs": split_specs,
        "quality_gate": quality_gate,
        "showcase_selection": showcase_selection,
    }
    with open(split_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "split_results_path": split_results_path,
        "donor_predictions_path": donor_predictions_path,
        "split_manifest_path": split_manifest_path,
        "holdout_summary_path": holdout_summary_path,
        "dominance_path": dominance_path,
        "threshold_path": threshold_path,
        "stability_path": stability_path,
    }


def save_showcase_supporting_outputs(
    donor_summary: pd.DataFrame,
    selection_manifest: dict[str, object],
    split_manifest: dict[str, object],
    split_result: dict[str, object],
    donor_predictions: pd.DataFrame,
    threshold_diagnostics: pd.DataFrame,
    dominance_diagnostics: pd.DataFrame,
    excluded_confound_genes: list[str],
    showcase_selection: dict[str, object],
) -> dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_results_path = OUTPUT_DIR / "showcase_split_results.tsv"
    donor_predictions_path = OUTPUT_DIR / "showcase_donor_predictions.tsv"
    split_manifest_path = OUTPUT_DIR / "showcase_split_manifest.json"
    holdout_summary_path = OUTPUT_DIR / "showcase_holdout_donor_summary.tsv"
    dominance_path = OUTPUT_DIR / "showcase_feature_dominance_diagnostics.tsv"
    threshold_path = OUTPUT_DIR / "showcase_threshold_diagnostics.tsv"
    confound_path = OUTPUT_DIR / "excluded_confound_genes.txt"

    pd.DataFrame([split_result]).to_csv(split_results_path, sep="\t", index=False)
    donor_predictions.to_csv(donor_predictions_path, sep="\t", index=False)
    threshold_diagnostics.to_csv(threshold_path, sep="\t", index=False)
    donor_summary.to_csv(holdout_summary_path, sep="\t", index=False)
    dominance_diagnostics.to_csv(dominance_path, sep="\t", index=False)
    with open(confound_path, "w", encoding="utf-8") as f:
        for gene in excluded_confound_genes:
            f.write(f"{gene}\n")

    manifest = {
        **selection_manifest,
        "analysis_mode": "showcase_split",
        "showcase_split_id": split_manifest["split_id"],
        "showcase_split_manifest": split_manifest,
        "showcase_selection": showcase_selection,
    }
    with open(split_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "split_results_path": split_results_path,
        "donor_predictions_path": donor_predictions_path,
        "split_manifest_path": split_manifest_path,
        "holdout_summary_path": holdout_summary_path,
        "dominance_path": dominance_path,
        "threshold_path": threshold_path,
        "confound_path": confound_path,
    }


def save_showcase_metrics(
    split_result: dict[str, object],
    support_paths: dict[str, Path],
    selection_manifest: dict[str, object],
    excluded_confound_genes: list[str],
    showcase_selection: dict[str, object],
    final_model_saved: bool,
    final_model_paths: dict[str, str] | None = None,
) -> None:
    metrics = {
        "input_data_path": str(DATA_PATH),
        "output_dir": str(OUTPUT_DIR),
        "analysis_scope": "male_only",
        "analysis_mode": "showcase_split",
        "showcase_split_id": str(split_result["split_id"]),
        "model_type": "elasticnet_logistic_regression",
        "min_donor_raw_musc_cells": int(MIN_DONOR_RAW_MUSC_CELLS),
        "n_retained_donors": int(len(selection_manifest["retained_donors"])),
        "retained_donors": selection_manifest["retained_donors"],
        "excluded_donors": selection_manifest["excluded_donors"],
        "holdout_old_donor": str(split_result["holdout_old_donor"]),
        "holdout_young_donor": str(split_result["holdout_young_donor"]),
        "selected_threshold": float(split_result["selected_threshold"]),
        "selected_threshold_source": str(split_result["selected_threshold_source"]),
        "n_train_donors": int(split_result["n_train_donors"]),
        "n_test_donors": int(split_result["n_test_donors"]),
        "train_test_donor_overlap_count": int(split_result["train_test_donor_overlap_count"]),
        "n_train_metacells": int(split_result["n_train_metacells"]),
        "n_test_metacells": int(split_result["n_test_metacells"]),
        "n_training_genes": int(split_result["n_training_genes"]),
        "train_auc_train_only": float(split_result["train_auc_train_only"]),
        "train_metacell_balanced_accuracy_thr_opt": float(split_result["train_metacell_balanced_accuracy_thr_opt"]),
        "holdout_metacell_auc": float(split_result["holdout_metacell_auc"]),
        "holdout_metacell_balanced_accuracy_thr_0p5": float(split_result["holdout_metacell_balanced_accuracy_thr_0p5"]),
        "holdout_metacell_balanced_accuracy_thr_opt": float(split_result["holdout_metacell_balanced_accuracy_thr_opt"]),
        "holdout_metacell_recall_old_thr_opt": float(split_result["holdout_metacell_recall_old_thr_opt"]),
        "holdout_metacell_recall_young_thr_opt": float(split_result["holdout_metacell_recall_young_thr_opt"]),
        "holdout_donor_auc": float(split_result["holdout_donor_auc"]),
        "holdout_donor_balanced_accuracy_thr_0p5": float(split_result["holdout_donor_balanced_accuracy_thr_0p5"]),
        "holdout_donor_balanced_accuracy_thr_opt": float(split_result["holdout_donor_balanced_accuracy_thr_opt"]),
        "holdout_donor_recall_old_thr_opt": float(split_result["holdout_donor_recall_old_thr_opt"]),
        "holdout_donor_recall_young_thr_opt": float(split_result["holdout_donor_recall_young_thr_opt"]),
        "bootstrap_iters": int(BOOTSTRAP_ITERS),
        "n_training_hvg": int(N_TRAINING_HVG),
        "n_pyscenic_hvg": int(N_PYSCENIC_HVG),
        "min_gene_counts": int(MIN_GENE_COUNTS),
        "model_c": float(MODEL_C),
        "model_l1_ratio": float(MODEL_L1_RATIO),
        "model_max_iter": int(MODEL_MAX_ITER),
        "max_top_gene_abs_weight_fraction": float(MAX_TOP_GENE_ABS_WEIGHT_FRACTION),
        "max_top5_gene_abs_weight_fraction": float(MAX_TOP5_GENE_ABS_WEIGHT_FRACTION),
        "top_gene": str(split_result["top_gene"]),
        "top_gene_abs_weight_fraction": float(split_result["top_gene_abs_weight_fraction"]),
        "top5_abs_weight_fraction": float(split_result["top5_abs_weight_fraction"]),
        "dominance_checks_passed": bool(split_result["dominance_checks_passed"]),
        "n_removed_cell_cycle_genes": int(split_result["n_removed_cell_cycle_genes"]),
        "n_removed_mitochondrial_genes": int(split_result["n_removed_mitochondrial_genes"]),
        "n_removed_sex_confound_genes": int(split_result["n_removed_sex_confound_genes"]),
        "excluded_confound_genes_path": str(support_paths["confound_path"]),
        "showcase_split_results_path": str(support_paths["split_results_path"]),
        "showcase_donor_predictions_path": str(support_paths["donor_predictions_path"]),
        "showcase_split_manifest_path": str(support_paths["split_manifest_path"]),
        "showcase_holdout_donor_summary_path": str(support_paths["holdout_summary_path"]),
        "showcase_feature_dominance_diagnostics_path": str(support_paths["dominance_path"]),
        "showcase_threshold_diagnostics_path": str(support_paths["threshold_path"]),
        "final_model_saved": bool(final_model_saved),
        "showcase_selection_mode": str(showcase_selection["showcase_selection_mode"]),
        "showcase_fallback_used": bool(showcase_selection["showcase_fallback_used"]),
        "showcase_selection_reason": str(showcase_selection["showcase_selection_reason"]),
        "showcase_candidate_pool": str(showcase_selection["showcase_candidate_pool"]),
        "showcase_selected_split_rank_metrics": showcase_selection["showcase_selected_split_rank_metrics"],
        "showcase_selection_thresholds": showcase_selection["showcase_selection_thresholds"],
    }
    if final_model_paths:
        metrics.update(final_model_paths)

    with open(OUTPUT_DIR / "showcase_training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_showcase_model_artifacts(
    model,
    train_hvg: sc.AnnData,
    gene_weights: pd.DataFrame,
    grn_result: dict[str, object],
    enrichment_results: dict[str, pd.DataFrame],
) -> dict[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(os.environ.get("ABCLOCK_ROOT", str(ROOT / "abclock")))))
    import abclock  # noqa: E402

    model_path = OUTPUT_DIR / "final_model.joblib"
    genes_path = OUTPUT_DIR / "model_genes.txt"
    weights_path = OUTPUT_DIR / "gene_weights.tsv"
    regulon_summary_path = OUTPUT_DIR / "grn_regulons_summary.tsv"
    overlap_path = OUTPUT_DIR / "grn_clock_regulon_overlap.tsv"
    auc_path = OUTPUT_DIR / "grn_auc_matrix.tsv"
    adj_path = OUTPUT_DIR / "grn_adjacencies.tsv"

    joblib.dump(model, model_path)
    gene_weights.to_csv(weights_path, sep="\t", index=False)
    with open(genes_path, "w", encoding="utf-8") as f:
        for gene in train_hvg.var_names.astype(str):
            f.write(f"{gene}\n")

    adj = grn_result.get("adjacencies", pd.DataFrame())
    regulons = grn_result.get("regulons", [])
    auc_mtx = grn_result.get("auc_mtx", pd.DataFrame())
    if isinstance(adj, pd.DataFrame) and not adj.empty:
        adj.to_csv(adj_path, sep="\t", index=False)
    if hasattr(auc_mtx, "to_csv") and not auc_mtx.empty:
        auc_mtx.to_csv(auc_path, sep="\t")

    regulon_rows = []
    for regulon in regulons:
        regulon_rows.append(
            {
                "regulon": getattr(regulon, "name", "unknown"),
                "n_genes": len(getattr(regulon, "genes", []) or []),
                "tf": getattr(regulon, "transcription_factor", ""),
            }
        )
    pd.DataFrame(regulon_rows).to_csv(regulon_summary_path, sep="\t", index=False)

    top_young = gene_weights.tail(10)["Gene"].astype(str).tolist()
    top_old = gene_weights.head(10)["Gene"].astype(str).tolist()
    clock_genes = {gene.capitalize() for gene in (top_young + top_old)}
    overlap_rows = abclock.find_clock_gene_regulons(regulons, clock_genes)
    pd.DataFrame(overlap_rows, columns=["regulon", "n_genes", "overlap_clock_genes"]).to_csv(
        overlap_path,
        sep="\t",
        index=False,
    )

    aging_df = enrichment_results.get("aging_enrichment", pd.DataFrame())
    youth_df = enrichment_results.get("youth_enrichment", pd.DataFrame())
    aging_df.to_csv(OUTPUT_DIR / "enrichment_aging.tsv", sep="\t", index=False)
    youth_df.to_csv(OUTPUT_DIR / "enrichment_youth.tsv", sep="\t", index=False)

    if not aging_df.empty or not youth_df.empty:
        fig = abclock.plot_age_enrichment_comparison(
            aging_df,
            youth_df,
            top_n=10,
            figsize=(14, 10),
            aging_color="#ff9f9b",
            youth_color="#8ECFC9",
        )
        _ = fig
        plt.title("Human MuSC Aging Clock: Pathway Enrichment", fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "human_clock_enrichment_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    if not aging_df.empty:
        abclock.plot_enrichment_results(
            aging_df,
            title="Aging-Associated Pathways (Human MuSC)",
            color="#ff9f9b",
        )
        plt.savefig(OUTPUT_DIR / "human_clock_aging_pathways.png", dpi=300, bbox_inches="tight")
        plt.close()

    if not youth_df.empty:
        abclock.plot_enrichment_results(
            youth_df,
            title="Youth-Associated Pathways (Human MuSC)",
            color="#8ECFC9",
        )
        plt.savefig(OUTPUT_DIR / "human_clock_youth_pathways.png", dpi=300, bbox_inches="tight")
        plt.close()

    return {
        "final_model_path": str(model_path),
        "model_genes_path": str(genes_path),
        "gene_weights_path": str(weights_path),
        "grn_regulons_summary_path": str(regulon_summary_path),
        "grn_clock_regulon_overlap_path": str(overlap_path),
        "grn_auc_matrix_path": str(auc_path),
        "grn_adjacencies_path": str(adj_path),
        "enrichment_aging_path": str(OUTPUT_DIR / "enrichment_aging.tsv"),
        "enrichment_youth_path": str(OUTPUT_DIR / "enrichment_youth.tsv"),
    }


# Runtime setup
validate_required_paths()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sc.settings.set_figure_params(dpi=400, frameon=False, facecolor="white")
warnings.filterwarnings("ignore")


# Load input data
print("Loading human skeletal muscle dataset...")
adata = sc.read_h5ad(DATA_PATH)
print(f"  Shape: {adata.shape}")
print(f"  Available obs columns: {list(adata.obs.columns)}")
print("  Treating input var_names as gene symbols; no Ensembl conversion step will be run.")


# Harmonize genes and metadata
print("\n--- Harmonizing Gene Names ---")
original_vars = adata.n_vars
adata.var_names_make_unique(join="first")
adata.var_names = pd.Index([str(name).upper() for name in adata.var_names])
print(f"  {original_vars} genes standardized to uppercase symbols")

print("\n--- Parsing Age Information ---")
print(f"  Available age columns: {[column for column in adata.obs.columns if 'age' in column.lower()]}")
adata = human_core.resolve_human_age_metadata(adata)
print(f"  Binary age groups: {adata.obs['Age_group_binary'].value_counts().to_dict()}")
print(f"  Age numeric range: {adata.obs['age_numeric'].min()} - {adata.obs['age_numeric'].max()}")


# Filter MuSC and select male cohort
print("\n--- MuSC filtering summary ---")
adata_musc_raw = human_core.filter_musc_human(adata)
print(f"  MuSC cells: {adata_musc_raw.n_obs}")

male_raw, donor_summary, selection_manifest = human_core.select_male_human_cohort(
    adata_musc_raw,
    min_donor_raw_musc_cells=MIN_DONOR_RAW_MUSC_CELLS,
)
print("Retained male donors:")
print(
    donor_summary[donor_summary["keep_for_human_clock"]][
        ["sample_id_std", "age_group_binary", "n_cells_raw_musc"]
    ].to_string(index=False)
)
print("Excluded donors:")
print(
    donor_summary[~donor_summary["keep_for_human_clock"]][
        ["sample_id_std", "sex", "age_group_binary", "n_cells_raw_musc", "exclusion_reason"]
    ].to_string(index=False)
)

adata_male = human_core.standardize_obs_human(male_raw)
cell_cycle_genes = human_core.load_cell_cycle_genes(CELL_CYCLE_GENES_PATH)


# Enumerate candidate donor holdouts
split_specs = human_core.enumerate_class_balanced_holdouts(donor_summary)
print(f"\nEnumerated {len(split_specs)} candidate male-only holdout splits")

# Evaluate all donor holdouts and select showcase
print("\n>>> Evaluating all candidate donor holdout splits")
split_rows: list[dict[str, object]] = []
donor_prediction_frames: list[pd.DataFrame] = []
threshold_frames: list[pd.DataFrame] = []
weight_frames: list[pd.DataFrame] = []
dominance_frames: list[pd.DataFrame] = []

for split_spec in split_specs:
    print(f"\n>>> Evaluating split {split_spec['split_id']}")
    train_cells, test_cells, split_manifest = human_core.split_standardized_cells_by_donor(adata_male, split_spec)
    train_bs = human_core.build_human_metacells(
        train_cells,
        N_CELLS_PER_BIN,
        BOOTSTRAP_ITERS,
        prefix=f"{split_spec['split_id']}_train",
    )
    test_bs = human_core.build_human_metacells(
        test_cells,
        N_CELLS_PER_BIN,
        BOOTSTRAP_ITERS,
        prefix=f"{split_spec['split_id']}_test",
    )
    train_hvg, test_hvg, preprocessing_diagnostics = human_core.preprocess_human_split(
        train_bs,
        test_bs,
        n_training_hvg=N_TRAINING_HVG,
        min_gene_counts=MIN_GENE_COUNTS,
        sex_confound_gene_blacklist=SEX_CONFOUND_GENE_BLACKLIST,
        cell_cycle_genes=cell_cycle_genes,
    )
    _, gene_weights_i, dominance_df_i, split_row_i, donor_predictions_i, threshold_table_i = human_core.fit_human_split_model(
        train_hvg,
        test_hvg,
        split_manifest=split_manifest,
        model_c=MODEL_C,
        model_l1_ratio=MODEL_L1_RATIO,
        model_max_iter=MODEL_MAX_ITER,
        max_top_gene_abs_weight_fraction=MAX_TOP_GENE_ABS_WEIGHT_FRACTION,
        max_top5_gene_abs_weight_fraction=MAX_TOP5_GENE_ABS_WEIGHT_FRACTION,
    )
    split_row_i.update(
        {
            "n_removed_cell_cycle_genes": preprocessing_diagnostics["n_removed_cell_cycle_genes"],
            "n_removed_mitochondrial_genes": preprocessing_diagnostics["n_removed_mitochondrial_genes"],
            "n_removed_sex_confound_genes": preprocessing_diagnostics["n_removed_sex_confound_genes"],
        }
    )
    split_rows.append(split_row_i)
    donor_prediction_frames.append(donor_predictions_i)
    threshold_frames.append(threshold_table_i)
    weight_frames.append(gene_weights_i.assign(split_id=split_spec["split_id"]))
    dominance_frames.append(dominance_df_i)

split_results = pd.DataFrame(split_rows)
donor_predictions_all = pd.concat(donor_prediction_frames, ignore_index=True)
threshold_diagnostics_all = pd.concat(threshold_frames, ignore_index=True)
dominance_diagnostics_all = pd.concat(dominance_frames, ignore_index=True)
coefficient_stability = human_core.summarize_coefficient_stability(weight_frames)
quality_gate = human_core.evaluate_quality_gates(
    split_results,
    min_mean_donor_auc=MIN_DONOR_AUC,
    min_mean_donor_balanced_accuracy=MIN_DONOR_BALANCED_ACCURACY,
)
showcase_split, showcase_selection = human_core.select_showcase_split(
    split_results,
    min_donor_auc=MIN_DONOR_AUC,
    min_donor_balanced_accuracy=MIN_DONOR_BALANCED_ACCURACY,
)
full_support_paths = save_full_split_outputs(
    donor_summary=donor_summary,
    selection_manifest=selection_manifest,
    split_specs=split_specs,
    split_results=split_results,
    donor_predictions=donor_predictions_all,
    threshold_diagnostics=threshold_diagnostics_all,
    dominance_diagnostics=dominance_diagnostics_all,
    coefficient_stability=coefficient_stability,
    quality_gate=quality_gate,
    showcase_selection=showcase_selection,
)

print("\nDonor-holdout quality gate summary:")
print(pd.DataFrame([quality_gate]).to_string(index=False))
print(f"\nSelected showcase split: {showcase_split['split_id']}")
print(f"  Selection reason: {showcase_selection['showcase_selection_reason']}")
print(f"  Fallback used: {showcase_selection['showcase_fallback_used']}")


# Re-run selected showcase split for downstream artifacts
print(f"\n>>> Re-running selected showcase split {showcase_split['split_id']}")
train_cells, test_cells, split_manifest = human_core.split_standardized_cells_by_donor(adata_male, showcase_split)
train_bs = human_core.build_human_metacells(
    train_cells,
    N_CELLS_PER_BIN,
    BOOTSTRAP_ITERS,
    prefix=f"{showcase_split['split_id']}_train",
)
test_bs = human_core.build_human_metacells(
    test_cells,
    N_CELLS_PER_BIN,
    BOOTSTRAP_ITERS,
    prefix=f"{showcase_split['split_id']}_test",
)
train_hvg, test_hvg, preprocessing_diagnostics = human_core.preprocess_human_split(
    train_bs,
    test_bs,
    n_training_hvg=N_TRAINING_HVG,
    min_gene_counts=MIN_GENE_COUNTS,
    sex_confound_gene_blacklist=SEX_CONFOUND_GENE_BLACKLIST,
    cell_cycle_genes=cell_cycle_genes,
)
model, gene_weights, dominance_df, split_row, donor_predictions, threshold_table = human_core.fit_human_split_model(
    train_hvg,
    test_hvg,
    split_manifest=split_manifest,
    model_c=MODEL_C,
    model_l1_ratio=MODEL_L1_RATIO,
    model_max_iter=MODEL_MAX_ITER,
    max_top_gene_abs_weight_fraction=MAX_TOP_GENE_ABS_WEIGHT_FRACTION,
    max_top5_gene_abs_weight_fraction=MAX_TOP5_GENE_ABS_WEIGHT_FRACTION,
)
split_row.update(
    {
        "n_removed_cell_cycle_genes": preprocessing_diagnostics["n_removed_cell_cycle_genes"],
        "n_removed_mitochondrial_genes": preprocessing_diagnostics["n_removed_mitochondrial_genes"],
        "n_removed_sex_confound_genes": preprocessing_diagnostics["n_removed_sex_confound_genes"],
    }
)
showcase_support_paths = save_showcase_supporting_outputs(
    donor_summary=donor_summary,
    selection_manifest=selection_manifest,
    split_manifest=split_manifest,
    split_result=split_row,
    donor_predictions=donor_predictions,
    threshold_diagnostics=threshold_table,
    dominance_diagnostics=dominance_df,
    excluded_confound_genes=sorted(preprocessing_diagnostics["excluded"]["sex_confound_genes"]),
    showcase_selection=showcase_selection,
)

print("\nShowcase summary:")
print(
    pd.DataFrame([split_row])[
        [
            "split_id",
            "selected_threshold",
            "holdout_metacell_auc",
            "holdout_metacell_balanced_accuracy_thr_opt",
            "holdout_donor_auc",
            "holdout_donor_balanced_accuracy_thr_opt",
            "top_gene",
        ]
    ].to_string(index=False)
)


# Downstream analysis from showcase training branch
final_model_saved = False
final_model_paths: dict[str, str] | None = None

if bool(split_row["dominance_checks_passed"]):
    print("\n>>> Dominance checks passed; running downstream analysis on showcase training branch")
    _, train_pyscenic, _ = human_core.preprocess_human_full_cohort(
        train_bs,
        n_training_hvg=N_TRAINING_HVG,
        n_pyscenic_hvg=N_PYSCENIC_HVG,
        min_gene_counts=MIN_GENE_COUNTS,
        sex_confound_gene_blacklist=SEX_CONFOUND_GENE_BLACKLIST,
        cell_cycle_genes=cell_cycle_genes,
    )
    showcase_grn = human_core.run_human_grn(
        train_pyscenic,
        human_db_dir=HUMAN_DB_DIR,
        human_tf_list_path=HUMAN_TF_LIST_PATH,
        human_motif_anno_path=HUMAN_MOTIF_ANNO_PATH,
        num_workers=GRN_NUM_WORKERS,
    )
    showcase_enrichment = human_core.run_human_enrichment(gene_weights, HUMAN_GMT_PATH)
    final_model_paths = save_showcase_model_artifacts(model, train_hvg, gene_weights, showcase_grn, showcase_enrichment)
    final_model_saved = True
else:
    print("\nDominance checks failed. Skipping showcase GRN and enrichment.")


# Save showcase metrics
save_showcase_metrics(
    split_result=split_row,
    support_paths=showcase_support_paths,
    selection_manifest=selection_manifest,
    excluded_confound_genes=sorted(preprocessing_diagnostics["excluded"]["sex_confound_genes"]),
    showcase_selection=showcase_selection,
    final_model_saved=final_model_saved,
    final_model_paths=final_model_paths,
)

print("\nSaved outputs:")
for path in full_support_paths.values():
    print(f"  - {path}")
for path in showcase_support_paths.values():
    print(f"  - {path}")
print(f"  - {OUTPUT_DIR / 'showcase_training_metrics.json'}")
if final_model_paths:
    for path in final_model_paths.values():
        print(f"  - {path}")
