from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, recall_score, roc_auc_score, roc_curve

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import scanpy as sc

from path_config import PROJECT_ROOT


ROOT = PROJECT_ROOT
ABCLOCK_ROOT = Path(os.environ.get("ABCLOCK_ROOT", str(ROOT / "abclock")))
sys.path.insert(0, str(ABCLOCK_ROOT))
import metacells as abclock_metacells  # noqa: E402


def _first_present_column(obs: pd.DataFrame, candidates: list[str], default: str = "unknown") -> pd.Series:
    for col in candidates:
        if col in obs.columns:
            return obs[col].astype(str)
    return pd.Series(default, index=obs.index, dtype=object)


def normalize_age_class_label(value: object) -> str:
    label = str(value).strip().lower()
    if label in {"young", "old"}:
        return label
    return "unknown"


def map_age_range_to_group(age_str: object) -> str:
    try:
        start_age = int(str(age_str).split("-")[0])
        return "old" if start_age >= 45 else "young"
    except Exception:
        return "unknown"


def map_age_to_numeric(age_str: object) -> float:
    try:
        start, end = map(int, str(age_str).split("-"))
        return (start + end) / 2.0
    except Exception:
        return float("nan")


def validate_required_paths(required_paths: dict[str, Path]) -> None:
    missing = [f"{label}: {path}" for label, path in required_paths.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n- " + "\n- ".join(missing))


def resolve_human_age_metadata(adata: sc.AnnData) -> sc.AnnData:
    obs = adata.obs.copy()

    age_bin = obs["Age_bin"].astype(str) if "Age_bin" in obs.columns else pd.Series(index=obs.index, dtype=object)
    age_group = obs["Age_group"].astype(str) if "Age_group" in obs.columns else pd.Series(index=obs.index, dtype=object)

    age_bin_labels = {normalize_age_class_label(value) for value in age_bin.dropna().tolist()}
    age_group_labels = {normalize_age_class_label(value) for value in age_group.dropna().tolist()}

    if age_bin_labels.issubset({"young", "old", "unknown"}):
        binary_age = age_bin.map(normalize_age_class_label)
        age_range_source = age_group
        binary_source_col = "Age_bin"
        range_source_col = "Age_group"
    elif age_group_labels.issubset({"young", "old", "unknown"}):
        binary_age = age_group.map(normalize_age_class_label)
        age_range_source = age_bin
        binary_source_col = "Age_group"
        range_source_col = "Age_bin"
    else:
        fallback_source = age_group if "Age_group" in obs.columns else age_bin
        binary_age = fallback_source.map(map_age_range_to_group)
        age_range_source = fallback_source
        binary_source_col = range_source_col = "Age_group" if "Age_group" in obs.columns else "Age_bin"

    obs["Age_group_binary"] = binary_age.astype(str)
    obs["age_numeric"] = age_range_source.map(map_age_to_numeric)
    obs["Age_range_source"] = age_range_source.astype(str)
    adata.obs = obs
    return adata


def filter_musc_human(adata: sc.AnnData) -> sc.AnnData:
    mask = adata.obs["annotation_level2"] == "MuSC"
    return adata[mask].copy()


def standardize_obs_human(adata: sc.AnnData) -> sc.AnnData:
    obs = adata.obs.copy()
    obs["sample_id_std"] = _first_present_column(obs, ["DonorID", "SampleID"])
    obs["Sex_std"] = _first_present_column(obs, ["Sex"])
    obs["celltype_std"] = _first_present_column(obs, ["annotation_level2", "annotation_level1"], default="MuSC")

    obs["Age_group_std"] = obs["Age_group_binary"].astype(str)
    obs["age_range_std"] = obs["Age_range_source"].astype(str)
    if "BMI" in obs.columns:
        obs["BMI_std"] = obs["BMI"]

    keep_cols = ["sample_id_std", "Sex_std", "celltype_std", "Age_group_std", "age_range_std", "BMI_std"]
    adata.obs = obs[[c for c in keep_cols if c in obs.columns]].copy()
    return adata


def build_raw_musc_donor_summary(adata_musc_raw: sc.AnnData) -> pd.DataFrame:
    obs = adata_musc_raw.obs.copy()
    if not any(col in obs.columns for col in ["DonorID", "SampleID", "sample_id_std"]):
        raise KeyError("MuSC donor summary requires DonorID/SampleID/sample_id_std")
    obs["sample_id_std"] = _first_present_column(obs, ["DonorID", "SampleID", "sample_id_std"])

    obs["Sex"] = _first_present_column(obs, ["Sex", "Sex_std"])

    if "Age_bin" in obs.columns:
        obs["Age_bin"] = obs["Age_bin"].astype(str)
    else:
        obs["Age_bin"] = obs.get("Age_group_std", "unknown")

    if "Age_group" in obs.columns:
        obs["Age_group"] = obs["Age_group"].astype(str)
    else:
        obs["Age_group"] = obs.get("age_range_std", "unknown")

    if "Age_group_binary" in obs.columns:
        obs["Age_group_binary"] = obs["Age_group_binary"].astype(str)
    else:
        obs["Age_group_binary"] = obs.get("Age_group_std", "unknown")

    summary = (
        obs.groupby("sample_id_std")
        .agg(
            n_cells_raw_musc=("sample_id_std", "size"),
            sex=("Sex", lambda values: sorted(set(map(str, values)))[0]),
            age_bin_raw=("Age_bin", lambda values: sorted(set(map(str, values)))[0]),
            age_group_raw=("Age_group", lambda values: sorted(set(map(str, values)))[0]),
            age_group_binary=("Age_group_binary", lambda values: sorted(set(map(str, values)))[0]),
        )
        .reset_index()
    )
    return summary


def select_male_human_cohort(
    adata_musc_raw: sc.AnnData,
    min_donor_raw_musc_cells: int,
) -> tuple[sc.AnnData, pd.DataFrame, dict[str, Any]]:
    donor_summary = build_raw_musc_donor_summary(adata_musc_raw)
    donor_summary["keep_for_human_clock"] = True
    donor_summary["exclusion_reason"] = ""

    donor_summary.loc[donor_summary["sex"] != "M", ["keep_for_human_clock", "exclusion_reason"]] = [False, "not_male"]
    donor_summary.loc[
        ~donor_summary["age_group_binary"].isin(["young", "old"]),
        ["keep_for_human_clock", "exclusion_reason"],
    ] = [False, "invalid_age_group"]
    donor_summary.loc[
        donor_summary["n_cells_raw_musc"] < int(min_donor_raw_musc_cells),
        ["keep_for_human_clock", "exclusion_reason"],
    ] = [False, "too_few_raw_musc_cells"]

    keep_donors = donor_summary.loc[donor_summary["keep_for_human_clock"], "sample_id_std"].astype(str).tolist()
    adata_male = adata_musc_raw[adata_musc_raw.obs["DonorID"].astype(str).isin(keep_donors)].copy()

    manifest = {
        "analysis_scope": "male_only",
        "min_donor_raw_musc_cells": int(min_donor_raw_musc_cells),
        "retained_donors": sorted(keep_donors),
        "excluded_donors": donor_summary.loc[~donor_summary["keep_for_human_clock"], "sample_id_std"].astype(str).tolist(),
    }
    return adata_male, donor_summary, manifest


def enumerate_class_balanced_holdouts(donor_summary: pd.DataFrame) -> list[dict[str, str]]:
    eligible = donor_summary[donor_summary["keep_for_human_clock"]].copy()
    old_donors = sorted(eligible.loc[eligible["age_group_binary"] == "old", "sample_id_std"].astype(str).tolist())
    young_donors = sorted(eligible.loc[eligible["age_group_binary"] == "young", "sample_id_std"].astype(str).tolist())
    splits = []
    for old_donor in old_donors:
        for young_donor in young_donors:
            splits.append(
                {
                    "split_id": f"old_{old_donor}__young_{young_donor}",
                    "holdout_old_donor": old_donor,
                    "holdout_young_donor": young_donor,
                }
            )
    return splits


def split_standardized_cells_by_donor(
    adata_musc_std: sc.AnnData,
    split_spec: dict[str, str],
) -> tuple[sc.AnnData, sc.AnnData, dict[str, Any]]:
    holdout_donors = {split_spec["holdout_old_donor"], split_spec["holdout_young_donor"]}
    donor_series = adata_musc_std.obs["sample_id_std"].astype(str)
    test_mask = donor_series.isin(holdout_donors)
    train_adata = adata_musc_std[~test_mask].copy()
    test_adata = adata_musc_std[test_mask].copy()

    train_donors = set(train_adata.obs["sample_id_std"].astype(str).unique())
    test_donors = set(test_adata.obs["sample_id_std"].astype(str).unique())
    overlap = sorted(train_donors & test_donors)

    if overlap:
        raise ValueError(f"Train/test donor leakage detected in split {split_spec['split_id']}: {overlap}")

    return train_adata, test_adata, {
        "split_id": split_spec["split_id"],
        "holdout_old_donor": split_spec["holdout_old_donor"],
        "holdout_young_donor": split_spec["holdout_young_donor"],
        "train_donors": sorted(train_donors),
        "test_donors": sorted(test_donors),
        "train_test_donor_overlap_count": int(len(overlap)),
    }


def build_human_metacells(
    adata: sc.AnnData,
    n_cells_per_bin: int,
    n_iter: int,
    prefix: str,
) -> sc.AnnData:
    adata_bs = abclock_metacells.generate_bootstrap_cells(
        adata,
        n_cells_per_bin=n_cells_per_bin,
        n_iter=n_iter,
        donor_col="sample_id_std",
        age_col="Age_group_std",
        state_col="celltype_std",
        extra_meta_cols=["Sex_std", "BMI_std", "Age_group_std", "sample_id_std"],
        balance_classes=True,
        n_iter_minority=n_iter,
    )
    adata_bs.obs.index = [f"{prefix}_metacell_{index}" for index in range(adata_bs.shape[0])]
    return adata_bs


def load_cell_cycle_genes(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().upper() for line in f.readlines()]


def build_exclusion_lists_from_var_names(
    var_names: pd.Index,
    sex_confound_gene_blacklist: list[str],
    cell_cycle_genes: list[str],
) -> dict[str, list[str]]:
    gene_set = {str(g).upper() for g in var_names}
    excluded_cell_cycle = sorted({gene for gene in cell_cycle_genes if gene in gene_set})
    excluded_mito = sorted({gene for gene in gene_set if gene.startswith("MT-")})
    excluded_sex = sorted({gene for gene in sex_confound_gene_blacklist if gene in gene_set})
    return {
        "cell_cycle_genes": excluded_cell_cycle,
        "mitochondrial_genes": excluded_mito,
        "sex_confound_genes": excluded_sex,
    }


def apply_gene_exclusions(adata: sc.AnnData, excluded: dict[str, list[str]]) -> sc.AnnData:
    excluded_genes = set(excluded["cell_cycle_genes"]) | set(excluded["mitochondrial_genes"]) | set(
        excluded["sex_confound_genes"]
    )
    keep_mask = ~adata.var_names.astype(str).isin(sorted(excluded_genes))
    return adata[:, keep_mask].copy()


def gene_count_mask(adata: sc.AnnData, min_counts: int) -> np.ndarray:
    counts = adata.X.sum(axis=0)
    if issparse(counts):
        counts = counts.A1
    else:
        counts = np.asarray(counts).ravel()
    return counts >= min_counts


def preprocess_human_split(
    train_bs: sc.AnnData,
    test_bs: sc.AnnData,
    n_training_hvg: int,
    min_gene_counts: int,
    sex_confound_gene_blacklist: list[str],
    cell_cycle_genes: list[str],
) -> tuple[sc.AnnData, sc.AnnData, dict[str, Any]]:
    excluded = build_exclusion_lists_from_var_names(train_bs.var_names, sex_confound_gene_blacklist, cell_cycle_genes)
    train_bs = apply_gene_exclusions(train_bs, excluded)
    test_bs = apply_gene_exclusions(test_bs, excluded)

    keep_mask = gene_count_mask(train_bs, min_gene_counts)
    train_bs = train_bs[:, keep_mask].copy()
    keep_genes = train_bs.var_names.astype(str).tolist()
    test_bs = test_bs[:, keep_genes].copy()

    sc.pp.normalize_total(train_bs, target_sum=1e4)
    sc.pp.log1p(train_bs)
    sc.pp.normalize_total(test_bs, target_sum=1e4)
    sc.pp.log1p(test_bs)

    sc.pp.highly_variable_genes(train_bs, n_top_genes=n_training_hvg, subset=True)
    train_genes = train_bs.var_names.astype(str).tolist()
    test_hvg = test_bs[:, train_genes].copy()

    diagnostics = {
        "excluded": excluded,
        "n_removed_cell_cycle_genes": int(len(excluded["cell_cycle_genes"])),
        "n_removed_mitochondrial_genes": int(len(excluded["mitochondrial_genes"])),
        "n_removed_sex_confound_genes": int(len(excluded["sex_confound_genes"])),
        "training_gene_list": train_genes,
    }
    return train_bs, test_hvg, diagnostics


def preprocess_human_full_cohort(
    full_bs: sc.AnnData,
    n_training_hvg: int,
    n_pyscenic_hvg: int,
    min_gene_counts: int,
    sex_confound_gene_blacklist: list[str],
    cell_cycle_genes: list[str],
) -> tuple[sc.AnnData, sc.AnnData, dict[str, Any]]:
    excluded = build_exclusion_lists_from_var_names(full_bs.var_names, sex_confound_gene_blacklist, cell_cycle_genes)
    full_bs = apply_gene_exclusions(full_bs, excluded)
    keep_mask = gene_count_mask(full_bs, min_gene_counts)
    full_bs = full_bs[:, keep_mask].copy()

    sc.pp.normalize_total(full_bs, target_sum=1e4)
    sc.pp.log1p(full_bs)

    full_pyscenic = full_bs.copy()
    sc.pp.highly_variable_genes(full_bs, n_top_genes=n_training_hvg, subset=True)
    sc.pp.highly_variable_genes(full_pyscenic, n_top_genes=n_pyscenic_hvg, subset=True)

    diagnostics = {
        "excluded": excluded,
        "n_removed_cell_cycle_genes": int(len(excluded["cell_cycle_genes"])),
        "n_removed_mitochondrial_genes": int(len(excluded["mitochondrial_genes"])),
        "n_removed_sex_confound_genes": int(len(excluded["sex_confound_genes"])),
        "training_gene_list": full_bs.var_names.astype(str).tolist(),
    }
    return full_bs, full_pyscenic, diagnostics


def evaluate_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc_old": float(average_precision_score(y_true, y_prob)),
        "recall_old": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_young": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def select_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    thresholds = [0.5]
    _, _, roc_thresholds = roc_curve(y_true, y_prob)
    thresholds.extend([float(t) for t in roc_thresholds if np.isfinite(t)])
    thresholds = sorted(set(thresholds))

    rows = []
    for threshold in thresholds:
        metrics = evaluate_binary_metrics(y_true, y_prob, threshold=threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "recall_old": float(metrics["recall_old"]),
                "recall_young": float(metrics["recall_young"]),
                "distance_to_0p5": abs(float(threshold) - 0.5),
            }
        )

    threshold_table = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "distance_to_0p5", "threshold"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    best_threshold = float(threshold_table.iloc[0]["threshold"])
    return best_threshold, threshold_table


def summarize_donor_predictions(
    obs: pd.DataFrame,
    y_prob: np.ndarray,
    split_manifest: dict[str, Any],
    selected_threshold: float | None = None,
) -> pd.DataFrame:
    donor_frame = obs[["Age", "sample_id_std", "Sex_std"]].copy()
    donor_frame["p_old"] = np.asarray(y_prob, dtype=float)
    donor_predictions = (
        donor_frame.groupby(["sample_id_std", "Age", "Sex_std"], sort=True)
        .agg(
            mean_p_old=("p_old", "mean"),
            median_p_old=("p_old", "median"),
            min_p_old=("p_old", "min"),
            max_p_old=("p_old", "max"),
            n_metacells=("p_old", "size"),
        )
        .reset_index()
        .rename(columns={"sample_id_std": "donor_id", "Age": "age_group", "Sex_std": "sex"})
    )
    donor_predictions["split_id"] = str(split_manifest["split_id"])
    donor_predictions["holdout_old_donor"] = str(split_manifest["holdout_old_donor"])
    donor_predictions["holdout_young_donor"] = str(split_manifest["holdout_young_donor"])
    if selected_threshold is not None:
        donor_predictions["selected_threshold"] = float(selected_threshold)
        donor_predictions["pred_old_thr_0p5"] = (donor_predictions["mean_p_old"] >= 0.5).astype(int)
        donor_predictions["pred_old_thr_opt"] = (
            donor_predictions["mean_p_old"] >= float(selected_threshold)
        ).astype(int)

    first_cols = [
        "donor_id",
        "age_group",
        "sex",
        "split_id",
        "holdout_old_donor",
        "holdout_young_donor",
        "mean_p_old",
        "median_p_old",
        "min_p_old",
        "max_p_old",
        "n_metacells",
    ]
    tail_cols = [
        col
        for col in ["selected_threshold", "pred_old_thr_0p5", "pred_old_thr_opt"]
        if col in donor_predictions.columns
    ]
    return donor_predictions[first_cols + tail_cols]


def compute_dominance_diagnostics(
    gene_weights: pd.DataFrame,
    max_top_gene_abs_weight_fraction: float,
    max_top5_gene_abs_weight_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dominance_df = gene_weights.copy()
    dominance_df["AbsWeight"] = dominance_df["Weight"].abs()
    total_abs = float(dominance_df["AbsWeight"].sum()) or 1.0
    dominance_df["AbsWeightFraction"] = dominance_df["AbsWeight"] / total_abs
    dominance_df = dominance_df.sort_values("AbsWeightFraction", ascending=False).reset_index(drop=True)

    metrics = {
        "top_gene": str(dominance_df.iloc[0]["Gene"]),
        "top_gene_abs_weight_fraction": float(dominance_df.iloc[0]["AbsWeightFraction"]),
        "top5_abs_weight_fraction": float(dominance_df.head(5)["AbsWeightFraction"].sum()),
        "max_top_gene_abs_weight_fraction": float(max_top_gene_abs_weight_fraction),
        "max_top5_gene_abs_weight_fraction": float(max_top5_gene_abs_weight_fraction),
    }
    metrics["dominance_checks_passed"] = bool(
        metrics["top_gene_abs_weight_fraction"] <= max_top_gene_abs_weight_fraction
        and metrics["top5_abs_weight_fraction"] <= max_top5_gene_abs_weight_fraction
    )
    return dominance_df, metrics


def select_showcase_split(
    split_results: pd.DataFrame,
    min_donor_auc: float,
    min_donor_balanced_accuracy: float,
) -> tuple[dict[str, str], dict[str, Any]]:
    if split_results.empty:
        raise ValueError("Cannot select showcase split from an empty split_results table.")

    ranked = split_results.copy()
    ranked["passes_donor_quality_gate"] = (
        (ranked["holdout_donor_auc"] >= float(min_donor_auc))
        & (ranked["holdout_donor_balanced_accuracy_thr_opt"] >= float(min_donor_balanced_accuracy))
    )
    ranked["passes_showcase_gate"] = ranked["dominance_checks_passed"] & ranked["passes_donor_quality_gate"]

    sort_cols = [
        "holdout_donor_balanced_accuracy_thr_opt",
        "holdout_donor_auc",
        "holdout_metacell_auc",
        "top5_abs_weight_fraction",
        "split_id",
    ]
    ascending = [False, False, False, True, True]

    if bool(ranked["passes_showcase_gate"].any()):
        candidates = ranked[ranked["passes_showcase_gate"]].copy()
        fallback_used = False
        selection_reason = "selected_best_split_passing_donor_quality_and_dominance_checks"
        candidate_pool = "donor_valid_and_dominance"
    elif bool(ranked["dominance_checks_passed"].any()):
        candidates = ranked[ranked["dominance_checks_passed"]].copy()
        fallback_used = True
        selection_reason = "no_split_passed_donor_quality_gate_selected_best_dominance_passing_split"
        candidate_pool = "dominance_only"
    else:
        candidates = ranked.copy()
        fallback_used = True
        selection_reason = "no_split_passed_dominance_checks_selected_best_available_split"
        candidate_pool = "best_available"

    selected = candidates.sort_values(sort_cols, ascending=ascending).reset_index(drop=True).iloc[0]
    split_spec = {
        "split_id": str(selected["split_id"]),
        "holdout_old_donor": str(selected["holdout_old_donor"]),
        "holdout_young_donor": str(selected["holdout_young_donor"]),
    }
    selection = {
        "showcase_selection_mode": "best_donor_valid",
        "showcase_selected_split_id": str(selected["split_id"]),
        "showcase_fallback_used": bool(fallback_used),
        "showcase_selection_reason": selection_reason,
        "showcase_candidate_pool": candidate_pool,
        "showcase_selected_split_rank_metrics": {
            "holdout_donor_balanced_accuracy_thr_opt": float(selected["holdout_donor_balanced_accuracy_thr_opt"]),
            "holdout_donor_auc": float(selected["holdout_donor_auc"]),
            "holdout_metacell_auc": float(selected["holdout_metacell_auc"]),
            "top5_abs_weight_fraction": float(selected["top5_abs_weight_fraction"]),
            "dominance_checks_passed": bool(selected["dominance_checks_passed"]),
            "passes_donor_quality_gate": bool(selected["passes_donor_quality_gate"]),
            "selected_threshold_source": str(selected["selected_threshold_source"]),
        },
        "showcase_selection_thresholds": {
            "min_donor_auc": float(min_donor_auc),
            "min_donor_balanced_accuracy": float(min_donor_balanced_accuracy),
        },
    }
    return split_spec, selection


def fit_human_split_model(
    train_hvg: sc.AnnData,
    test_hvg: sc.AnnData,
    split_manifest: dict[str, Any],
    model_c: float,
    model_l1_ratio: float,
    model_max_iter: int,
    max_top_gene_abs_weight_fraction: float,
    max_top5_gene_abs_weight_fraction: float,
) -> tuple[LogisticRegression, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    X_train = train_hvg.X.toarray() if issparse(train_hvg.X) else np.asarray(train_hvg.X)
    X_test = test_hvg.X.toarray() if issparse(test_hvg.X) else np.asarray(test_hvg.X)
    y_train = (train_hvg.obs["Age"].astype(str) == "old").astype(int).to_numpy()
    y_test = (test_hvg.obs["Age"].astype(str) == "old").astype(int).to_numpy()

    clf = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=model_c,
        l1_ratio=model_l1_ratio,
        max_iter=model_max_iter,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    train_prob = clf.predict_proba(X_train)[:, 1]
    test_prob = clf.predict_proba(X_test)[:, 1]
    train_donor_predictions = summarize_donor_predictions(train_hvg.obs, train_prob, split_manifest)
    train_donor_y = (train_donor_predictions["age_group"].astype(str) == "old").astype(int).to_numpy()
    train_donor_prob = train_donor_predictions["mean_p_old"].to_numpy(dtype=float)
    selected_threshold, threshold_table = select_optimal_threshold(train_donor_y, train_donor_prob)
    train_metrics_default = evaluate_binary_metrics(y_train, train_prob, threshold=0.5)
    train_metrics_opt = evaluate_binary_metrics(y_train, train_prob, threshold=selected_threshold)
    test_metrics_default = evaluate_binary_metrics(y_test, test_prob, threshold=0.5)
    test_metrics_opt = evaluate_binary_metrics(y_test, test_prob, threshold=selected_threshold)

    gene_weights = pd.DataFrame(
        {
            "Gene": train_hvg.var_names.astype(str),
            "Weight": clf.coef_[0],
        }
    ).sort_values(by="Weight", ascending=False).reset_index(drop=True)

    dominance_df, dominance_metrics = compute_dominance_diagnostics(
        gene_weights,
        max_top_gene_abs_weight_fraction=max_top_gene_abs_weight_fraction,
        max_top5_gene_abs_weight_fraction=max_top5_gene_abs_weight_fraction,
    )
    dominance_df["split_id"] = split_manifest["split_id"]
    donor_predictions = summarize_donor_predictions(test_hvg.obs, test_prob, split_manifest, selected_threshold)

    donor_y = (donor_predictions["age_group"].astype(str) == "old").astype(int).to_numpy()
    donor_prob = donor_predictions["mean_p_old"].to_numpy(dtype=float)
    donor_metrics_default = evaluate_binary_metrics(donor_y, donor_prob, threshold=0.5)
    donor_metrics_opt = evaluate_binary_metrics(donor_y, donor_prob, threshold=selected_threshold)

    split_row = {
        "split_id": split_manifest["split_id"],
        "holdout_old_donor": split_manifest["holdout_old_donor"],
        "holdout_young_donor": split_manifest["holdout_young_donor"],
        "n_train_donors": int(len(split_manifest["train_donors"])),
        "n_test_donors": int(len(split_manifest["test_donors"])),
        "train_test_donor_overlap_count": int(split_manifest["train_test_donor_overlap_count"]),
        "selected_threshold": float(selected_threshold),
        "selected_threshold_source": "train_donor",
        "n_train_metacells": int(train_hvg.n_obs),
        "n_test_metacells": int(test_hvg.n_obs),
        "n_training_genes": int(train_hvg.n_vars),
        "train_auc_train_only": float(train_metrics_default["auc"]),
        "train_metacell_balanced_accuracy_thr_opt": float(train_metrics_opt["balanced_accuracy"]),
        "holdout_metacell_auc": float(test_metrics_default["auc"]),
        "holdout_metacell_balanced_accuracy_thr_0p5": float(test_metrics_default["balanced_accuracy"]),
        "holdout_metacell_balanced_accuracy_thr_opt": float(test_metrics_opt["balanced_accuracy"]),
        "holdout_metacell_recall_old_thr_opt": float(test_metrics_opt["recall_old"]),
        "holdout_metacell_recall_young_thr_opt": float(test_metrics_opt["recall_young"]),
        "holdout_donor_auc": float(donor_metrics_default["auc"]),
        "holdout_donor_balanced_accuracy_thr_0p5": float(donor_metrics_default["balanced_accuracy"]),
        "holdout_donor_balanced_accuracy_thr_opt": float(donor_metrics_opt["balanced_accuracy"]),
        "holdout_donor_recall_old_thr_opt": float(donor_metrics_opt["recall_old"]),
        "holdout_donor_recall_young_thr_opt": float(donor_metrics_opt["recall_young"]),
        "top_gene": str(dominance_metrics["top_gene"]),
        "top_gene_abs_weight_fraction": float(dominance_metrics["top_gene_abs_weight_fraction"]),
        "top5_abs_weight_fraction": float(dominance_metrics["top5_abs_weight_fraction"]),
        "dominance_checks_passed": bool(dominance_metrics["dominance_checks_passed"]),
    }
    threshold_table["split_id"] = split_manifest["split_id"]
    threshold_table["threshold_source"] = "train_donor"
    return clf, gene_weights, dominance_df, split_row, donor_predictions, threshold_table


def summarize_coefficient_stability(weight_frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_gene: dict[str, list[float]] = {}
    total_splits = len(weight_frames)
    for frame in weight_frames:
        for _, row in frame.iterrows():
            by_gene.setdefault(str(row["Gene"]), []).append(float(row["Weight"]))

    for gene, values in by_gene.items():
        arr = np.asarray(values, dtype=float)
        rows.append(
            {
                "Gene": gene,
                "n_splits_present": int(len(arr)),
                "presence_fraction": float(len(arr) / max(total_splits, 1)),
                "mean_weight": float(arr.mean()),
                "std_weight": float(arr.std(ddof=0)),
                "mean_abs_weight": float(np.abs(arr).mean()),
                "positive_fraction": float(np.mean(arr > 0)),
                "negative_fraction": float(np.mean(arr < 0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_abs_weight", "presence_fraction"], ascending=[False, False]).reset_index(
        drop=True
    )


def evaluate_quality_gates(
    split_results: pd.DataFrame,
    min_mean_donor_auc: float,
    min_mean_donor_balanced_accuracy: float,
) -> dict[str, Any]:
    if split_results.empty:
        return {
            "validation_passed": False,
            "reason": "no_splits_completed",
        }

    mean_donor_auc = float(split_results["holdout_donor_auc"].mean())
    std_donor_auc = float(split_results["holdout_donor_auc"].std(ddof=0))
    mean_donor_balacc = float(split_results["holdout_donor_balanced_accuracy_thr_opt"].mean())
    std_donor_balacc = float(split_results["holdout_donor_balanced_accuracy_thr_opt"].std(ddof=0))
    mean_metacell_auc = float(split_results["holdout_metacell_auc"].mean())
    std_metacell_auc = float(split_results["holdout_metacell_auc"].std(ddof=0))
    mean_metacell_balacc = float(split_results["holdout_metacell_balanced_accuracy_thr_opt"].mean())
    std_metacell_balacc = float(split_results["holdout_metacell_balanced_accuracy_thr_opt"].std(ddof=0))
    mean_selected_threshold = float(split_results["selected_threshold"].mean())
    std_selected_threshold = float(split_results["selected_threshold"].std(ddof=0))
    dominance_all_passed = bool(split_results["dominance_checks_passed"].all())

    validation_passed = bool(
        mean_donor_auc >= min_mean_donor_auc
        and mean_donor_balacc >= min_mean_donor_balanced_accuracy
        and dominance_all_passed
    )
    return {
        "validation_passed": validation_passed,
        "mean_donor_auc": mean_donor_auc,
        "std_donor_auc": std_donor_auc,
        "mean_donor_balanced_accuracy": mean_donor_balacc,
        "std_donor_balanced_accuracy": std_donor_balacc,
        "mean_metacell_auc": mean_metacell_auc,
        "std_metacell_auc": std_metacell_auc,
        "mean_metacell_balanced_accuracy": mean_metacell_balacc,
        "std_metacell_balanced_accuracy": std_metacell_balacc,
        "mean_selected_threshold": mean_selected_threshold,
        "std_selected_threshold": std_selected_threshold,
        "dominance_all_passed": dominance_all_passed,
        "min_mean_donor_auc": float(min_mean_donor_auc),
        "min_mean_donor_balanced_accuracy": float(min_mean_donor_balanced_accuracy),
    }


def fit_final_human_model(
    full_hvg: sc.AnnData,
    model_c: float,
    model_l1_ratio: float,
    model_max_iter: int,
) -> tuple[LogisticRegression, pd.DataFrame]:
    X = full_hvg.X.toarray() if issparse(full_hvg.X) else np.asarray(full_hvg.X)
    y = (full_hvg.obs["Age"].astype(str) == "old").astype(int).to_numpy()
    clf = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=model_c,
        l1_ratio=model_l1_ratio,
        max_iter=model_max_iter,
        class_weight="balanced",
    )
    clf.fit(X, y)
    gene_weights = pd.DataFrame(
        {
            "Gene": full_hvg.var_names.astype(str),
            "Weight": clf.coef_[0],
        }
    ).sort_values(by="Weight", ascending=False).reset_index(drop=True)
    return clf, gene_weights


def run_human_grn(
    adata_musc_pyscenic: sc.AnnData,
    human_db_dir: Path,
    human_tf_list_path: Path,
    human_motif_anno_path: Path,
    num_workers: int,
) -> dict[str, object]:
    sys.path.insert(0, str(ABCLOCK_ROOT))
    import abclock  # noqa: E402

    db_paths = sorted(Path(human_db_dir).glob("*.feather"))
    tf_genes_list = pd.read_csv(human_tf_list_path, header=None)[0].astype(str).tolist()
    return abclock.run_regdiffusion_pyscenic_pipeline(
        adata=adata_musc_pyscenic,
        tf_list=tf_genes_list,
        db_paths=db_paths,
        motif_anno_path=str(human_motif_anno_path),
        num_workers=num_workers,
    )


def run_human_enrichment(gene_weights: pd.DataFrame, human_gmt_path: Path) -> dict[str, pd.DataFrame]:
    sys.path.insert(0, str(ABCLOCK_ROOT))
    import abclock  # noqa: E402

    return abclock.run_age_gene_enrichment(
        gene_weights,
        top_n=100,
        gmt_path=str(human_gmt_path),
        organism="Human",
    )
