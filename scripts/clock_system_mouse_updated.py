#!/usr/bin/env python3
"""
Mouse MuSC baseline age-state classifier training from the cleaned annotated atlas.
"""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import sys
import warnings

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import issparse
from scipy.stats import ConstantInputWarning, spearmanr

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import scanpy as sc

from path_config import ARTIFACT_ROOT, PROJECT_ROOT

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT / "scripts"))

import mouse_workflow_core as mouse_core  # noqa: E402


OUTPUT_DIR = ARTIFACT_ROOT / "clock_artifacts" / "0310_results"
ANNOTATED_ATLAS_PATH = ARTIFACT_ROOT / "clock_artifacts" / "musc_annotation" / "musc_atlas_annotated.h5ad"
TRAINING_ATLAS_PATH = OUTPUT_DIR / "training_atlas_from_cleaned_annotated.h5ad"
MONOCLE3_CELL_TABLE_PATH = ARTIFACT_ROOT / "processed_adata" / "monocle3" / "musc_monocle3_cells.tsv"

# Historical reference:
# raw preprocessing previously produced `training_atlas_trainonly_raw.h5ad`.
# This script now derives a training-ready atlas from the cleaned annotation.

MODEL_PARAMS = {"C": 0.1, "l1_ratio": 0.5}
CANDIDATE = {"C": 0.1, "l1_ratio": 0.5, "calibration": "none"}
ARTICLE_HOLDOUT_SOURCE = "SKM_mouse_raw"
ABLATION_VARIANTS = (
    # {"variant_name": "expression_only", "feature_mode": "expression_only"},
    {"variant_name": "expression_only_local_pseudotime", "feature_mode": "local_pseudotime"},
    # {"variant_name": "expression_plus_pseudotime", "feature_mode": "scalar_pseudotime"},
    # {"variant_name": "expression_plus_dynamic_interactions", "feature_mode": "dynamic_interactions"},
)
DYNAMIC_INTERACTION_TOP_N = 150
DYNAMIC_INTERACTION_FEATURE_PREFIX = "__int_pt__"
PSEUDOTIME_INTERACTION_SCALING = "train_split_minmax_0_1_clipped"
CURATED_DYNAMIC_MARKERS = (
    "Pax7",
    "Myf5",
    "Gpx3",
    "Ryr3",
    "Cd34",
    "Junb",
    "Hes1",
    "Anxa1",
    "Gpx1",
    "Mki67",
    "Cenpa",
    "Myod1",
    "Cdkn1c",
    "Notch1",
    "Anxa2",
    "Myog",
    "Actc1",
    "Cdk6",
    "Ccnd1",
    "Tgfbr3",
    "Smad4",
)

TRAINING_CONFIG = replace(
    mouse_core.TRAINING_CONFIG,
    artifact_mode="metrics_only",
    split_policy="class_balanced_no_coarse",
    test_fraction=0.15,
    n_split_repeats=1,
    outer_split_repeats=1,
    inner_split_repeats=1,
    min_test_donors_per_class=3,
    bootstrap_iters=1000,
    random_state=42,
    n_training_hvg=1500,
    balance_mode="donor_stratified",
    target_class_ratio=1.0,
)

REQUIRED_OBS_COLS = [
    "Age_group_std",
    "sample_id_std",
    "source",
    "donor_bootstrap_id",
    "donor_split_id",
    "Sex_std",
    "celltype_std",
]
PSEUDOTIME_CELL_COL = mouse_core.PSEUDOTIME_CELL_COL
PSEUDOTIME_METACELL_COL = mouse_core.PSEUDOTIME_METACELL_COL
PSEUDOTIME_FEATURE_NAME = mouse_core.PSEUDOTIME_FEATURE_NAME
LOCAL_CLOCK_N_WINDOWS = 5
LOCAL_CLOCK_SIGMA_SCALE = 1.25
LOCAL_CLOCK_MIN_EFFECTIVE_CLASS_WEIGHT = 5.0


def _evaluate_variable_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    y_pred = (y_prob >= thresholds).astype(int)

    recall_old = float(np.mean(y_pred[y_true == 1] == 1)) if np.any(y_true == 1) else 0.0
    recall_young = float(np.mean(y_pred[y_true == 0] == 0)) if np.any(y_true == 0) else 0.0
    balanced_accuracy = float(0.5 * (recall_old + recall_young))
    return {
        "auc": float(mouse_core.abclock_model._safe_auc(y_true, y_prob)),
        "pr_auc_old": float(mouse_core.abclock_model._safe_pr_auc(y_true, y_prob)),
        "recall_old": recall_old,
        "recall_young": recall_young,
        "balanced_accuracy": balanced_accuracy,
    }


def _build_local_window_memberships(
    pseudotime: np.ndarray,
    *,
    n_windows: int = LOCAL_CLOCK_N_WINDOWS,
    sigma_scale: float = LOCAL_CLOCK_SIGMA_SCALE,
) -> dict[str, object]:
    pt = np.asarray(pseudotime, dtype=float)
    if not np.isfinite(pt).all():
        raise ValueError("Encountered non-finite metacell pseudotime while building local windows.")
    if pt.size < 10:
        raise ValueError("Need at least 10 metacells to define local pseudotime windows.")

    pt_min = float(np.min(pt))
    pt_max = float(np.max(pt))
    if np.isclose(pt_min, pt_max):
        centers = np.array([pt_min], dtype=float)
    else:
        centers = np.linspace(pt_min, pt_max, int(max(2, n_windows)), dtype=float)

    if centers.size == 1:
        sigma = 1.0
    else:
        spacing = float(np.median(np.diff(centers)))
        sigma = max(spacing * float(sigma_scale), 1e-6)

    raw = np.exp(-0.5 * ((pt[:, None] - centers[None, :]) / sigma) ** 2)
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    normalized = raw / row_sums
    nearest = np.argmax(normalized, axis=1)

    window_rows = []
    for idx, center in enumerate(centers):
        window_rows.append(
            {
                "window_id": int(idx),
                "center_pseudotime": float(center),
                "support_min": float(center - 2.0 * sigma),
                "support_max": float(center + 2.0 * sigma),
                "sigma": float(sigma),
                "n_nearest_assigned": int(np.sum(nearest == idx)),
            }
        )

    return {
        "centers": centers,
        "sigma": float(sigma),
        "raw_membership": raw,
        "normalized_membership": normalized,
        "nearest_window": nearest,
        "window_summary": pd.DataFrame(window_rows),
    }


def fit_local_pseudotime_clock(
    train_hvg: sc.AnnData,
    *,
    model_params: dict[str, object] | None = None,
    n_windows: int = LOCAL_CLOCK_N_WINDOWS,
    random_state: int = 42,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    base_train = _drop_scalar_pseudotime_feature(train_hvg)
    if PSEUDOTIME_METACELL_COL not in base_train.obs.columns:
        raise ValueError(f"{PSEUDOTIME_METACELL_COL} is required for local pseudotime clocks.")

    pt = pd.to_numeric(base_train.obs[PSEUDOTIME_METACELL_COL], errors="coerce").to_numpy(dtype=float)
    memberships = _build_local_window_memberships(pt, n_windows=n_windows)
    X_train = mouse_core.dense_matrix(base_train.X)
    y_train = (base_train.obs["Age"].astype(str) == "old").astype(int).to_numpy()

    window_models: list[object] = []
    window_thresholds: list[float] = []
    threshold_tables: list[pd.DataFrame] = []
    window_prob_matrix: list[np.ndarray] = []
    coef_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []

    for idx, center in enumerate(memberships["centers"]):
        sample_weight = memberships["raw_membership"][:, idx].astype(float)
        old_eff = float(sample_weight[y_train == 1].sum())
        young_eff = float(sample_weight[y_train == 0].sum())
        if old_eff < LOCAL_CLOCK_MIN_EFFECTIVE_CLASS_WEIGHT or young_eff < LOCAL_CLOCK_MIN_EFFECTIVE_CLASS_WEIGHT:
            raise ValueError(
                f"Local window {idx} has insufficient effective class weight: old={old_eff:.3f}, young={young_eff:.3f}"
            )

        clf = mouse_core.abclock_model.fit_elasticnet_classifier(
            base_train,
            age_col="Age",
            model_params=model_params,
            sample_weight=sample_weight,
        )
        window_models.append(clf)

        probs = clf.predict_proba(X_train)[:, 1]
        window_prob_matrix.append(probs)

        assigned_mask = memberships["nearest_window"] == idx
        if np.sum(assigned_mask) >= 10 and np.unique(y_train[assigned_mask]).size >= 2:
            selected_threshold, threshold_table = mouse_core.abclock_model.select_optimal_threshold(
                y_train[assigned_mask],
                probs[assigned_mask],
                objective=TRAINING_CONFIG.threshold_objective,
            )
        else:
            selected_threshold = 0.5
            threshold_table = pd.DataFrame(
                [
                    {
                        "threshold": 0.5,
                        "balanced_accuracy": float("nan"),
                        "recall_old": float("nan"),
                        "recall_young": float("nan"),
                        "precision_old": float("nan"),
                        "f1_old": float("nan"),
                        "youden_j": float("nan"),
                        "distance_to_0p5": 0.0,
                    }
                ]
            )
        window_thresholds.append(float(selected_threshold))
        threshold_table["window_id"] = int(idx)
        threshold_table["center_pseudotime"] = float(center)
        threshold_tables.append(threshold_table)

        coef = np.asarray(clf.coef_[0], dtype=float)
        for gene, weight in zip(base_train.var_names.astype(str), coef):
            coef_rows.append(
                {
                    "window_id": int(idx),
                    "center_pseudotime": float(center),
                    "Gene": str(gene),
                    "Weight": float(weight),
                }
            )

        window_rows.append(
            {
                "window_id": int(idx),
                "center_pseudotime": float(center),
                "sigma": float(memberships["sigma"]),
                "support_min": float(center - 2.0 * memberships["sigma"]),
                "support_max": float(center + 2.0 * memberships["sigma"]),
                "selected_threshold": float(selected_threshold),
                "n_nearest_assigned": int(np.sum(assigned_mask)),
                "effective_old_weight": old_eff,
                "effective_young_weight": young_eff,
            }
        )

    bundle = {
        "model_family": "local_pseudotime_elasticnet",
        "gene_names": base_train.var_names.astype(str).tolist(),
        "window_centers": [float(x) for x in memberships["centers"]],
        "window_sigma": float(memberships["sigma"]),
        "window_models": window_models,
        "window_thresholds": [float(x) for x in window_thresholds],
        "n_windows": int(len(window_models)),
        "selected_threshold": float(np.mean(window_thresholds)),
        "metadata": {
            "threshold_objective": str(TRAINING_CONFIG.threshold_objective),
            "n_windows": int(len(window_models)),
            "sigma": float(memberships["sigma"]),
            "random_state": int(random_state),
        },
    }
    return bundle, pd.DataFrame(window_rows), pd.concat(threshold_tables, ignore_index=True)


def predict_local_pseudotime_clock(
    model_bundle: dict[str, object],
    adata: sc.AnnData,
) -> pd.DataFrame:
    base_adata = _drop_scalar_pseudotime_feature(adata)
    if PSEUDOTIME_METACELL_COL not in base_adata.obs.columns:
        raise ValueError(f"{PSEUDOTIME_METACELL_COL} is required for local pseudotime inference.")

    gene_names = [str(g) for g in model_bundle["gene_names"]]
    if list(base_adata.var_names.astype(str)) != gene_names:
        raise ValueError("Local pseudotime clock inference requires identical gene ordering to training.")

    pt = pd.to_numeric(base_adata.obs[PSEUDOTIME_METACELL_COL], errors="coerce").to_numpy(dtype=float)
    centers = np.asarray(model_bundle["window_centers"], dtype=float)
    sigma = float(model_bundle["window_sigma"])
    raw = np.exp(-0.5 * ((pt[:, None] - centers[None, :]) / sigma) ** 2)
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    memberships = raw / row_sums

    X = mouse_core.dense_matrix(base_adata.X)
    prob_matrix = np.column_stack(
        [np.asarray(clf.predict_proba(X)[:, 1], dtype=float) for clf in model_bundle["window_models"]]
    )
    thresholds = np.asarray(model_bundle["window_thresholds"], dtype=float)
    score = np.sum(memberships * prob_matrix, axis=1)
    local_threshold = np.sum(memberships * thresholds[None, :], axis=1)
    nearest_idx = np.argmax(memberships, axis=1)

    out = pd.DataFrame(
        {
            "p_old": score,
            "local_threshold": local_threshold,
            "pred_old_thr_local": (score >= local_threshold).astype(int),
            "nearest_window_id": nearest_idx.astype(int),
            "pseudotime_metacell": pt,
        },
        index=base_adata.obs_names.astype(str),
    )
    for idx in range(prob_matrix.shape[1]):
        out[f"window_{idx}_membership"] = memberships[:, idx]
        out[f"window_{idx}_p_old"] = prob_matrix[:, idx]
    return out


def summarize_local_donor_predictions(
    adata: sc.AnnData,
    pred_df: pd.DataFrame,
    *,
    donor_col: str = "donor_split_id",
    age_col: str = "Age",
    extra_group_cols: list[str] | None = None,
) -> pd.DataFrame:
    extra_group_cols = list(extra_group_cols or [])
    obs = adata.obs.copy()
    obs["p_old"] = pred_df["p_old"].to_numpy(dtype=float)
    obs["local_threshold"] = pred_df["local_threshold"].to_numpy(dtype=float)
    grouped = (
        obs.groupby([donor_col, age_col] + [col for col in extra_group_cols if col in obs.columns], sort=True)
        .agg(
            mean_p_old=("p_old", "mean"),
            median_p_old=("p_old", "median"),
            mean_local_threshold=("local_threshold", "mean"),
            n_metacells=("p_old", "size"),
        )
        .reset_index()
    )
    grouped = grouped.rename(columns={donor_col: "donor_id", age_col: "age_group"})
    grouped["pred_old_thr_0p5"] = (grouped["mean_p_old"] >= 0.5).astype(int)
    grouped["pred_old_thr_opt"] = (grouped["mean_p_old"] >= grouped["mean_local_threshold"]).astype(int)
    return grouped


def fit_local_clock_on_processed_split(
    train_hvg: sc.AnnData,
    test_hvg: sc.AnnData,
    *,
    split_manifest: dict[str, object],
    model_params: dict[str, object] | None = None,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
    bundle, window_summary, threshold_table = fit_local_pseudotime_clock(
        train_hvg,
        model_params=model_params,
        n_windows=LOCAL_CLOCK_N_WINDOWS,
        random_state=int(TRAINING_CONFIG.random_state),
    )
    train_pred = predict_local_pseudotime_clock(bundle, train_hvg)
    test_pred = predict_local_pseudotime_clock(bundle, test_hvg)

    train_y = (train_hvg.obs["Age"].astype(str) == "old").astype(int).to_numpy()
    test_y = (test_hvg.obs["Age"].astype(str) == "old").astype(int).to_numpy()

    train_metrics_default = mouse_core.abclock_model.evaluate_binary_metrics(train_y, train_pred["p_old"].to_numpy(), threshold=0.5)
    test_metrics_default = mouse_core.abclock_model.evaluate_binary_metrics(test_y, test_pred["p_old"].to_numpy(), threshold=0.5)
    train_metrics_opt = _evaluate_variable_threshold_metrics(train_y, train_pred["p_old"].to_numpy(), train_pred["local_threshold"].to_numpy())
    test_metrics_opt = _evaluate_variable_threshold_metrics(test_y, test_pred["p_old"].to_numpy(), test_pred["local_threshold"].to_numpy())

    donor_predictions = summarize_local_donor_predictions(
        test_hvg,
        test_pred,
        extra_group_cols=["sample_id_std", "source"],
    )
    donor_y = (donor_predictions["age_group"].astype(str) == "old").astype(int).to_numpy()
    donor_prob = donor_predictions["mean_p_old"].to_numpy(dtype=float)
    donor_thr = donor_predictions["mean_local_threshold"].to_numpy(dtype=float)
    donor_metrics_default = mouse_core.abclock_model.evaluate_binary_metrics(donor_y, donor_prob, threshold=0.5)
    donor_metrics_opt = _evaluate_variable_threshold_metrics(donor_y, donor_prob, donor_thr)

    split_row = {
        "split_id": split_manifest.get("split_id", "fixed_direct_split"),
        "threshold_source": "local_window_train_only",
        "selected_threshold": float(bundle["selected_threshold"]),
        "selected_threshold_source": "local_window_train_only",
        "train_auc_train_only": float(train_metrics_default["auc"]),
        "train_metacell_balanced_accuracy_thr_opt": float(train_metrics_opt["balanced_accuracy"]),
        "holdout_metacell_auc": float(test_metrics_default["auc"]),
        "holdout_metacell_balanced_accuracy_thr_0p5": float(test_metrics_default["balanced_accuracy"]),
        "holdout_metacell_balanced_accuracy_thr_opt": float(test_metrics_opt["balanced_accuracy"]),
        "holdout_metacell_recall_old_thr_opt": float(test_metrics_opt["recall_old"]),
        "holdout_metacell_recall_young_thr_opt": float(test_metrics_opt["recall_young"]),
        "holdout_metacell_pr_auc_old": float(test_metrics_default["pr_auc_old"]),
        "holdout_donor_auc": float(donor_metrics_default["auc"]),
        "holdout_donor_balanced_accuracy_thr_0p5": float(donor_metrics_default["balanced_accuracy"]),
        "holdout_donor_balanced_accuracy_thr_opt": float(donor_metrics_opt["balanced_accuracy"]),
        "holdout_donor_recall_old_thr_opt": float(donor_metrics_opt["recall_old"]),
        "holdout_donor_recall_young_thr_opt": float(donor_metrics_opt["recall_young"]),
        "holdout_donor_pr_auc_old": float(donor_metrics_default["pr_auc_old"]),
        "n_train_donors": int(len(split_manifest.get("train_donors", []))),
        "n_test_donors": int(len(split_manifest.get("test_donors", []))),
        "n_train_metacells": int(train_hvg.n_obs),
        "n_test_metacells": int(test_hvg.n_obs),
        "n_training_genes": int(train_hvg.n_vars),
    }

    metacell_predictions = pd.concat(
        [
            train_hvg.obs.assign(split_set="train").reset_index().rename(columns={"index": "metacell_id"}),
            test_hvg.obs.assign(split_set="test").reset_index().rename(columns={"index": "metacell_id"}),
        ],
        ignore_index=True,
    )
    pred_concat = pd.concat(
        [
            train_pred.assign(split_set="train").reset_index().rename(columns={"index": "metacell_id"}),
            test_pred.assign(split_set="test").reset_index().rename(columns={"index": "metacell_id"}),
        ],
        ignore_index=True,
    )
    metacell_predictions = metacell_predictions.merge(pred_concat, on=["metacell_id", "split_set"], how="left", validate="one_to_one")
    return bundle, donor_predictions, mouse_core.pd.DataFrame([split_row]), {"window_summary": window_summary, "threshold_table": threshold_table}, metacell_predictions


def _build_adata_with_appended_features(
    adata: sc.AnnData,
    feature_matrix: np.ndarray | sparse.spmatrix,
    feature_names: list[str],
) -> sc.AnnData:
    if not feature_names:
        return adata

    if issparse(adata.X):
        appended_x = sparse.hstack([adata.X, sparse.csr_matrix(feature_matrix)], format="csr")
    else:
        appended_x = np.hstack([np.asarray(adata.X), np.asarray(feature_matrix, dtype=float)])

    new_var = adata.var.copy()
    feature_row = {}
    for col, dtype in new_var.dtypes.items():
        if pd.api.types.is_bool_dtype(dtype):
            feature_row[col] = False
        elif pd.api.types.is_numeric_dtype(dtype):
            feature_row[col] = 0.0
        else:
            feature_row[col] = ""
    for feature_name in feature_names:
        new_var.loc[feature_name] = pd.Series(feature_row)
    new_adata = sc.AnnData(X=appended_x, obs=adata.obs.copy(), var=new_var)
    for key, value in adata.uns.items():
        new_adata.uns[key] = value
    for key, value in adata.obsm.items():
        new_adata.obsm[key] = value
    for key, value in adata.varm.items():
        new_adata.varm[key] = value
    return new_adata


def _drop_scalar_pseudotime_feature(adata: sc.AnnData) -> sc.AnnData:
    if PSEUDOTIME_FEATURE_NAME not in adata.var_names:
        return adata
    return adata[:, adata.var_names.astype(str) != PSEUDOTIME_FEATURE_NAME].copy()


def _count_base_gene_features(adata: sc.AnnData) -> int:
    names = adata.var_names.astype(str)
    return int(
        np.sum(
            [
                (name != PSEUDOTIME_FEATURE_NAME) and (not name.startswith(DYNAMIC_INTERACTION_FEATURE_PREFIX))
                for name in names
            ]
        )
    )


def _minmax_scale_pseudotime(values: np.ndarray, *, pt_min: float, pt_max: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Encountered non-finite pseudotime values while constructing interaction features.")
    if not np.isfinite(pt_min) or not np.isfinite(pt_max):
        raise ValueError("Encountered non-finite pseudotime scaling bounds.")
    if pt_max <= pt_min:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - pt_min) / (pt_max - pt_min), 0.0, 1.0)


def fit_dynamic_interaction_spec(train_adata: sc.AnnData) -> dict[str, object]:
    base_train = _drop_scalar_pseudotime_feature(train_adata)
    if PSEUDOTIME_METACELL_COL not in base_train.obs.columns:
        raise ValueError(f"{PSEUDOTIME_METACELL_COL} is required for dynamic interaction features.")

    pt_all = pd.to_numeric(base_train.obs[PSEUDOTIME_METACELL_COL], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(pt_all).all():
        raise ValueError("Training metacells contain non-finite Monocle3 pseudotime values.")

    age = base_train.obs["Age"].astype(str).to_numpy()
    young_mask = age == "young"
    if int(np.sum(young_mask)) < 10:
        raise ValueError("Need at least 10 young training metacells to define dynamic interaction genes.")

    pt_young = pt_all[young_mask]
    expr = np.asarray(mouse_core.dense_matrix(base_train.X), dtype=float)
    expr_young = expr[young_mask, :]

    score_rows: list[dict[str, object]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        for idx, gene in enumerate(base_train.var_names.astype(str)):
            rho = spearmanr(expr_young[:, idx], pt_young).correlation
            if not np.isfinite(rho):
                rho = 0.0
            score_rows.append(
                {
                    "gene": gene,
                    "spearman_rho_young_train": float(rho),
                    "abs_spearman_rho_young_train": float(abs(rho)),
                }
            )

    score_df = pd.DataFrame(score_rows).sort_values(
        ["abs_spearman_rho_young_train", "gene"],
        ascending=[False, True],
    ).reset_index(drop=True)
    score_df["dynamic_rank"] = np.arange(1, len(score_df) + 1, dtype=int)

    top_dynamic = score_df.head(DYNAMIC_INTERACTION_TOP_N)["gene"].astype(str).tolist()
    curated_present = [gene for gene in CURATED_DYNAMIC_MARKERS if gene in base_train.var_names]

    selected_genes: list[str] = []
    for gene in top_dynamic + curated_present:
        if gene not in selected_genes:
            selected_genes.append(gene)

    selected_set = set(selected_genes)
    top_dynamic_set = set(top_dynamic)
    curated_set = set(curated_present)
    score_df["selected_for_interaction"] = score_df["gene"].isin(selected_set)
    score_df["selected_by_dynamic_rank"] = score_df["gene"].isin(top_dynamic_set)
    score_df["selected_by_curated_marker"] = score_df["gene"].isin(curated_set)

    panel_df = score_df.loc[score_df["selected_for_interaction"]].copy()
    panel_df["selection_reason"] = np.where(
        panel_df["selected_by_dynamic_rank"] & panel_df["selected_by_curated_marker"],
        "dynamic_and_curated",
        np.where(panel_df["selected_by_dynamic_rank"], "dynamic_top150", "curated_only"),
    )
    panel_df = panel_df[
        [
            "gene",
            "dynamic_rank",
            "spearman_rho_young_train",
            "abs_spearman_rho_young_train",
            "selected_by_dynamic_rank",
            "selected_by_curated_marker",
            "selection_reason",
        ]
    ].reset_index(drop=True)

    return {
        "selected_genes": selected_genes,
        "score_table": score_df,
        "panel_table": panel_df,
        "pseudotime_min": float(np.min(pt_all)),
        "pseudotime_max": float(np.max(pt_all)),
        "interaction_scaling": PSEUDOTIME_INTERACTION_SCALING,
        "top_dynamic_gene_count": int(len(top_dynamic)),
        "curated_marker_count_requested": int(len(CURATED_DYNAMIC_MARKERS)),
        "curated_marker_count_present": int(len(curated_present)),
        "curated_marker_count_added_only": int(sum(gene not in top_dynamic_set for gene in curated_present)),
    }


def apply_dynamic_interaction_spec(
    adata: sc.AnnData,
    spec: dict[str, object],
    *,
    allow_missing_genes: bool = False,
) -> sc.AnnData:
    base_adata = _drop_scalar_pseudotime_feature(adata)
    if PSEUDOTIME_METACELL_COL not in base_adata.obs.columns:
        raise ValueError(f"{PSEUDOTIME_METACELL_COL} is required for dynamic interaction features.")

    selected_genes = [str(gene) for gene in spec["selected_genes"]]
    missing = [gene for gene in selected_genes if gene not in base_adata.var_names]
    if missing:
        if not allow_missing_genes:
            raise ValueError(f"Dynamic interaction genes missing from matrix: {missing[:10]}")
        selected_genes = [gene for gene in selected_genes if gene in base_adata.var_names]
        if not selected_genes:
            raise ValueError("No dynamic interaction genes remain after dropping missing genes.")

    pt_raw = pd.to_numeric(base_adata.obs[PSEUDOTIME_METACELL_COL], errors="coerce").to_numpy(dtype=float)
    pt_scaled = _minmax_scale_pseudotime(
        pt_raw,
        pt_min=float(spec["pseudotime_min"]),
        pt_max=float(spec["pseudotime_max"]),
    )
    base_slice = base_adata[:, selected_genes]
    base_x = base_slice.X
    if issparse(base_x):
        interaction_x = base_x.multiply(pt_scaled[:, None]).tocsr()
    else:
        interaction_x = np.asarray(base_x, dtype=float) * pt_scaled[:, None]

    feature_names = [f"{DYNAMIC_INTERACTION_FEATURE_PREFIX}{gene}" for gene in selected_genes]
    out = _build_adata_with_appended_features(base_adata, interaction_x, feature_names)
    out.obs["pseudotime_interaction_0to1"] = pt_scaled
    out.uns["dynamic_interaction_genes_applied"] = selected_genes
    out.uns["dynamic_interaction_genes_missing"] = missing
    return out


def load_monocle3_pseudotime(cell_table_path: Path) -> pd.Series:
    cell_table = pd.read_csv(cell_table_path, sep="\t")
    required_cols = {"cell_id", "pseudotime"}
    missing_cols = sorted(required_cols - set(cell_table.columns))
    if missing_cols:
        raise ValueError(f"Monocle3 cell table missing required columns: {missing_cols}")

    cell_table["cell_id"] = cell_table["cell_id"].astype(str)
    if cell_table["cell_id"].duplicated().any():
        duplicated = cell_table.loc[cell_table["cell_id"].duplicated(), "cell_id"].astype(str).unique().tolist()[:10]
        raise ValueError(f"Monocle3 cell table contains duplicated cell_id values: {duplicated}")

    pseudotime = pd.to_numeric(cell_table["pseudotime"], errors="coerce")
    if not pseudotime.notna().all():
        raise ValueError("Monocle3 pseudotime contains missing or non-numeric values.")

    return pd.Series(pseudotime.to_numpy(dtype=float), index=cell_table["cell_id"], name=PSEUDOTIME_CELL_COL)


def attach_monocle3_pseudotime(adata: sc.AnnData, cell_table_path: Path) -> sc.AnnData:
    pseudotime = load_monocle3_pseudotime(cell_table_path)
    obs_index = adata.obs_names.astype(str)
    missing_cells = obs_index[~obs_index.isin(pseudotime.index)]
    extra_cells = pseudotime.index[~pseudotime.index.isin(obs_index)]
    if len(missing_cells) > 0 or len(extra_cells) > 0:
        raise ValueError(
            "Monocle3 pseudotime cell IDs do not exactly match the training atlas. "
            f"missing_in_monocle3={list(missing_cells[:10])}, extra_in_monocle3={list(extra_cells[:10])}"
        )

    adata.obs[PSEUDOTIME_CELL_COL] = pseudotime.reindex(obs_index).to_numpy(dtype=float)
    return adata


def prepare_training_atlas_from_cleaned_annotated(
    source_path: Path,
    output_path: Path,
) -> Path:
    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    newest_input_mtime = max(source_path.stat().st_mtime, MONOCLE3_CELL_TABLE_PATH.stat().st_mtime)

    if output_path.exists() and output_path.stat().st_mtime >= newest_input_mtime:
        cached = sc.read_h5ad(output_path)
        try:
            if PSEUDOTIME_CELL_COL in cached.obs.columns:
                return output_path
        finally:
            if getattr(cached, "file", None) is not None:
                cached.file.close()

    shutil.copy2(source_path, output_path)

    with h5py.File(output_path, "r+") as f:
        if "layers" not in f or "counts" not in f["layers"]:
            raise ValueError("cleaned atlas must contain raw counts in /layers/counts.")

        if "X" in f:
            del f["X"]
        f.copy(f["layers"]["counts"], "X")

        for key in ("uns", "obsm", "obsp", "raw", "varm", "varp"):
            if key in f:
                del f[key]

    adata = sc.read_h5ad(output_path)
    try:
        adata = attach_monocle3_pseudotime(adata, MONOCLE3_CELL_TABLE_PATH)
        missing_obs = [col for col in REQUIRED_OBS_COLS if col not in adata.obs.columns]
        if missing_obs:
            raise ValueError(f"derived training atlas missing required obs columns: {missing_obs}")
        if "counts" not in adata.layers:
            raise ValueError("derived training atlas missing .layers['counts'].")
        if PSEUDOTIME_CELL_COL not in adata.obs.columns:
            raise ValueError(f"derived training atlas missing {PSEUDOTIME_CELL_COL}.")
        mouse_core.safe_write_h5ad(adata, output_path)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    return output_path


def evaluate_single_source_holdout(
    adata_all: sc.AnnData,
    source_name: str,
    *,
    feature_mode: str,
) -> dict[str, object]:
    source_mask = adata_all.obs["source"].astype(str) == str(source_name)
    source_adata = adata_all[source_mask].copy()
    source_donor_table = mouse_core.build_donor_table(source_adata)
    class_counts = source_donor_table["age_label"].value_counts().to_dict()

    result: dict[str, object] = {
        "holdout_source": str(source_name),
        "feature_mode": str(feature_mode),
        "evaluable": False,
        "reason": None,
        "metrics_row": None,
        "selection_summary": mouse_core.pd.DataFrame(),
        "threshold_diagnostics": mouse_core.pd.DataFrame(),
        "n_holdout_cells": int(source_adata.n_obs),
        "n_holdout_donors": int(source_donor_table.shape[0]),
        "young_holdout_donors": int(class_counts.get("young", 0)),
        "old_holdout_donors": int(class_counts.get("old", 0)),
    }

    if class_counts.get("young", 0) < int(TRAINING_CONFIG.min_test_donors_per_class) or class_counts.get(
        "old", 0
    ) < int(TRAINING_CONFIG.min_test_donors_per_class):
        result["reason"] = "insufficient_test_donors_per_class"
        return result

    holdout_donors = set(source_donor_table["donor_id"].astype(str))
    train_mask = ~adata_all.obs["donor_split_id"].astype(str).isin(holdout_donors)
    train_raw = adata_all[train_mask].copy()
    test_raw = adata_all[~train_mask].copy()

    train_donor_table = mouse_core.build_donor_table(train_raw)
    train_class_counts = train_donor_table["age_label"].value_counts().to_dict()
    if train_class_counts.get("young", 0) < 2 or train_class_counts.get("old", 0) < 2:
        result["reason"] = "insufficient_training_donors_per_class"
        return result

    _, train_bs = mouse_core.generate_mouse_bootstrap_metacells(
        train_raw,
        TRAINING_CONFIG,
        random_state=int(TRAINING_CONFIG.random_state) + 2000,
        rebalance=True,
    )
    _, test_bs = mouse_core.generate_mouse_bootstrap_metacells(
        test_raw,
        TRAINING_CONFIG,
        random_state=int(TRAINING_CONFIG.random_state) + 3000,
        rebalance=False,
    )
    train_hvg, test_hvg, _ = mouse_core.preprocess_mouse_metacell_pair(train_bs, test_bs, TRAINING_CONFIG)
    train_hvg, test_hvg, _ = prepare_variant_matrices(train_hvg, test_hvg, feature_mode=feature_mode)

    train_prob = mouse_core.abclock_model.fit_elasticnet_classifier(
        train_hvg,
        age_col="Age",
        model_params=MODEL_PARAMS,
    ).predict_proba(mouse_core.dense_matrix(train_hvg.X))[:, 1]
    train_y = (train_hvg.obs["Age"].astype(str) == "old").astype(int).to_numpy()
    selected_threshold, threshold_table = mouse_core.abclock_model.select_optimal_threshold(
        train_y,
        train_prob,
        objective=TRAINING_CONFIG.threshold_objective,
    )
    split_manifest = {
        "split_id": f"study_holdout::{source_name}",
        "train_donors": sorted(train_raw.obs["donor_split_id"].astype(str).unique().tolist()),
        "test_donors": sorted(test_raw.obs["donor_split_id"].astype(str).unique().tolist()),
        "train_test_donor_overlap_count": 0,
    }
    _, donor_predictions, split_row, _ = mouse_core.fit_mouse_candidate_on_processed_split(
        train_hvg,
        test_hvg,
        CANDIDATE,
        selected_threshold=float(selected_threshold),
        calibrator=None,
        split_manifest=split_manifest,
        threshold_source="train_only",
    )

    split_row["holdout_source"] = str(source_name)
    split_row["context_label"] = f"study_holdout::{source_name}"
    split_row["n_holdout_cells"] = int(test_raw.n_obs)
    split_row["n_holdout_train_cells"] = int(train_raw.n_obs)
    split_row["n_holdout_train_donors"] = int(len(split_manifest["train_donors"]))
    split_row["feature_mode"] = str(feature_mode)

    result.update(
        {
            "evaluable": True,
            "metrics_row": split_row,
            "selection_summary": mouse_core.pd.DataFrame([{"candidate_id": "fixed_main_model", **CANDIDATE}]),
            "threshold_diagnostics": threshold_table,
            "donor_predictions": donor_predictions,
        }
    )
    return result


def compare_threshold_settings(
    donor_predictions: pd.DataFrame,
    *,
    train_selected_threshold: float,
    context_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    if donor_predictions.empty:
        raise ValueError("donor_predictions is empty.")

    work = donor_predictions.copy()
    if "age_group" not in work.columns or "mean_p_old" not in work.columns:
        raise ValueError("donor_predictions must contain age_group and mean_p_old.")

    y_true = (work["age_group"].astype(str) == "old").astype(int).to_numpy()
    y_prob = work["mean_p_old"].to_numpy(dtype=float)

    source_threshold, source_threshold_table = mouse_core.abclock_model.select_optimal_threshold(
        y_true,
        y_prob,
        objective=TRAINING_CONFIG.threshold_objective,
    )
    source_threshold_table["context_label"] = str(context_label)
    source_threshold_table["threshold_setting"] = "source_recalibrated_candidates"

    settings = [
        ("fixed_0p5", 0.5, "default"),
        ("train_split_optimal", float(train_selected_threshold), "training_split"),
        ("source_recalibrated", float(source_threshold), "source_holdout_oracle"),
    ]

    rows: list[dict[str, object]] = []
    for setting_name, threshold_value, threshold_source in settings:
        metrics = mouse_core.abclock_model.evaluate_binary_metrics(y_true, y_prob, threshold=threshold_value)
        rows.append(
            {
                "context_label": str(context_label),
                "threshold_setting": str(setting_name),
                "threshold_source": str(threshold_source),
                "threshold": float(threshold_value),
                "n_donors": int(work.shape[0]),
                "auc": float(metrics["auc"]),
                "pr_auc_old": float(metrics["pr_auc_old"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "recall_old": float(metrics["recall_old"]),
                "recall_young": float(metrics["recall_young"]),
            }
        )

    return pd.DataFrame(rows), source_threshold_table, float(source_threshold)


def plot_source_holdout_thresholds(
    donor_predictions: pd.DataFrame,
    *,
    train_selected_threshold: float,
    source_recalibrated_threshold: float,
    output_path: Path,
) -> Path:
    work = donor_predictions.copy().sort_values("mean_p_old").reset_index(drop=True)
    work["y_pos"] = np.arange(work.shape[0])
    colors = np.where(work["age_group"].astype(str) == "old", "#B55D3D", "#4F7A5A")

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.scatter(work["mean_p_old"], work["y_pos"], c=colors, s=55, edgecolors="white", linewidths=0.6, zorder=3)

    for x, color, label in [
        (0.5, "#6B7280", "0.5"),
        (float(train_selected_threshold), "#2B6CB0", "Train-opt"),
        (float(source_recalibrated_threshold), "#C05621", "Source-recal"),
    ]:
        ax.axvline(x, color=color, linestyle="--", linewidth=2, label=f"{label}: {x:.3f}")

    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Donor mean p_old")
    ax.set_ylabel("Source-holdout donors")
    ax.set_yticks([])
    ax.set_title("SKM_mouse_raw Donor Scores and Threshold Settings", fontweight="bold", pad=14)
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def prepare_variant_matrices(
    train_hvg: sc.AnnData,
    test_hvg: sc.AnnData | None,
    *,
    feature_mode: str,
) -> tuple[sc.AnnData, sc.AnnData | None, dict[str, object]]:
    feature_mode = str(feature_mode)
    if feature_mode == "expression_only":
        return _drop_scalar_pseudotime_feature(train_hvg), (
            _drop_scalar_pseudotime_feature(test_hvg) if test_hvg is not None else None
        ), {
            "feature_mode": feature_mode,
            "uses_pseudotime_covariate": False,
            "uses_interaction_features": False,
            "uses_local_pseudotime_model": False,
        }

    if feature_mode == "local_pseudotime":
        return _drop_scalar_pseudotime_feature(train_hvg), (
            _drop_scalar_pseudotime_feature(test_hvg) if test_hvg is not None else None
        ), {
            "feature_mode": feature_mode,
            "uses_pseudotime_covariate": False,
            "uses_interaction_features": False,
            "uses_local_pseudotime_model": True,
            "local_window_count": int(LOCAL_CLOCK_N_WINDOWS),
            "local_window_sigma_scale": float(LOCAL_CLOCK_SIGMA_SCALE),
        }

    if feature_mode == "scalar_pseudotime":
        if PSEUDOTIME_FEATURE_NAME not in train_hvg.var_names:
            raise ValueError(f"{PSEUDOTIME_FEATURE_NAME} missing for pseudotime-enabled variant.")
        if test_hvg is not None and PSEUDOTIME_FEATURE_NAME not in test_hvg.var_names:
            raise ValueError(f"{PSEUDOTIME_FEATURE_NAME} missing in test matrix for pseudotime-enabled variant.")
        return train_hvg, test_hvg, {
            "feature_mode": feature_mode,
            "uses_pseudotime_covariate": True,
            "uses_interaction_features": False,
            "uses_local_pseudotime_model": False,
        }

    if feature_mode == "dynamic_interactions":
        spec = fit_dynamic_interaction_spec(train_hvg)
        train_dynamic = apply_dynamic_interaction_spec(train_hvg, spec)
        test_dynamic = apply_dynamic_interaction_spec(test_hvg, spec) if test_hvg is not None else None
        applied_genes = list(train_dynamic.uns.get("dynamic_interaction_genes_applied", spec["selected_genes"]))
        diag = {
            "feature_mode": feature_mode,
            "uses_pseudotime_covariate": False,
            "uses_interaction_features": True,
            "uses_local_pseudotime_model": False,
            "dynamic_gene_selection_method": "abs_spearman_young_train_metacells",
            "dynamic_top_n": int(DYNAMIC_INTERACTION_TOP_N),
            "dynamic_panel_size": int(len(spec["selected_genes"])),
            "dynamic_panel_size_applied_split": int(len(applied_genes)),
            "dynamic_panel_size_precurated": int(spec["top_dynamic_gene_count"]),
            "dynamic_curated_marker_count_requested": int(spec["curated_marker_count_requested"]),
            "dynamic_curated_marker_count_present": int(spec["curated_marker_count_present"]),
            "dynamic_curated_marker_count_added": int(spec["curated_marker_count_added_only"]),
            "pseudotime_interaction_scaling": str(spec["interaction_scaling"]),
            "pseudotime_min": float(spec["pseudotime_min"]),
            "pseudotime_max": float(spec["pseudotime_max"]),
            "dynamic_interaction_panel": list(spec["selected_genes"]),
            "dynamic_interaction_panel_applied_split": applied_genes,
            "dynamic_interaction_score_table": spec["score_table"],
            "dynamic_interaction_panel_table": spec["panel_table"],
            "pseudotime_feature_name": None,
        }
        return train_dynamic, test_dynamic, diag

    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def run_variant_training(
    *,
    variant_name: str,
    feature_mode: str,
    adata_combined: sc.AnnData,
    train_raw: sc.AnnData,
    test_raw: sc.AnnData,
    split_manifest: dict[str, object],
    train_bs_pre: sc.AnnData,
    train_bs: sc.AnnData,
    test_bs: sc.AnnData,
    full_bs: sc.AnnData,
) -> dict[str, object]:
    print(f"[4/5] Training variant `{variant_name}` (feature_mode={feature_mode})")
    variant_dir = OUTPUT_DIR / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    train_hvg, test_hvg, prep_diag = mouse_core.preprocess_mouse_metacell_pair(
        train_bs,
        test_bs,
        TRAINING_CONFIG,
    )
    train_hvg, test_hvg, variant_diag = prepare_variant_matrices(train_hvg, test_hvg, feature_mode=feature_mode)
    uses_local_model = bool(variant_diag.get("uses_local_pseudotime_model", False))

    if uses_local_model:
        local_bundle, donor_predictions, split_row_df, local_diag, metacell_predictions = fit_local_clock_on_processed_split(
            train_hvg,
            test_hvg,
            split_manifest=split_manifest,
            model_params=MODEL_PARAMS,
        )
        split_row = split_row_df.iloc[0].to_dict()
        selected_threshold = float(local_bundle["selected_threshold"])
        threshold_table = local_diag["threshold_table"]
    else:
        train_prob = mouse_core.abclock_model.fit_elasticnet_classifier(
            train_hvg,
            age_col="Age",
            model_params=MODEL_PARAMS,
        ).predict_proba(mouse_core.dense_matrix(train_hvg.X))[:, 1]
        train_y = (train_hvg.obs["Age"].astype(str) == "old").astype(int).to_numpy()
        selected_threshold, threshold_table = mouse_core.abclock_model.select_optimal_threshold(
            train_y,
            train_prob,
            objective=TRAINING_CONFIG.threshold_objective,
        )

        _, donor_predictions, split_row, _ = mouse_core.fit_mouse_candidate_on_processed_split(
            train_hvg,
            test_hvg,
            candidate=CANDIDATE,
            selected_threshold=float(selected_threshold),
            calibrator=None,
            split_manifest=split_manifest,
            threshold_source="train_only",
        )
        local_bundle = None
        local_diag = None
        metacell_predictions = None

    full_processed, _, full_diag = mouse_core.preprocess_mouse_full_metacells(full_bs, TRAINING_CONFIG)
    if feature_mode == "dynamic_interactions":
        full_processed = apply_dynamic_interaction_spec(
            full_processed,
            {
                "selected_genes": variant_diag["dynamic_interaction_panel"],
                "pseudotime_min": variant_diag["pseudotime_min"],
                "pseudotime_max": variant_diag["pseudotime_max"],
            },
            allow_missing_genes=True,
        )
        variant_diag["dynamic_interaction_panel_applied_full"] = list(
            full_processed.uns.get("dynamic_interaction_genes_applied", [])
        )
        variant_diag["dynamic_interaction_genes_missing_full"] = list(
            full_processed.uns.get("dynamic_interaction_genes_missing", [])
        )
    elif uses_local_model:
        full_processed, _, _ = prepare_variant_matrices(full_processed, None, feature_mode="expression_only")
        final_bundle, full_window_summary, full_threshold_table = fit_local_pseudotime_clock(
            full_processed,
            model_params=MODEL_PARAMS,
            n_windows=LOCAL_CLOCK_N_WINDOWS,
            random_state=int(TRAINING_CONFIG.random_state),
        )
        final_classifier = None
        final_gene_weights = pd.DataFrame(final_bundle["window_models"][0].coef_.T, index=full_processed.var_names)
        del final_gene_weights
        mean_coef = np.mean(
            np.vstack([np.asarray(clf.coef_[0], dtype=float) for clf in final_bundle["window_models"]]),
            axis=0,
        )
        final_gene_weights = mouse_core.pd.DataFrame(
            {
                "Gene": full_processed.var_names.astype(str),
                "Weight": mean_coef,
            }
        ).sort_values(by="Weight", ascending=False).reset_index(drop=True)
        model_bundle = {
            "format_version": 3,
            "model_family": "local_pseudotime_elasticnet",
            "gene_names": full_processed.var_names.astype(str).tolist(),
            "selected_threshold": float(final_bundle["selected_threshold"]),
            "metadata": {
                "model_params": mouse_core.abclock_model.default_model_params(MODEL_PARAMS),
                "analysis_mode": "single_split_direct_ablation",
                "variant_name": variant_name,
                "feature_mode": str(feature_mode),
                "use_pseudotime_covariate": False,
                "uses_interaction_features": False,
                "uses_local_pseudotime_model": True,
            },
            "local_clock_family": final_bundle,
        }
        variant_diag["local_window_summary_split"] = local_diag["window_summary"]
        variant_diag["local_window_summary_full"] = full_window_summary
        variant_diag["local_threshold_table_full"] = full_threshold_table
        variant_diag["local_threshold_table_split"] = threshold_table
    else:
        full_processed, _, _ = prepare_variant_matrices(full_processed, None, feature_mode=feature_mode)
        final_classifier = mouse_core.abclock_model.fit_elasticnet_classifier(
            full_processed,
            age_col="Age",
            model_params=MODEL_PARAMS,
        )
        final_gene_weights = mouse_core.pd.DataFrame(
            {
                "Gene": full_processed.var_names.astype(str),
                "Weight": final_classifier.coef_[0],
            }
        ).sort_values(by="Weight", ascending=False).reset_index(drop=True)

        model_bundle = mouse_core.abclock_model.build_model_bundle(
            final_classifier,
            gene_names=full_processed.var_names.astype(str).tolist(),
            selected_threshold=float(selected_threshold),
            selected_calibration="none",
            calibrator=None,
            metadata={
                "model_params": mouse_core.abclock_model.default_model_params(MODEL_PARAMS),
                "analysis_mode": "single_split_direct_ablation",
                "variant_name": variant_name,
                "feature_mode": str(feature_mode),
                "use_pseudotime_covariate": bool(variant_diag["uses_pseudotime_covariate"]),
                "uses_interaction_features": bool(variant_diag["uses_interaction_features"]),
                "uses_local_pseudotime_model": False,
            },
        )

    print(f"[4b/5] Running article holdout for `{variant_name}`")
    if uses_local_model:
        source_mask = adata_combined.obs["source"].astype(str) == str(ARTICLE_HOLDOUT_SOURCE)
        source_adata = adata_combined[source_mask].copy()
        source_donor_table = mouse_core.build_donor_table(source_adata)
        class_counts = source_donor_table["age_label"].value_counts().to_dict()
        source_holdout_result: dict[str, object] = {
            "holdout_source": str(ARTICLE_HOLDOUT_SOURCE),
            "feature_mode": str(feature_mode),
            "evaluable": False,
            "reason": None,
            "metrics_row": None,
            "selection_summary": mouse_core.pd.DataFrame(),
            "threshold_diagnostics": mouse_core.pd.DataFrame(),
            "n_holdout_cells": int(source_adata.n_obs),
            "n_holdout_donors": int(source_donor_table.shape[0]),
            "young_holdout_donors": int(class_counts.get("young", 0)),
            "old_holdout_donors": int(class_counts.get("old", 0)),
        }
        if class_counts.get("young", 0) < int(TRAINING_CONFIG.min_test_donors_per_class) or class_counts.get(
            "old", 0
        ) < int(TRAINING_CONFIG.min_test_donors_per_class):
            source_holdout_result["reason"] = "insufficient_test_donors_per_class"
        else:
            holdout_donors = set(source_donor_table["donor_id"].astype(str))
            train_mask = ~adata_combined.obs["donor_split_id"].astype(str).isin(holdout_donors)
            train_raw_holdout = adata_combined[train_mask].copy()
            test_raw_holdout = adata_combined[~train_mask].copy()
            _, train_bs_holdout = mouse_core.generate_mouse_bootstrap_metacells(
                train_raw_holdout,
                TRAINING_CONFIG,
                random_state=int(TRAINING_CONFIG.random_state) + 2000,
                rebalance=True,
            )
            _, test_bs_holdout = mouse_core.generate_mouse_bootstrap_metacells(
                test_raw_holdout,
                TRAINING_CONFIG,
                random_state=int(TRAINING_CONFIG.random_state) + 3000,
                rebalance=False,
            )
            train_hvg_holdout, test_hvg_holdout, _ = mouse_core.preprocess_mouse_metacell_pair(
                train_bs_holdout,
                test_bs_holdout,
                TRAINING_CONFIG,
            )
            train_hvg_holdout, test_hvg_holdout, _ = prepare_variant_matrices(
                train_hvg_holdout,
                test_hvg_holdout,
                feature_mode="local_pseudotime",
            )
            holdout_split_manifest = {
                "split_id": f"study_holdout::{ARTICLE_HOLDOUT_SOURCE}",
                "train_donors": sorted(train_raw_holdout.obs["donor_split_id"].astype(str).unique().tolist()),
                "test_donors": sorted(test_raw_holdout.obs["donor_split_id"].astype(str).unique().tolist()),
                "train_test_donor_overlap_count": 0,
            }
            local_bundle_holdout, donor_pred_holdout, split_holdout_df, local_holdout_diag, _ = fit_local_clock_on_processed_split(
                train_hvg_holdout,
                test_hvg_holdout,
                split_manifest=holdout_split_manifest,
                model_params=MODEL_PARAMS,
            )
            split_holdout_row = split_holdout_df.iloc[0].to_dict()
            split_holdout_row["holdout_source"] = str(ARTICLE_HOLDOUT_SOURCE)
            split_holdout_row["context_label"] = f"study_holdout::{ARTICLE_HOLDOUT_SOURCE}"
            split_holdout_row["n_holdout_cells"] = int(test_raw_holdout.n_obs)
            split_holdout_row["n_holdout_train_cells"] = int(train_raw_holdout.n_obs)
            split_holdout_row["n_holdout_train_donors"] = int(len(holdout_split_manifest["train_donors"]))
            split_holdout_row["feature_mode"] = str(feature_mode)
            source_holdout_result.update(
                {
                    "evaluable": True,
                    "metrics_row": split_holdout_row,
                    "selection_summary": mouse_core.pd.DataFrame(
                        [{"candidate_id": "fixed_main_model_local", **CANDIDATE}]
                    ),
                    "threshold_diagnostics": local_holdout_diag["threshold_table"],
                    "donor_predictions": donor_pred_holdout,
                    "local_bundle": local_bundle_holdout,
                }
            )
    else:
        source_holdout_result = evaluate_single_source_holdout(
            adata_combined,
            ARTICLE_HOLDOUT_SOURCE,
            feature_mode=feature_mode,
        )

    print(f"[5/5] Saving `{variant_name}` outputs")
    model_path = variant_dir / "final_model.joblib"
    genes_path = variant_dir / "model_genes.txt"
    weights_path = variant_dir / "gene_weights.tsv"
    metrics_path = variant_dir / "training_metrics.json"
    split_diag_path = variant_dir / "split_diagnostics.tsv"
    donor_pred_path = variant_dir / "donor_predictions.tsv"
    metacell_pred_path = variant_dir / "metacell_predictions.tsv"
    balance_diag_path = variant_dir / "class_balance_diagnostics.tsv"
    train_hvg_path = variant_dir / "adata_musc_combined_hvg.h5ad"
    threshold_diag_path = variant_dir / "threshold_diagnostics.tsv"
    local_window_summary_path = variant_dir / "local_window_summary.tsv"
    local_window_summary_full_path = variant_dir / "local_window_summary_full.tsv"
    local_gene_weights_path = variant_dir / "local_gene_weights.tsv"
    source_holdout_path = variant_dir / "source_holdout_diagnostics.tsv"
    source_holdout_selection_path = variant_dir / "source_holdout_selection.tsv"
    source_holdout_threshold_path = variant_dir / "source_holdout_thresholds.tsv"
    source_holdout_donor_pred_path = variant_dir / "source_holdout_donor_predictions.tsv"
    source_holdout_threshold_compare_path = variant_dir / "source_holdout_threshold_setting_comparison.tsv"
    source_holdout_threshold_compare_json_path = variant_dir / "source_holdout_threshold_setting_comparison.json"
    source_holdout_recal_path = variant_dir / "source_holdout_recalibrated_thresholds.tsv"
    source_holdout_threshold_plot_path = variant_dir / "source_holdout_threshold_comparison.pdf"
    interaction_score_path = variant_dir / "dynamic_interaction_gene_scores.tsv"
    interaction_panel_path = variant_dir / "dynamic_interaction_panel.tsv"

    joblib.dump(model_bundle, model_path)
    final_gene_weights.to_csv(weights_path, sep="\t", index=False)
    with open(genes_path, "w", encoding="utf-8") as f:
        for gene in full_processed.var_names.astype(str):
            f.write(f"{gene}\n")

    mouse_core.pd.DataFrame([split_row]).to_csv(split_diag_path, sep="\t", index=False)
    donor_predictions.to_csv(donor_pred_path, sep="\t", index=False)
    threshold_table.to_csv(threshold_diag_path, sep="\t", index=False)
    if metacell_predictions is not None:
        metacell_predictions.to_csv(metacell_pred_path, sep="\t", index=False)

    balance_diag = mouse_core.build_class_balance_diagnostics(
        train_raw,
        train_bs_pre,
        train_bs,
    )
    balance_diag.to_csv(balance_diag_path, sep="\t", index=False)
    mouse_core.safe_write_h5ad(full_processed, train_hvg_path)

    if bool(variant_diag["uses_interaction_features"]):
        pd.DataFrame(variant_diag["dynamic_interaction_score_table"]).to_csv(interaction_score_path, sep="\t", index=False)
        pd.DataFrame(variant_diag["dynamic_interaction_panel_table"]).to_csv(interaction_panel_path, sep="\t", index=False)
    if uses_local_model:
        pd.DataFrame(variant_diag["local_window_summary_split"]).to_csv(local_window_summary_path, sep="\t", index=False)
        pd.DataFrame(variant_diag["local_window_summary_full"]).to_csv(local_window_summary_full_path, sep="\t", index=False)
        local_coef_rows = []
        for idx, clf in enumerate(model_bundle["local_clock_family"]["window_models"]):
            center = model_bundle["local_clock_family"]["window_centers"][idx]
            for gene, weight in zip(full_processed.var_names.astype(str), np.asarray(clf.coef_[0], dtype=float)):
                local_coef_rows.append(
                    {"window_id": int(idx), "center_pseudotime": float(center), "Gene": str(gene), "Weight": float(weight)}
                )
        pd.DataFrame(local_coef_rows).to_csv(local_gene_weights_path, sep="\t", index=False)

    if source_holdout_result["evaluable"]:
        mouse_core.pd.DataFrame([source_holdout_result["metrics_row"]]).to_csv(
            source_holdout_path,
            sep="\t",
            index=False,
        )
        source_holdout_result["donor_predictions"].to_csv(source_holdout_donor_pred_path, sep="\t", index=False)
        source_holdout_result["selection_summary"].to_csv(source_holdout_selection_path, sep="\t", index=False)
        source_holdout_result["threshold_diagnostics"].to_csv(source_holdout_threshold_path, sep="\t", index=False)
        if uses_local_model:
            source_recal_threshold = None
        else:
            threshold_compare_df, recal_threshold_df, source_recal_threshold = compare_threshold_settings(
                source_holdout_result["donor_predictions"],
                train_selected_threshold=float(selected_threshold),
                context_label=f"study_holdout::{ARTICLE_HOLDOUT_SOURCE}",
            )
            threshold_compare_df.to_csv(source_holdout_threshold_compare_path, sep="\t", index=False)
            recal_threshold_df.to_csv(source_holdout_recal_path, sep="\t", index=False)
            with open(source_holdout_threshold_compare_json_path, "w", encoding="utf-8") as f:
                json.dump(threshold_compare_df.to_dict(orient="records"), f, indent=2)
            plot_source_holdout_thresholds(
                source_holdout_result["donor_predictions"],
                train_selected_threshold=float(selected_threshold),
                source_recalibrated_threshold=float(source_recal_threshold),
                output_path=source_holdout_threshold_plot_path,
            )
    else:
        source_recal_threshold = None

    uses_pseudotime = bool(variant_diag["uses_pseudotime_covariate"])
    uses_interactions = bool(variant_diag["uses_interaction_features"])
    uses_local = bool(variant_diag.get("uses_local_pseudotime_model", False))
    metrics = {
        "variant_name": variant_name,
        "feature_mode": str(feature_mode),
        "use_pseudotime_covariate": uses_pseudotime,
        "uses_interaction_features": uses_interactions,
        "uses_local_pseudotime_model": uses_local,
        "training_atlas_path": str(TRAINING_ATLAS_PATH),
        "input_training_atlas": str(ANNOTATED_ATLAS_PATH),
        "output_dir": str(variant_dir),
        "analysis_mode": "single_split_direct_ablation",
        "model_type": "elasticnet_logistic_regression",
        "bootstrap_iters": int(TRAINING_CONFIG.bootstrap_iters),
        "random_state": int(TRAINING_CONFIG.random_state),
        "split_policy": str(TRAINING_CONFIG.split_policy),
        "test_fraction": float(TRAINING_CONFIG.test_fraction),
        "n_split_repeats": 1,
        "min_test_donors_per_class": int(TRAINING_CONFIG.min_test_donors_per_class),
        "n_cells_training_atlas": int(adata_combined.n_obs),
        "n_genes_training_atlas": int(adata_combined.n_vars),
        "n_cells_with_monocle3_pseudotime": int(adata_combined.obs[PSEUDOTIME_CELL_COL].notna().sum()),
        "n_train_raw_cells": int(train_raw.n_obs),
        "n_test_raw_cells": int(test_raw.n_obs),
        "n_training_metacells": int(full_processed.n_obs),
        "n_training_genes": int(_count_base_gene_features(full_processed)),
        "n_training_features": int(full_processed.n_vars),
        "n_training_hvg": int(TRAINING_CONFIG.n_training_hvg),
        "pseudotime_feature_name": PSEUDOTIME_FEATURE_NAME if uses_pseudotime else None,
        "monocle3_cell_table_path": str(MONOCLE3_CELL_TABLE_PATH),
        "pseudotime_covariate_enabled_in_preprocess": bool(prep_diag.get("pseudotime_covariate_enabled", False)),
        "pseudotime_center": prep_diag.get("pseudotime_center") if uses_pseudotime else None,
        "pseudotime_scale": prep_diag.get("pseudotime_scale") if uses_pseudotime else None,
        "dynamic_gene_selection_method": variant_diag.get("dynamic_gene_selection_method"),
        "dynamic_top_n": variant_diag.get("dynamic_top_n"),
        "dynamic_panel_size": variant_diag.get("dynamic_panel_size"),
        "dynamic_panel_size_applied_split": len(variant_diag.get("dynamic_interaction_panel_applied_split", []))
        if uses_interactions
        else None,
        "dynamic_panel_size_applied_full": len(variant_diag.get("dynamic_interaction_panel_applied_full", []))
        if uses_interactions
        else None,
        "dynamic_panel_size_precurated": variant_diag.get("dynamic_panel_size_precurated"),
        "dynamic_curated_marker_count_requested": variant_diag.get("dynamic_curated_marker_count_requested"),
        "dynamic_curated_marker_count_present": variant_diag.get("dynamic_curated_marker_count_present"),
        "dynamic_curated_marker_count_added": variant_diag.get("dynamic_curated_marker_count_added"),
        "pseudotime_interaction_scaling": variant_diag.get("pseudotime_interaction_scaling"),
        "interaction_feature_prefix": DYNAMIC_INTERACTION_FEATURE_PREFIX if uses_interactions else None,
        "local_window_count": int(model_bundle["local_clock_family"]["n_windows"]) if uses_local else None,
        "local_window_sigma": float(model_bundle["local_clock_family"]["window_sigma"]) if uses_local else None,
        "train_auc": float(split_row["train_auc_train_only"]),
        "split_train_auc_mean": float(split_row["train_auc_train_only"]),
        "split_train_auc_std": 0.0,
        "split_balanced_accuracy_mean": float(split_row["train_metacell_balanced_accuracy_thr_opt"]),
        "split_recall_old_mean": float(split_row["holdout_donor_recall_old_thr_opt"]),
        "split_recall_young_mean": float(split_row["holdout_donor_recall_young_thr_opt"]),
        "test_auc": float(split_row["holdout_donor_auc"]),
        "test_auc_std": 0.0,
        "test_pr_auc_old_mean": float(split_row["holdout_donor_pr_auc_old"]),
        "test_balanced_accuracy_mean": float(split_row["holdout_donor_balanced_accuracy_thr_opt"]),
        "test_recall_old_mean": float(split_row["holdout_donor_recall_old_thr_opt"]),
        "test_recall_young_mean": float(split_row["holdout_donor_recall_young_thr_opt"]),
        "test_donors": list(map(str, split_manifest["test_donors"])),
        "article_holdout_source": str(ARTICLE_HOLDOUT_SOURCE),
        "article_holdout_evaluable": bool(source_holdout_result["evaluable"]),
        "article_holdout_reason": source_holdout_result["reason"],
        "article_holdout_auc": float(source_holdout_result["metrics_row"]["holdout_donor_auc"])
        if source_holdout_result["evaluable"]
        else None,
        "article_holdout_balanced_accuracy": float(
            source_holdout_result["metrics_row"]["holdout_donor_balanced_accuracy_thr_opt"]
        )
        if source_holdout_result["evaluable"]
        else None,
        "article_holdout_n_test_donors": int(source_holdout_result["metrics_row"]["n_test_donors"])
        if source_holdout_result["evaluable"]
        else int(source_holdout_result["n_holdout_donors"]),
        "article_holdout_n_test_cells": int(source_holdout_result["n_holdout_cells"]),
        "selected_threshold": float(selected_threshold),
        "selected_threshold_source": "train_only",
        "source_recalibrated_threshold": float(source_recal_threshold) if source_recal_threshold is not None else None,
        "selected_calibration": "none",
        "model_params": mouse_core.abclock_model.default_model_params(MODEL_PARAMS),
        "runtime_versions": mouse_core.collect_runtime_versions({"scanpy": sc}),
        "model_path": str(model_path),
        "model_genes_path": str(genes_path),
        "gene_weights_path": str(weights_path),
        "split_diagnostics_path": str(split_diag_path),
        "donor_predictions_path": str(donor_pred_path),
        "threshold_diagnostics_path": str(threshold_diag_path),
        "class_balance_diagnostics_path": str(balance_diag_path),
        "training_hvg_path": str(train_hvg_path),
        "metacell_predictions_path": str(metacell_pred_path) if uses_local else None,
        "local_window_summary_path": str(local_window_summary_path) if uses_local else None,
        "local_window_summary_full_path": str(local_window_summary_full_path) if uses_local else None,
        "local_gene_weights_path": str(local_gene_weights_path) if uses_local else None,
        "source_holdout_diagnostics_path": str(source_holdout_path) if source_holdout_result["evaluable"] else None,
        "source_holdout_donor_predictions_path": str(source_holdout_donor_pred_path)
        if source_holdout_result["evaluable"]
        else None,
        "source_holdout_selection_path": str(source_holdout_selection_path)
        if source_holdout_result["evaluable"]
        else None,
        "source_holdout_threshold_path": str(source_holdout_threshold_path)
        if source_holdout_result["evaluable"]
        else None,
        "source_holdout_threshold_setting_comparison_path": str(source_holdout_threshold_compare_path)
        if source_holdout_result["evaluable"]
        and not uses_local
        else None,
        "source_holdout_threshold_setting_comparison_json_path": str(source_holdout_threshold_compare_json_path)
        if source_holdout_result["evaluable"]
        and not uses_local
        else None,
        "source_holdout_recalibrated_thresholds_path": str(source_holdout_recal_path)
        if source_holdout_result["evaluable"]
        and not uses_local
        else None,
        "source_holdout_threshold_plot_path": str(source_holdout_threshold_plot_path)
        if source_holdout_result["evaluable"]
        and not uses_local
        else None,
        "dynamic_interaction_gene_scores_path": str(interaction_score_path) if uses_interactions else None,
        "dynamic_interaction_panel_path": str(interaction_panel_path) if uses_interactions else None,
        "dynamic_interaction_genes_missing_full": variant_diag.get("dynamic_interaction_genes_missing_full")
        if uses_interactions
        else None,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    summary_row = {
        "variant_name": variant_name,
        "feature_mode": str(feature_mode),
        "use_pseudotime_covariate": uses_pseudotime,
        "uses_interaction_features": uses_interactions,
        "uses_local_pseudotime_model": uses_local,
        "n_training_genes": metrics["n_training_genes"],
        "n_training_features": metrics["n_training_features"],
        "dynamic_panel_size": metrics["dynamic_panel_size"],
        "dynamic_panel_size_applied_full": metrics.get("dynamic_panel_size_applied_full"),
        "local_window_count": metrics.get("local_window_count"),
        "selected_threshold": metrics["selected_threshold"],
        "train_auc": metrics["train_auc"],
        "test_auc": metrics["test_auc"],
        "test_balanced_accuracy": metrics["test_balanced_accuracy_mean"],
        "test_recall_old": metrics["test_recall_old_mean"],
        "test_recall_young": metrics["test_recall_young_mean"],
        "article_holdout_auc": metrics["article_holdout_auc"],
        "article_holdout_balanced_accuracy": metrics["article_holdout_balanced_accuracy"],
        "output_dir": str(variant_dir),
    }
    return {"metrics": metrics, "summary_row": summary_row}


def main() -> None:
    mouse_core.validate_required_paths(
        {
            "cleaned annotated atlas": ANNOTATED_ATLAS_PATH,
            "Monocle3 cell table": MONOCLE3_CELL_TABLE_PATH,
        }
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mouse_core.step_setup_runtime()

    print("[1/4] Preparing training atlas from cleaned annotated atlas")
    prepare_training_atlas_from_cleaned_annotated(ANNOTATED_ATLAS_PATH, TRAINING_ATLAS_PATH)

    print("[2/4] Loading prepared training atlas")
    adata_combined = sc.read_h5ad(TRAINING_ATLAS_PATH)
    missing_obs = [col for col in REQUIRED_OBS_COLS if col not in adata_combined.obs.columns]
    if "counts" not in adata_combined.layers:
        raise ValueError("training atlas must contain raw counts in .layers['counts'].")
    if missing_obs:
        raise ValueError(f"training atlas missing required obs columns: {missing_obs}")
    if PSEUDOTIME_CELL_COL not in adata_combined.obs.columns:
        raise ValueError(f"training atlas missing {PSEUDOTIME_CELL_COL}.")

    print("[3/5] Splitting donors and preparing shared metacells")
    donor_table = mouse_core.build_donor_table(adata_combined)
    test_donors = mouse_core.select_test_donors_from_table(
        donor_table,
        seed=TRAINING_CONFIG.random_state,
        split_policy=TRAINING_CONFIG.split_policy,
        test_fraction=TRAINING_CONFIG.test_fraction,
        min_test_donors_per_class=TRAINING_CONFIG.min_test_donors_per_class,
        coarse_donor_ids=TRAINING_CONFIG.coarse_donor_ids,
    )
    train_raw, test_raw, split_manifest = mouse_core.split_raw_adata_by_donor(
        adata_combined,
        test_donors,
        donor_col="donor_split_id",
    )

    train_bs_pre, train_bs = mouse_core.generate_mouse_bootstrap_metacells(
        train_raw,
        TRAINING_CONFIG,
        random_state=TRAINING_CONFIG.random_state,
        rebalance=True,
    )
    _, test_bs = mouse_core.generate_mouse_bootstrap_metacells(
        test_raw,
        TRAINING_CONFIG,
        random_state=TRAINING_CONFIG.random_state + 1000,
        rebalance=False,
    )
    full_bs_pre, full_bs = mouse_core.generate_mouse_bootstrap_metacells(
        adata_combined,
        TRAINING_CONFIG,
        random_state=TRAINING_CONFIG.random_state,
        rebalance=True,
    )
    ablation_rows = []
    for variant in ABLATION_VARIANTS:
        result = run_variant_training(
            variant_name=str(variant["variant_name"]),
            feature_mode=str(variant["feature_mode"]),
            adata_combined=adata_combined,
            train_raw=train_raw,
            test_raw=test_raw,
            split_manifest=split_manifest,
            train_bs_pre=train_bs_pre,
            train_bs=train_bs,
            test_bs=test_bs,
            full_bs=full_bs,
        )
        ablation_rows.append(result["summary_row"])

    ablation_df = mouse_core.pd.DataFrame(ablation_rows)
    ablation_df.to_csv(OUTPUT_DIR / "ablation_summary.tsv", sep="\t", index=False)
    with open(OUTPUT_DIR / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(ablation_rows, f, indent=2)

    print(ablation_df.to_dict(orient="records"))


if __name__ == "__main__":
    main()
