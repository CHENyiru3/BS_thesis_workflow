from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Mapping, Sequence
import warnings

import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from scipy.sparse import issparse
from sklearn.linear_model import LogisticRegression

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import scanpy as sc

from path_config import ARTIFACT_ROOT, CISTARGET_DIR, MOUSE_TRAINING_DIR, MOUSE_VERIFICATION_DIR, PROJECT_ROOT


ROOT = PROJECT_ROOT
DATA_DIR = MOUSE_TRAINING_DIR
ABCLOCK_ROOT = Path(os.environ.get("ABCLOCK_ROOT", str(ROOT / "abclock")))

sys.path.insert(0, str(ABCLOCK_ROOT))
import metacells as abclock_metacells  # noqa: E402
import model as abclock_model  # noqa: E402
from validation import collect_runtime_versions  # noqa: E402


def _default_input_files() -> dict[str, Path]:
    return {
        "walter2024_main": DATA_DIR / "walter2024_main.h5ad",
        "GSE226907_wt": DATA_DIR / "GSE226907_wt.h5ad",
        "GSE150366_noninj": DATA_DIR / "GSE150366_noninj.h5ad",
        "SKM_mouse_raw": DATA_DIR / "SKM_mouse_raw_cells2nuclei_2022-03-30.h5ad",
        "TabulaMuris_limb_10x": DATA_DIR / "TabulaMuris_limb_10x.h5ad",
        "TabulaMuris_limb_smartseq2": DATA_DIR / "TabulaMuris_limb_smartseq2.h5ad",
        "TabulaMuris_diaphragm_smartseq2": DATA_DIR / "TabulaMuris_diaphragm_smartseq2.h5ad",
    }


@dataclass(frozen=True)
class MouseTrainingPaths:
    artifact_dir: Path = ARTIFACT_ROOT / "clock_artifacts"
    training_atlas_path: Path = ARTIFACT_ROOT / "clock_artifacts" / "training_atlas_trainonly_raw.h5ad"
    mapping_path: Path = CISTARGET_DIR / "ensembl_to_symbol_mouse.txt"
    input_files: dict[str, Path] = field(default_factory=_default_input_files)


@dataclass(frozen=True)
class MouseTrainingConfig:
    bootstrap_iters: int = 1000
    random_state: int = 42
    execution_mode: str = "train_and_evaluate"
    holdout_policy: str = "none"
    holdout_sources: tuple[str, ...] = ("walter2024_main",)
    exclude_sources: tuple[str, ...] = ()
    split_policy: str = "class_balanced_no_coarse"
    test_fraction: float = 0.2
    coarse_donor_ids: tuple[str, ...] = ("young", "old", "geriatric", "unknown", "unknown_age")
    outer_split_repeats: int = 20
    inner_split_repeats: int = 5
    n_split_repeats: int = 20
    balance_mode: str = "donor_stratified"
    target_class_ratio: float = 1.0
    min_test_donors_per_class: int = 3
    threshold_objective: str = "balanced_accuracy"
    n_training_hvg: int = 1500
    n_pyscenic_hvg: int = 5000
    model_c: float = 0.1
    model_l1_ratio: float = 0.5
    candidate_c: tuple[float, ...] = (0.03, 0.1, 0.3)
    candidate_l1_ratio: tuple[float, ...] = (0.1, 0.5, 0.9)
    candidate_calibration: tuple[str, ...] = ("none", "sigmoid")
    study_holdout_enabled: bool = True
    artifact_mode: str = "full"


@dataclass(frozen=True)
class MousePostPaths:
    artifact_dir: Path = ARTIFACT_ROOT / "clock_artifacts"
    outdir: Path = ARTIFACT_ROOT / "post_results"
    verification_input: Path = MOUSE_VERIFICATION_DIR / "Myo_Aged_SkM_mm10_v1-1_MuSC.h5ad"
    db_root: Path = CISTARGET_DIR / "databases" / "mouse"
    motif_anno_path: Path = CISTARGET_DIR / "motif2tf" / "motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl"
    tf_list_path: Path = CISTARGET_DIR / "tf_lists" / "allTFs_mm.txt"
    gmt_path: Path = CISTARGET_DIR / "m5.go.bp.v2024.1.Mm.symbols.gmt"


@dataclass(frozen=True)
class MousePostConfig:
    grn_num_workers: int = 8


TRAINING_PATHS = MouseTrainingPaths()
TRAINING_CONFIG = MouseTrainingConfig()
POST_PATHS = MousePostPaths()
POST_CONFIG = MousePostConfig()

PSEUDOTIME_CELL_COL = "pseudotime_monocle3"
PSEUDOTIME_METACELL_COL = f"{PSEUDOTIME_CELL_COL}_mean"
PSEUDOTIME_FEATURE_NAME = "__cov_pseudotime_monocle3__"


def make_mouse_training_paths(
    artifact_dir: Path | None = None,
    **overrides: Any,
) -> MouseTrainingPaths:
    paths = TRAINING_PATHS
    if artifact_dir is not None:
        paths = replace(
            paths,
            artifact_dir=Path(artifact_dir),
            training_atlas_path=Path(artifact_dir) / "training_atlas_trainonly_raw.h5ad",
        )
    if overrides:
        normalized = {
            key: {name: Path(value) for name, value in value.items()} if key == "input_files" else Path(value)
            if key.endswith("_path") or key.endswith("_dir")
            else value
            for key, value in overrides.items()
        }
        paths = replace(paths, **normalized)
    return paths


def make_mouse_post_paths(
    artifact_dir: Path | None = None,
    outdir: Path | None = None,
    verification_input: Path | None = None,
    **overrides: Any,
) -> MousePostPaths:
    paths = POST_PATHS
    if artifact_dir is not None:
        paths = replace(paths, artifact_dir=Path(artifact_dir))
    if outdir is not None:
        paths = replace(paths, outdir=Path(outdir))
    if verification_input is not None:
        paths = replace(paths, verification_input=Path(verification_input))
    if overrides:
        normalized = {
            key: Path(value) if key.endswith("_path") or key.endswith("_dir") or key == "verification_input" else value
            for key, value in overrides.items()
        }
        paths = replace(paths, **normalized)
    return paths


def validate_required_paths(required_paths: Mapping[str, Path]) -> None:
    missing = [f"{label}: {path}" for label, path in required_paths.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n- " + "\n- ".join(missing))


def identify_gene_type(genes: pd.Index) -> str:
    sample = list(genes[:100])
    ensembl_count = sum(1 for gene in sample if str(gene).startswith("ENS"))
    symbol_count = len(sample) - ensembl_count
    return "Ensembl ID" if ensembl_count > symbol_count else "Gene Symbol"


def parse_tabulamuris_age(age_str: object) -> float | None:
    if pd.isna(age_str):
        return None
    months = re.findall(r"(\d+)m", str(age_str))
    return float(int(months[0])) if months else None


def parse_skm_age_bin(age_bin: object) -> float | None:
    if pd.isna(age_bin):
        return None
    s = str(age_bin).lower()
    if s == "young":
        return 6.0
    if s == "old":
        return 24.0
    return None


def _first_present_column(obs: pd.DataFrame, candidates: Sequence[str], default: str) -> pd.Series:
    for col in candidates:
        if col in obs.columns:
            return obs[col].astype(str)
    return pd.Series(default, index=obs.index, dtype=object)


def filter_musc(adata: sc.AnnData, source_name: str) -> sc.AnnData:
    if source_name in {"walter2024_main", "GSE150366_noninj"}:
        mask = adata.obs["celltype"].str.contains("MuSC|Myoblast" if source_name == "walter2024_main" else "MuSC", na=False)
    elif source_name == "GSE226907_wt":
        mask = adata.obs["celltype"].eq("MuSC")
    elif source_name == "SKM_mouse_raw":
        mask = adata.obs["annotation"].eq("MuSC")
    elif "TabulaMuris" in source_name:
        mask = adata.obs["cell_type"].eq("skeletal muscle satellite cell")
    else:
        mask = pd.Series(False, index=adata.obs.index, dtype=bool)

    filtered = adata[mask].copy()
    print(f"  {source_name}: {mask.sum()} / {adata.n_obs} MuSC cells")
    return filtered


def standardize_obs(adata: sc.AnnData, source_name: str) -> sc.AnnData:
    obs = adata.obs.copy()
    obs["source"] = source_name

    obs["sample_id_std"] = _first_present_column(obs, ["sample_id", "MouseID", "donor_id", "SampleID"], default=source_name)
    obs["Sex_std"] = _first_present_column(obs, ["Sex", "sex"], default="unknown")
    obs["celltype_std"] = _first_present_column(obs, ["celltype", "annotation", "cell_type"], default="MuSC")

    obs["Age_group_std"] = obs["Age_group"]

    cols = ["sample_id_std", "Sex_std", "celltype_std", "Age_group_std", "source"]
    adata.obs = obs[[c for c in cols if c in obs.columns]].copy()
    return adata


def sanitize_anndata_for_h5ad_write(adata: sc.AnnData) -> sc.AnnData:
    adata.obs.index = pd.Index(adata.obs.index.astype(str), dtype=object)
    adata.var.index = pd.Index(adata.var.index.astype(str), dtype=object)

    for frame in (adata.obs, adata.var):
        for col in frame.columns:
            series = frame[col]
            dtype_str = str(series.dtype)
            if isinstance(series.dtype, pd.CategoricalDtype):
                frame[col] = series.astype(str).astype(object)
                continue
            if dtype_str == "str" or dtype_str == "string" or dtype_str.startswith("string["):
                frame[col] = series.astype(str).astype(object)
    return adata


def _find_string_extension_cols(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        dtype_str = str(frame[col].dtype)
        if dtype_str == "str" or dtype_str == "string" or dtype_str.startswith("string["):
            cols.append(str(col))
    return cols


def safe_write_h5ad(adata: sc.AnnData, path: Path) -> None:
    sanitize_anndata_for_h5ad_write(adata)
    obs_string_cols = _find_string_extension_cols(adata.obs)
    var_string_cols = _find_string_extension_cols(adata.var)
    if obs_string_cols or var_string_cols:
        raise RuntimeError(
            "String extension dtypes remain after sanitize; refusing write. "
            f"obs={obs_string_cols}, var={var_string_cols}"
        )
    try:
        adata.write_h5ad(path, convert_strings_to_categoricals=False)
    except TypeError:
        adata.write_h5ad(path)


def concat_on_common_genes(adatas: List[sc.AnnData]) -> sc.AnnData:
    all_gene_sets = [set(adata.var_names) for adata in adatas]
    common_genes = set.intersection(*all_gene_sets)
    print(f"  Common genes across selected datasets: {len(common_genes)}")
    filtered = [adata[:, adata.var_names.isin(common_genes)].copy() for adata in adatas]
    return sc.concat(filtered, index_unique="-", join="outer")


def add_linkage_aware_donor_ids(adata: sc.AnnData) -> tuple[sc.AnnData, pd.DataFrame]:
    obs = adata.obs.copy()
    obs["sample_id_std"] = obs["sample_id_std"].astype(str)
    obs["source"] = obs["source"].astype(str)

    source_counts = obs.groupby("sample_id_std")["source"].nunique()
    linked_ids = set(source_counts[source_counts > 1].index.astype(str))

    obs["donor_bootstrap_id"] = obs["source"] + "::" + obs["sample_id_std"]
    obs["donor_split_id"] = np.where(
        obs["sample_id_std"].isin(linked_ids),
        "LINK::" + obs["sample_id_std"],
        obs["donor_bootstrap_id"],
    )
    adata.obs = obs

    linkage_map = (
        obs.groupby("sample_id_std")
        .agg(
            n_sources=("source", "nunique"),
            sources=("source", lambda s: "|".join(sorted(set(map(str, s))))),
            n_cells=("source", "size"),
        )
        .reset_index()
    )
    linkage_map["is_linked_across_sources"] = linkage_map["n_sources"] > 1
    return adata, linkage_map


def dense_matrix(X: Any) -> np.ndarray:
    if issparse(X):
        return X.toarray()
    return np.asarray(X)


def _append_feature_column(adata: sc.AnnData, values: np.ndarray, feature_name: str) -> sc.AnnData:
    feature_values = np.asarray(values, dtype=float).reshape(-1, 1)
    if feature_values.shape[0] != adata.n_obs:
        raise ValueError(f"Feature {feature_name} has {feature_values.shape[0]} rows; expected {adata.n_obs}.")

    if issparse(adata.X):
        new_x = sparse.hstack([adata.X, sparse.csr_matrix(feature_values)], format="csr")
    else:
        new_x = np.hstack([np.asarray(adata.X), feature_values])

    new_var = adata.var.copy()
    feature_row = {}
    for col, dtype in new_var.dtypes.items():
        if pd.api.types.is_bool_dtype(dtype):
            feature_row[col] = False
        elif pd.api.types.is_numeric_dtype(dtype):
            feature_row[col] = 0.0
        else:
            feature_row[col] = ""
    new_var.loc[feature_name] = pd.Series(feature_row)

    new_adata = sc.AnnData(X=new_x, obs=adata.obs.copy(), var=new_var)
    for key, value in adata.uns.items():
        new_adata.uns[key] = value
    for key, value in adata.obsm.items():
        new_adata.obsm[key] = value
    for key, value in adata.varm.items():
        new_adata.varm[key] = value
    for key, value in adata.layers.items():
        new_adata.layers[key] = value
    return new_adata


def _append_standardized_pseudotime_feature(
    train_adata: sc.AnnData,
    test_adata: sc.AnnData | None = None,
) -> tuple[sc.AnnData, sc.AnnData | None, dict[str, float] | None]:
    if PSEUDOTIME_METACELL_COL not in train_adata.obs.columns:
        return train_adata, test_adata, None
    if test_adata is not None and PSEUDOTIME_METACELL_COL not in test_adata.obs.columns:
        raise ValueError(f"Missing {PSEUDOTIME_METACELL_COL} in test metacells.")

    train_values = pd.to_numeric(train_adata.obs[PSEUDOTIME_METACELL_COL], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(train_values).all():
        raise ValueError("Training metacells contain non-finite Monocle3 pseudotime values.")

    center = float(train_values.mean())
    scale = float(train_values.std(ddof=0))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0

    train_std = (train_values - center) / scale
    train_adata.obs["pseudotime_std"] = train_std
    train_adata = _append_feature_column(train_adata, train_std, PSEUDOTIME_FEATURE_NAME)

    if test_adata is not None:
        test_values = pd.to_numeric(test_adata.obs[PSEUDOTIME_METACELL_COL], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(test_values).all():
            raise ValueError("Test metacells contain non-finite Monocle3 pseudotime values.")
        test_std = (test_values - center) / scale
        test_adata.obs["pseudotime_std"] = test_std
        test_adata = _append_feature_column(test_adata, test_std, PSEUDOTIME_FEATURE_NAME)

    return train_adata, test_adata, {"pseudotime_center": center, "pseudotime_scale": scale}


def _stratified_downsample_indices(
    donor_values: pd.Series,
    keep_n: int,
    random_state: int,
) -> np.ndarray:
    if keep_n >= len(donor_values):
        return donor_values.index.to_numpy()

    rng = np.random.RandomState(random_state)
    counts = donor_values.value_counts()
    total = int(counts.sum())

    expected = counts * (float(keep_n) / float(total))
    base = expected.astype(int).clip(lower=0)
    remainder = int(keep_n - base.sum())

    if remainder > 0:
        frac = (expected - base).sort_values(ascending=False)
        for donor in frac.index[:remainder]:
            if base[donor] < counts[donor]:
                base[donor] += 1

    picked = []
    for donor, n_keep in base.items():
        if n_keep <= 0:
            continue
        donor_idx = donor_values[donor_values == donor].index.to_numpy()
        take = min(int(n_keep), len(donor_idx))
        sel = rng.choice(donor_idx, size=take, replace=False)
        picked.extend(sel.tolist())

    if len(picked) < keep_n:
        left = np.setdiff1d(donor_values.index.to_numpy(), np.asarray(picked), assume_unique=False)
        extra = rng.choice(left, size=min(keep_n - len(picked), len(left)), replace=False)
        picked.extend(extra.tolist())

    return np.asarray(picked[:keep_n])


def rebalance_metacells_by_donor(
    adata_bs: sc.AnnData,
    target_old_to_young: float = 1.0,
    age_col: str = "Age",
    donor_col: str = "donor_split_id",
    random_state: int = 42,
) -> sc.AnnData:
    if target_old_to_young <= 0:
        raise ValueError("target_old_to_young must be > 0")

    age = adata_bs.obs[age_col].astype(str)
    idx_old = np.where(age.values == "old")[0]
    idx_young = np.where(age.values == "young")[0]
    n_old, n_young = len(idx_old), len(idx_young)
    if n_old == 0 or n_young == 0:
        return adata_bs

    current_ratio = float(n_old) / float(n_young)
    if current_ratio > target_old_to_young:
        keep_old = max(1, min(n_old, int(round(n_young * target_old_to_young))))
        donor_values = pd.Series(adata_bs.obs.iloc[idx_old][donor_col].astype(str).values, index=idx_old)
        keep_old_idx = _stratified_downsample_indices(donor_values, keep_old, random_state)
        keep_idx = np.concatenate([idx_young, keep_old_idx])
    else:
        keep_young = max(1, min(n_young, int(round(n_old / target_old_to_young))))
        donor_values = pd.Series(adata_bs.obs.iloc[idx_young][donor_col].astype(str).values, index=idx_young)
        keep_young_idx = _stratified_downsample_indices(donor_values, keep_young, random_state)
        keep_idx = np.concatenate([idx_old, keep_young_idx])

    keep_idx = np.sort(keep_idx)
    return adata_bs[keep_idx].copy()


def build_class_balance_diagnostics(
    raw_adata: sc.AnnData,
    metacell_pre: sc.AnnData,
    metacell_post: sc.AnnData,
) -> pd.DataFrame:
    rows = []
    for stage_name, adata, age_col in [
        ("raw", raw_adata, "Age_group_std"),
        ("metacell_pre_balance", metacell_pre, "Age"),
        ("metacell_post_balance", metacell_post, "Age"),
    ]:
        tmp = adata.obs.copy()
        tmp["source"] = tmp.get("source", "unknown").astype(str)
        tmp[age_col] = tmp[age_col].astype(str)

        by_source = (
            tmp.groupby(["source", age_col]).size().reset_index(name="n_cells")
            .rename(columns={age_col: "age"})
        )
        by_source["stage"] = stage_name
        by_source["slice"] = "source_age"
        rows.append(by_source)

        by_age = tmp.groupby(age_col).size().reset_index(name="n_cells").rename(columns={age_col: "age"})
        by_age["source"] = "ALL"
        by_age["stage"] = stage_name
        by_age["slice"] = "global_age"
        rows.append(by_age[["stage", "slice", "source", "age", "n_cells"]])

    out = pd.concat(rows, ignore_index=True)
    return out[["stage", "slice", "source", "age", "n_cells"]]


def resolve_outer_split_repeats(config: MouseTrainingConfig) -> int:
    if int(config.outer_split_repeats) > 0:
        return int(config.outer_split_repeats)
    return max(1, int(config.n_split_repeats))


def candidate_grid(config: MouseTrainingConfig) -> list[dict[str, Any]]:
    c_values = tuple(float(value) for value in config.candidate_c) or (float(config.model_c),)
    l1_values = tuple(float(value) for value in config.candidate_l1_ratio) or (float(config.model_l1_ratio),)
    calibration_values = tuple(str(value) for value in config.candidate_calibration) or ("none",)

    grid: list[dict[str, Any]] = []
    for c_value in c_values:
        for l1_value in l1_values:
            for calibration in calibration_values:
                grid.append(
                    {
                        "candidate_id": f"C={c_value:g}|l1={l1_value:g}|cal={calibration}",
                        "C": float(c_value),
                        "l1_ratio": float(l1_value),
                        "calibration": str(calibration),
                    }
                )
    return grid


def build_donor_table(
    adata: sc.AnnData,
    donor_col: str = "donor_split_id",
    age_col: str = "Age_group_std",
    source_col: str = "source",
) -> pd.DataFrame:
    obs = adata.obs[[donor_col, age_col, source_col]].copy()
    obs[donor_col] = obs[donor_col].astype(str)
    obs[age_col] = obs[age_col].astype(str)
    obs[source_col] = obs[source_col].astype(str)

    donor_table = (
        obs.groupby(donor_col)
        .agg(
            age_labels=(age_col, lambda s: sorted(set(map(str, s)))),
            sources=(source_col, lambda s: sorted(set(map(str, s)))),
            n_cells=(source_col, "size"),
        )
        .reset_index()
        .rename(columns={donor_col: "donor_id"})
    )
    donor_table["age_label"] = donor_table["age_labels"].apply(
        lambda values: values[0] if len(values) == 1 and values[0] in {"young", "old"} else "mixed"
    )
    donor_table["n_sources"] = donor_table["sources"].apply(len)
    donor_table["source_label"] = donor_table["sources"].apply(lambda values: "|".join(values))
    return donor_table


def donor_key_for_coarse(donor_value: object) -> str:
    s = str(donor_value)
    if s.startswith("LINK::"):
        return s.split("LINK::", 1)[1].lower()
    if "::" in s:
        return s.split("::", 1)[1].lower()
    return s.lower()


def select_test_donors_from_table(
    donor_table: pd.DataFrame,
    seed: int,
    split_policy: str,
    test_fraction: float,
    min_test_donors_per_class: int,
    coarse_donor_ids: Sequence[str],
) -> np.ndarray:
    rng = np.random.RandomState(int(seed))
    coarse_set = {str(x).strip().lower() for x in coarse_donor_ids}
    eligible = donor_table.copy()
    eligible["donor_id"] = eligible["donor_id"].astype(str)

    if split_policy == "class_balanced_no_coarse":
        young_pool = eligible[
            (eligible["age_label"] == "young")
            & (~eligible["donor_id"].map(donor_key_for_coarse).isin(coarse_set))
        ]["donor_id"].tolist()
        old_pool = eligible[
            (eligible["age_label"] == "old")
            & (~eligible["donor_id"].map(donor_key_for_coarse).isin(coarse_set))
        ]["donor_id"].tolist()
        if not young_pool or not old_pool:
            raise ValueError("No valid donors for class_balanced_no_coarse split.")

        n_young = max(int(min_test_donors_per_class), int(len(young_pool) * float(test_fraction)), 1)
        n_old = max(int(min_test_donors_per_class), int(len(old_pool) * float(test_fraction)), 1)
        test_young = rng.choice(young_pool, min(n_young, len(young_pool)), replace=False)
        test_old = rng.choice(old_pool, min(n_old, len(old_pool)), replace=False)
        return np.asarray(list(test_young) + list(test_old), dtype=object)

    if split_policy == "random20_all":
        donors = eligible["donor_id"].astype(str).tolist()
        n_test = max(1, int(len(donors) * float(test_fraction)))
        return rng.choice(donors, min(n_test, len(donors)), replace=False)

    raise ValueError(f"Unknown split_policy: {split_policy}")


def split_raw_adata_by_donor(
    adata: sc.AnnData,
    test_donors: Sequence[str],
    donor_col: str = "donor_split_id",
) -> tuple[sc.AnnData, sc.AnnData, dict[str, Any]]:
    donor_series = adata.obs[donor_col].astype(str)
    test_set = set(map(str, test_donors))
    test_mask = donor_series.isin(test_set)
    train_adata = adata[~test_mask].copy()
    test_adata = adata[test_mask].copy()

    train_donors = sorted(train_adata.obs[donor_col].astype(str).unique().tolist())
    test_donors_sorted = sorted(test_adata.obs[donor_col].astype(str).unique().tolist())
    manifest = {
        "train_donors": train_donors,
        "test_donors": test_donors_sorted,
        "train_test_donor_overlap_count": int(len(set(train_donors).intersection(test_donors_sorted))),
    }
    return train_adata, test_adata, manifest


def gene_count_mask(adata: sc.AnnData, min_counts: int) -> np.ndarray:
    X = adata.X
    if issparse(X):
        counts = np.asarray(X.sum(axis=0)).ravel()
    else:
        counts = np.asarray(X.sum(axis=0)).ravel()
    return counts >= int(min_counts)


def generate_mouse_bootstrap_metacells(
    adata: sc.AnnData,
    config: MouseTrainingConfig,
    random_state: int,
    rebalance: bool,
) -> tuple[sc.AnnData, sc.AnnData]:
    aggregate_numeric_meta_cols: list[str] = []
    if PSEUDOTIME_CELL_COL in adata.obs.columns:
        aggregate_numeric_meta_cols.append(PSEUDOTIME_CELL_COL)

    adata_bs = abclock_metacells.generate_bootstrap_cells(
        adata,
        n_cells_per_bin=15,
        n_iter=int(config.bootstrap_iters),
        donor_col="donor_bootstrap_id",
        age_col="Age_group_std",
        state_col="celltype_std",
        extra_meta_cols=[
            "Sex_std",
            "celltype_std",
            "Age_group_std",
            "source",
            "sample_id_std",
            "donor_bootstrap_id",
            "donor_split_id",
        ],
        aggregate_numeric_meta_cols=aggregate_numeric_meta_cols,
        balance_classes=True,
        n_iter_minority=int(config.bootstrap_iters),
        random_state=int(random_state),
    )
    pre_balance = adata_bs.copy()
    if rebalance and config.balance_mode == "donor_stratified":
        adata_bs = rebalance_metacells_by_donor(
            adata_bs,
            target_old_to_young=float(config.target_class_ratio),
            age_col="Age",
            donor_col="donor_split_id",
            random_state=int(random_state),
        )
    return pre_balance, adata_bs


def preprocess_mouse_metacell_pair(
    train_bs: sc.AnnData,
    test_bs: sc.AnnData,
    config: MouseTrainingConfig,
) -> tuple[sc.AnnData, sc.AnnData, dict[str, Any]]:
    train_work = train_bs.copy()
    test_work = test_bs.copy()

    keep_mask = gene_count_mask(train_work, min_counts=50)
    keep_genes = train_work.var_names[keep_mask]
    keep_genes = pd.Index([gene for gene in keep_genes if not str(gene).lower().startswith("mt-")])

    train_work = train_work[:, keep_genes].copy()
    test_work = test_work[:, keep_genes].copy()

    sc.pp.normalize_total(train_work, target_sum=1e4)
    sc.pp.log1p(train_work)
    sc.pp.normalize_total(test_work, target_sum=1e4)
    sc.pp.log1p(test_work)

    sc.pp.highly_variable_genes(train_work, n_top_genes=int(config.n_training_hvg), subset=True)
    selected_genes = train_work.var_names.astype(str).tolist()
    test_hvg = test_work[:, selected_genes].copy()
    train_work, test_hvg, pseudotime_diag = _append_standardized_pseudotime_feature(train_work, test_hvg)

    diagnostics = {
        "n_genes_after_count_filter": int(len(keep_genes)),
        "n_training_genes": int(len(selected_genes)),
        "n_training_features": int(train_work.n_vars),
        "training_gene_list": selected_genes,
        "pseudotime_covariate_enabled": bool(pseudotime_diag is not None),
    }
    if pseudotime_diag is not None:
        diagnostics.update(pseudotime_diag)
    return train_work, test_hvg, diagnostics


def preprocess_mouse_full_metacells(
    full_bs: sc.AnnData,
    config: MouseTrainingConfig,
) -> tuple[sc.AnnData, sc.AnnData | None, dict[str, Any]]:
    full_work = full_bs.copy()
    keep_mask = gene_count_mask(full_work, min_counts=50)
    keep_genes = full_work.var_names[keep_mask]
    keep_genes = pd.Index([gene for gene in keep_genes if not str(gene).lower().startswith("mt-")])
    full_work = full_work[:, keep_genes].copy()

    sc.pp.normalize_total(full_work, target_sum=1e4)
    sc.pp.log1p(full_work)

    full_pyscenic = full_work.copy() if config.artifact_mode == "full" else None
    sc.pp.highly_variable_genes(full_work, n_top_genes=int(config.n_training_hvg), subset=True)
    if full_pyscenic is not None:
        sc.pp.highly_variable_genes(full_pyscenic, n_top_genes=int(config.n_pyscenic_hvg), subset=True)
    full_work, _, pseudotime_diag = _append_standardized_pseudotime_feature(full_work, None)

    diagnostics = {
        "n_genes_after_count_filter": int(len(keep_genes)),
        "training_gene_list": [gene for gene in full_work.var_names.astype(str).tolist() if gene != PSEUDOTIME_FEATURE_NAME],
        "n_training_features": int(full_work.n_vars),
        "pseudotime_covariate_enabled": bool(pseudotime_diag is not None),
    }
    if pseudotime_diag is not None:
        diagnostics.update(pseudotime_diag)
    return full_work, full_pyscenic, diagnostics


def summarize_mouse_donor_predictions(
    adata: sc.AnnData,
    y_prob: np.ndarray,
    selected_threshold: float | None = None,
    extra_group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    extra_cols = list(extra_group_cols or [])
    donor_predictions = abclock_model.summarize_group_predictions(
        adata.obs,
        y_prob,
        group_col="donor_split_id",
        age_col="Age",
        extra_group_cols=extra_cols,
        selected_threshold=selected_threshold,
    )
    donor_predictions = donor_predictions.rename(columns={"group_id": "donor_id"})
    return donor_predictions


def fit_mouse_candidate_on_processed_split(
    train_hvg: sc.AnnData,
    test_hvg: sc.AnnData,
    candidate: Mapping[str, Any],
    selected_threshold: float,
    calibrator: LogisticRegression | None,
    split_manifest: Mapping[str, Any],
    threshold_source: str,
) -> tuple[LogisticRegression, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    clf = abclock_model.fit_elasticnet_classifier(
        train_hvg,
        age_col="Age",
        model_params={"C": float(candidate["C"]), "l1_ratio": float(candidate["l1_ratio"])},
    )

    train_prob_raw = clf.predict_proba(dense_matrix(train_hvg.X))[:, 1]
    test_prob_raw = clf.predict_proba(dense_matrix(test_hvg.X))[:, 1]
    train_prob = abclock_model.apply_calibration(calibrator, train_prob_raw)
    test_prob = abclock_model.apply_calibration(calibrator, test_prob_raw)

    train_y = (train_hvg.obs["Age"].astype(str) == "old").astype(int).to_numpy()
    test_y = (test_hvg.obs["Age"].astype(str) == "old").astype(int).to_numpy()
    train_metrics_default = abclock_model.evaluate_binary_metrics(train_y, train_prob, threshold=0.5)
    train_metrics_opt = abclock_model.evaluate_binary_metrics(train_y, train_prob, threshold=selected_threshold)
    test_metrics_default = abclock_model.evaluate_binary_metrics(test_y, test_prob, threshold=0.5)
    test_metrics_opt = abclock_model.evaluate_binary_metrics(test_y, test_prob, threshold=selected_threshold)

    donor_predictions = summarize_mouse_donor_predictions(
        test_hvg,
        test_prob,
        selected_threshold=selected_threshold,
        extra_group_cols=["source"],
    )
    donor_y = (donor_predictions["age_group"].astype(str) == "old").astype(int).to_numpy()
    donor_prob = donor_predictions["mean_p_old"].to_numpy(dtype=float)
    donor_metrics_default = abclock_model.evaluate_binary_metrics(donor_y, donor_prob, threshold=0.5)
    donor_metrics_opt = abclock_model.evaluate_binary_metrics(donor_y, donor_prob, threshold=selected_threshold)

    split_row = {
        "selected_model_c": float(candidate["C"]),
        "selected_model_l1_ratio": float(candidate["l1_ratio"]),
        "selected_calibration": str(candidate["calibration"]),
        "selected_threshold": float(selected_threshold),
        "selected_threshold_source": str(threshold_source),
        "n_train_donors": int(len(split_manifest["train_donors"])),
        "n_test_donors": int(len(split_manifest["test_donors"])),
        "train_test_donor_overlap_count": int(split_manifest["train_test_donor_overlap_count"]),
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
        "holdout_metacell_pr_auc_old": float(test_metrics_default["pr_auc_old"]),
        "holdout_donor_auc": float(donor_metrics_default["auc"]),
        "holdout_donor_balanced_accuracy_thr_0p5": float(donor_metrics_default["balanced_accuracy"]),
        "holdout_donor_balanced_accuracy_thr_opt": float(donor_metrics_opt["balanced_accuracy"]),
        "holdout_donor_recall_old_thr_opt": float(donor_metrics_opt["recall_old"]),
        "holdout_donor_recall_young_thr_opt": float(donor_metrics_opt["recall_young"]),
        "holdout_donor_pr_auc_old": float(donor_metrics_default["pr_auc_old"]),
        "test_donors": "|".join(map(str, split_manifest["test_donors"])),
    }

    donor_predictions["selected_model_c"] = float(candidate["C"])
    donor_predictions["selected_model_l1_ratio"] = float(candidate["l1_ratio"])
    donor_predictions["selected_calibration"] = str(candidate["calibration"])
    donor_predictions["selected_threshold_source"] = str(threshold_source)
    return clf, donor_predictions, split_row, pd.DataFrame(
        {"Gene": train_hvg.var_names.astype(str), "Weight": clf.coef_[0]}
    ).sort_values(by="Weight", ascending=False).reset_index(drop=True)


def select_best_mouse_candidate(
    raw_adata: sc.AnnData,
    config: MouseTrainingConfig,
    seed_base: int,
    context_label: str,
) -> dict[str, Any]:
    candidates = candidate_grid(config)
    donor_table = build_donor_table(raw_adata)
    if donor_table.empty:
        raise ValueError(f"No donors available for selection context {context_label}.")

    candidate_prediction_frames: dict[str, list[pd.DataFrame]] = {item["candidate_id"]: [] for item in candidates}
    candidate_repeat_rows: list[dict[str, Any]] = []

    for repeat_idx in range(max(1, int(config.inner_split_repeats))):
        test_donors = select_test_donors_from_table(
            donor_table,
            seed=int(seed_base) + repeat_idx,
            split_policy=config.split_policy,
            test_fraction=float(config.test_fraction),
            min_test_donors_per_class=int(config.min_test_donors_per_class),
            coarse_donor_ids=config.coarse_donor_ids,
        )
        inner_train_raw, inner_val_raw, split_manifest = split_raw_adata_by_donor(raw_adata, test_donors)
        split_manifest["context_label"] = context_label
        split_manifest["repeat_idx"] = int(repeat_idx)

        _, inner_train_bs = generate_mouse_bootstrap_metacells(
            inner_train_raw,
            config,
            random_state=int(seed_base) + repeat_idx,
            rebalance=True,
        )
        _, inner_val_bs = generate_mouse_bootstrap_metacells(
            inner_val_raw,
            config,
            random_state=int(seed_base) + 1000 + repeat_idx,
            rebalance=False,
        )
        inner_train_hvg, inner_val_hvg, prep_diag = preprocess_mouse_metacell_pair(inner_train_bs, inner_val_bs, config)

        for candidate in candidates:
            clf = abclock_model.fit_elasticnet_classifier(
                inner_train_hvg,
                age_col="Age",
                model_params={"C": float(candidate["C"]), "l1_ratio": float(candidate["l1_ratio"])},
            )
            val_prob_raw = clf.predict_proba(dense_matrix(inner_val_hvg.X))[:, 1]
            donor_pred = summarize_mouse_donor_predictions(inner_val_hvg, val_prob_raw, extra_group_cols=["source"])
            donor_pred["context_label"] = context_label
            donor_pred["repeat_idx"] = int(repeat_idx)
            donor_pred["candidate_id"] = str(candidate["candidate_id"])
            donor_pred["model_c"] = float(candidate["C"])
            donor_pred["model_l1_ratio"] = float(candidate["l1_ratio"])
            donor_pred["calibration"] = str(candidate["calibration"])
            donor_pred["raw_mean_p_old"] = donor_pred["mean_p_old"].astype(float)
            candidate_prediction_frames[str(candidate["candidate_id"])].append(donor_pred)

            donor_y = (donor_pred["age_group"].astype(str) == "old").astype(int).to_numpy()
            donor_prob = donor_pred["mean_p_old"].to_numpy(dtype=float)
            donor_metrics = abclock_model.evaluate_binary_metrics(donor_y, donor_prob, threshold=0.5)
            candidate_repeat_rows.append(
                {
                    "context_label": context_label,
                    "repeat_idx": int(repeat_idx),
                    "candidate_id": str(candidate["candidate_id"]),
                    "model_c": float(candidate["C"]),
                    "model_l1_ratio": float(candidate["l1_ratio"]),
                    "calibration": str(candidate["calibration"]),
                    "inner_repeat_donor_auc": float(donor_metrics["auc"]),
                    "inner_repeat_donor_balanced_accuracy_thr_0p5": float(donor_metrics["balanced_accuracy"]),
                    "n_val_donors": int(donor_pred.shape[0]),
                    "n_training_genes": int(prep_diag["n_training_genes"]),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    threshold_frames: list[pd.DataFrame] = []
    selection_predictions: dict[str, pd.DataFrame] = {}
    calibrators: dict[str, LogisticRegression | None] = {}

    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        oof = pd.concat(candidate_prediction_frames[candidate_id], ignore_index=True)
        y_true = (oof["age_group"].astype(str) == "old").astype(int).to_numpy()
        raw_prob = oof["raw_mean_p_old"].to_numpy(dtype=float)

        calibrator = None
        calibrated_prob = raw_prob
        if str(candidate["calibration"]) == "sigmoid":
            calibrator = abclock_model.fit_sigmoid_calibrator(y_true, raw_prob, random_state=int(seed_base))
            calibrated_prob = abclock_model.apply_calibration(calibrator, raw_prob)
        calibrators[candidate_id] = calibrator

        selected_threshold, threshold_table = abclock_model.select_optimal_threshold(
            y_true,
            calibrated_prob,
            objective=config.threshold_objective,
        )
        donor_metrics_default = abclock_model.evaluate_binary_metrics(y_true, calibrated_prob, threshold=0.5)
        donor_metrics_opt = abclock_model.evaluate_binary_metrics(y_true, calibrated_prob, threshold=selected_threshold)

        threshold_table["context_label"] = context_label
        threshold_table["candidate_id"] = candidate_id
        threshold_table["model_c"] = float(candidate["C"])
        threshold_table["model_l1_ratio"] = float(candidate["l1_ratio"])
        threshold_table["calibration"] = str(candidate["calibration"])
        threshold_frames.append(threshold_table)

        oof["calibrated_mean_p_old"] = calibrated_prob
        oof["selected_threshold"] = float(selected_threshold)
        oof["pred_old_thr_opt"] = (oof["calibrated_mean_p_old"] >= float(selected_threshold)).astype(int)
        selection_predictions[candidate_id] = oof

        summary_rows.append(
            {
                "context_label": context_label,
                "candidate_id": candidate_id,
                "model_c": float(candidate["C"]),
                "model_l1_ratio": float(candidate["l1_ratio"]),
                "calibration": str(candidate["calibration"]),
                "selected_threshold": float(selected_threshold),
                "inner_donor_auc": float(donor_metrics_default["auc"]),
                "inner_donor_pr_auc_old": float(donor_metrics_default["pr_auc_old"]),
                "inner_donor_balanced_accuracy_thr_0p5": float(donor_metrics_default["balanced_accuracy"]),
                "inner_donor_balanced_accuracy_thr_opt": float(donor_metrics_opt["balanced_accuracy"]),
                "inner_donor_recall_old_thr_opt": float(donor_metrics_opt["recall_old"]),
                "inner_donor_recall_young_thr_opt": float(donor_metrics_opt["recall_young"]),
                "n_oof_donors": int(oof.shape[0]),
            }
        )

    selection_df = pd.DataFrame(summary_rows).sort_values(
        [
            "inner_donor_balanced_accuracy_thr_opt",
            "inner_donor_auc",
            "model_c",
            "model_l1_ratio",
            "candidate_id",
        ],
        ascending=[False, False, True, False, True],
    ).reset_index(drop=True)
    selection_df["selection_rank"] = selection_df.index + 1
    selection_df["selected_candidate"] = False
    selection_df.loc[0, "selected_candidate"] = True

    selected_row = selection_df.iloc[0]
    selected_candidate = {
        "candidate_id": str(selected_row["candidate_id"]),
        "C": float(selected_row["model_c"]),
        "l1_ratio": float(selected_row["model_l1_ratio"]),
        "calibration": str(selected_row["calibration"]),
        "selected_threshold": float(selected_row["selected_threshold"]),
    }
    selected_candidate["calibrator"] = calibrators[selected_candidate["candidate_id"]]
    threshold_df = pd.concat(threshold_frames, ignore_index=True)
    threshold_df["selected_candidate"] = threshold_df["candidate_id"] == selected_candidate["candidate_id"]
    selection_predictions_df = selection_predictions[selected_candidate["candidate_id"]].copy()
    selection_predictions_df["selected_candidate"] = True

    repeat_df = pd.DataFrame(candidate_repeat_rows)
    repeat_df["selected_candidate"] = repeat_df["candidate_id"] == selected_candidate["candidate_id"]
    selection_df = selection_df.merge(
        repeat_df.groupby("candidate_id", as_index=False)
        .agg(
            inner_repeat_donor_auc_mean=("inner_repeat_donor_auc", "mean"),
            inner_repeat_donor_balanced_accuracy_thr_0p5_mean=("inner_repeat_donor_balanced_accuracy_thr_0p5", "mean"),
            inner_repeat_n_val_donors_mean=("n_val_donors", "mean"),
        ),
        on="candidate_id",
        how="left",
    )

    return {
        "selected_candidate": selected_candidate,
        "selection_summary": selection_df,
        "selection_predictions": selection_predictions_df,
        "threshold_diagnostics": threshold_df,
        "candidate_repeat_summary": repeat_df,
    }


def summarize_outer_split_results(split_df: pd.DataFrame) -> dict[str, float]:
    if split_df.empty:
        return {
            "mean_donor_auc": float("nan"),
            "std_donor_auc": float("nan"),
            "mean_donor_balanced_accuracy": float("nan"),
            "std_donor_balanced_accuracy": float("nan"),
            "mean_donor_recall_old": float("nan"),
            "mean_donor_recall_young": float("nan"),
            "mean_selected_threshold": float("nan"),
            "std_selected_threshold": float("nan"),
        }
    return {
        "mean_donor_auc": float(split_df["holdout_donor_auc"].mean()),
        "std_donor_auc": float(split_df["holdout_donor_auc"].std(ddof=0)),
        "mean_donor_balanced_accuracy": float(split_df["holdout_donor_balanced_accuracy_thr_opt"].mean()),
        "std_donor_balanced_accuracy": float(split_df["holdout_donor_balanced_accuracy_thr_opt"].std(ddof=0)),
        "mean_donor_recall_old": float(split_df["holdout_donor_recall_old_thr_opt"].mean()),
        "mean_donor_recall_young": float(split_df["holdout_donor_recall_young_thr_opt"].mean()),
        "mean_selected_threshold": float(split_df["selected_threshold"].mean()),
        "std_selected_threshold": float(split_df["selected_threshold"].std(ddof=0)),
    }


def run_mouse_outer_cv(
    raw_adata: sc.AnnData,
    config: MouseTrainingConfig,
) -> dict[str, Any]:
    donor_table = build_donor_table(raw_adata)
    outer_rows: list[dict[str, Any]] = []
    selection_frames: list[pd.DataFrame] = []
    threshold_frames: list[pd.DataFrame] = []
    donor_prediction_frames: list[pd.DataFrame] = []

    for repeat_idx in range(resolve_outer_split_repeats(config)):
        test_donors = select_test_donors_from_table(
            donor_table,
            seed=int(config.random_state) + repeat_idx,
            split_policy=config.split_policy,
            test_fraction=float(config.test_fraction),
            min_test_donors_per_class=int(config.min_test_donors_per_class),
            coarse_donor_ids=config.coarse_donor_ids,
        )
        outer_train_raw, outer_test_raw, split_manifest = split_raw_adata_by_donor(raw_adata, test_donors)
        split_manifest["repeat_idx"] = int(repeat_idx)
        split_manifest["split_id"] = f"outer_{repeat_idx:02d}"

        selection = select_best_mouse_candidate(
            outer_train_raw,
            config,
            seed_base=int(config.random_state) * 1000 + repeat_idx * 100,
            context_label=str(split_manifest["split_id"]),
        )
        selected_candidate = selection["selected_candidate"]

        _, outer_train_bs = generate_mouse_bootstrap_metacells(
            outer_train_raw,
            config,
            random_state=int(config.random_state) * 10 + repeat_idx,
            rebalance=True,
        )
        _, outer_test_bs = generate_mouse_bootstrap_metacells(
            outer_test_raw,
            config,
            random_state=int(config.random_state) * 10 + 1000 + repeat_idx,
            rebalance=False,
        )
        outer_train_hvg, outer_test_hvg, _ = preprocess_mouse_metacell_pair(outer_train_bs, outer_test_bs, config)
        _, donor_predictions, split_row, _ = fit_mouse_candidate_on_processed_split(
            outer_train_hvg,
            outer_test_hvg,
            selected_candidate,
            selected_threshold=float(selected_candidate["selected_threshold"]),
            calibrator=selected_candidate["calibrator"],
            split_manifest=split_manifest,
            threshold_source="inner_oof_donor",
        )
        split_row["repeat_idx"] = int(repeat_idx)
        split_row["split_id"] = str(split_manifest["split_id"])
        outer_rows.append(split_row)

        donor_predictions["context_label"] = str(split_manifest["split_id"])
        donor_predictions["repeat_idx"] = int(repeat_idx)
        donor_prediction_frames.append(donor_predictions)
        selection_frames.append(selection["selection_summary"])
        threshold_frames.append(selection["threshold_diagnostics"])

    split_df = pd.DataFrame(outer_rows).sort_values("repeat_idx").reset_index(drop=True)
    return {
        "split_results": split_df,
        "summary": summarize_outer_split_results(split_df),
        "inner_selection_summary": pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame(),
        "threshold_diagnostics": pd.concat(threshold_frames, ignore_index=True) if threshold_frames else pd.DataFrame(),
        "donor_predictions": pd.concat(donor_prediction_frames, ignore_index=True)
        if donor_prediction_frames
        else pd.DataFrame(),
    }


def evaluate_mouse_source_holdouts(
    adata_all: sc.AnnData,
    config: MouseTrainingConfig,
) -> dict[str, Any]:
    if not config.study_holdout_enabled:
        return {
            "metrics": pd.DataFrame(),
            "not_evaluable": [],
            "selection_summary": pd.DataFrame(),
            "threshold_diagnostics": pd.DataFrame(),
        }

    source_rows: list[dict[str, Any]] = []
    not_evaluable: list[dict[str, Any]] = []
    selection_frames: list[pd.DataFrame] = []
    threshold_frames: list[pd.DataFrame] = []

    for source_name in sorted(map(str, adata_all.obs["source"].astype(str).unique())):
        source_mask = adata_all.obs["source"].astype(str) == source_name
        source_adata = adata_all[source_mask].copy()
        source_donor_table = build_donor_table(source_adata)
        class_counts = source_donor_table["age_label"].value_counts().to_dict()

        if class_counts.get("young", 0) < int(config.min_test_donors_per_class) or class_counts.get(
            "old",
            0,
        ) < int(config.min_test_donors_per_class):
            not_evaluable.append(
                {
                    "source": source_name,
                    "reason": "insufficient_test_donors_per_class",
                    "young_donors": int(class_counts.get("young", 0)),
                    "old_donors": int(class_counts.get("old", 0)),
                }
            )
            continue

        holdout_donors = set(source_donor_table["donor_id"].astype(str))
        train_mask = ~adata_all.obs["donor_split_id"].astype(str).isin(holdout_donors)
        train_raw = adata_all[train_mask].copy()
        test_raw = adata_all[~train_mask].copy()
        train_donor_table = build_donor_table(train_raw)
        train_class_counts = train_donor_table["age_label"].value_counts().to_dict()
        if train_class_counts.get("young", 0) < 2 or train_class_counts.get("old", 0) < 2:
            not_evaluable.append(
                {
                    "source": source_name,
                    "reason": "insufficient_training_donors_per_class",
                    "young_donors": int(train_class_counts.get("young", 0)),
                    "old_donors": int(train_class_counts.get("old", 0)),
                }
            )
            continue

        selection = select_best_mouse_candidate(
            train_raw,
            config,
            seed_base=int(config.random_state) * 10000 + sum(ord(ch) for ch in source_name),
            context_label=f"study_holdout::{source_name}",
        )
        selected_candidate = selection["selected_candidate"]
        selection_frames.append(selection["selection_summary"])
        threshold_frames.append(selection["threshold_diagnostics"])
        _, train_bs = generate_mouse_bootstrap_metacells(
            train_raw,
            config,
            random_state=int(config.random_state) + 2000 + len(source_rows),
            rebalance=True,
        )
        _, test_bs = generate_mouse_bootstrap_metacells(
            test_raw,
            config,
            random_state=int(config.random_state) + 3000 + len(source_rows),
            rebalance=False,
        )
        train_hvg, test_hvg, _ = preprocess_mouse_metacell_pair(train_bs, test_bs, config)
        split_manifest = {
            "split_id": f"study_holdout::{source_name}",
            "train_donors": sorted(train_raw.obs["donor_split_id"].astype(str).unique().tolist()),
            "test_donors": sorted(test_raw.obs["donor_split_id"].astype(str).unique().tolist()),
            "train_test_donor_overlap_count": 0,
        }
        _, _, split_row, _ = fit_mouse_candidate_on_processed_split(
            train_hvg,
            test_hvg,
            selected_candidate,
            selected_threshold=float(selected_candidate["selected_threshold"]),
            calibrator=selected_candidate["calibrator"],
            split_manifest=split_manifest,
            threshold_source="inner_oof_donor",
        )
        split_row["holdout_source"] = source_name
        split_row["context_label"] = f"study_holdout::{source_name}"
        split_row["configured_excluded_source"] = bool(source_name in set(map(str, config.holdout_sources)))
        source_rows.append(split_row)

    return {
        "metrics": pd.DataFrame(source_rows),
        "not_evaluable": not_evaluable,
        "selection_summary": pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame(),
        "threshold_diagnostics": pd.concat(threshold_frames, ignore_index=True) if threshold_frames else pd.DataFrame(),
    }


def build_mouse_full_model_artifacts(
    raw_adata: sc.AnnData,
    config: MouseTrainingConfig,
) -> dict[str, Any]:
    selection = select_best_mouse_candidate(
        raw_adata,
        config,
        seed_base=int(config.random_state) * 50000,
        context_label="full_data_selection",
    )
    selected_candidate = selection["selected_candidate"]
    full_bs_pre, full_bs = generate_mouse_bootstrap_metacells(
        raw_adata,
        config,
        random_state=int(config.random_state),
        rebalance=True,
    )
    full_hvg, full_pyscenic, _ = preprocess_mouse_full_metacells(full_bs, config)
    clf = abclock_model.fit_elasticnet_classifier(
        full_hvg,
        age_col="Age",
        model_params={"C": float(selected_candidate["C"]), "l1_ratio": float(selected_candidate["l1_ratio"])},
    )
    full_prob_raw = clf.predict_proba(dense_matrix(full_hvg.X))[:, 1]
    full_prob = abclock_model.apply_calibration(selected_candidate["calibrator"], full_prob_raw)
    full_y = (full_hvg.obs["Age"].astype(str) == "old").astype(int).to_numpy()
    full_metrics = abclock_model.evaluate_binary_metrics(
        full_y,
        full_prob,
        threshold=float(selected_candidate["selected_threshold"]),
    )
    gene_weights = pd.DataFrame({"Gene": full_hvg.var_names.astype(str), "Weight": clf.coef_[0]}).sort_values(
        by="Weight",
        ascending=False,
    ).reset_index(drop=True)

    bundle = abclock_model.build_model_bundle(
        clf,
        gene_names=full_hvg.var_names.astype(str).tolist(),
        selected_threshold=float(selected_candidate["selected_threshold"]),
        selected_calibration=str(selected_candidate["calibration"]),
        calibrator=selected_candidate["calibrator"],
        metadata={
            "model_params": {
                "C": float(selected_candidate["C"]),
                "l1_ratio": float(selected_candidate["l1_ratio"]),
                "penalty": "elasticnet",
                "solver": "saga",
                "max_iter": 5000,
                "class_weight": "balanced",
            },
            "threshold_objective": str(config.threshold_objective),
        },
    )
    return {
        "selection": selection,
        "model_bundle": bundle,
        "classifier": clf,
        "gene_weights": gene_weights,
        "train_metrics": full_metrics,
        "adata_musc_bs_pre_balance": full_bs_pre,
        "adata_musc_bs": full_bs,
        "adata_musc_combined": full_hvg,
        "adata_musc_pyscenic": full_pyscenic,
    }


def step_setup_runtime() -> None:
    sc.settings.set_figure_params(dpi=400, frameon=False, facecolor="white")
    warnings.filterwarnings("ignore")


def step_load_mouse_datasets(paths: MouseTrainingPaths) -> dict[str, sc.AnnData]:
    print("[1/6] Loading datasets")
    adatas: dict[str, sc.AnnData] = {}
    for name, path in paths.input_files.items():
        print(f"  Loading {name}...")
        adatas[name] = sc.read_h5ad(path)
        print(f"    shape={adatas[name].shape}")
    return adatas


def step_harmonize_mouse_datasets(
    adatas: dict[str, sc.AnnData],
    paths: MouseTrainingPaths,
) -> dict[str, sc.AnnData]:
    print("\n[2/6] Harmonizing genes and age metadata")
    for name, adata in adatas.items():
        print(f"  {name}: {identify_gene_type(adata.var_names)}")

    ensembl_to_symbol = pd.read_csv(paths.mapping_path, sep="\t")
    mapping = dict(zip(ensembl_to_symbol["ensembl"], ensembl_to_symbol["symbol"]))

    for name in [
        "TabulaMuris_limb_10x",
        "TabulaMuris_limb_smartseq2",
        "TabulaMuris_diaphragm_smartseq2",
    ]:
        adata = adatas[name]
        adata.var_names = pd.Index([mapping.get(str(g), str(g)) for g in adata.var_names])

    for name, adata in adatas.items():
        adata.var_names_make_unique(join="first")
        adata.var_names = pd.Index([str(x).upper() for x in adata.var_names])

        if "age_month" not in adata.obs.columns:
            if name == "SKM_mouse_raw":
                adata.obs["age_month"] = adata.obs["Age_bin"].apply(parse_skm_age_bin)
                adata.obs["Age_group"] = adata.obs["Age_bin"].apply(
                    lambda x: "young"
                    if str(x).lower() == "young"
                    else "old"
                    if str(x).lower() == "old"
                    else "unknown"
                )
            elif name.startswith("TabulaMuris"):
                adata.obs["age_month"] = adata.obs["age"].apply(parse_tabulamuris_age)
                adata.obs["Age_group"] = adata.obs["age_month"].apply(
                    lambda x: "young" if x is not None and x < 18 else "old" if x is not None else "unknown"
                )
            elif name == "GSE150366_noninj":
                adata.obs["age_month"] = 3.0
                adata.obs["Age_group"] = "young"
            else:
                adata.obs["age_month"] = np.nan
                adata.obs["Age_group"] = "unknown"

        if "Age_group" not in adata.obs.columns:
            if "age_group" in adata.obs.columns:
                adata.obs["Age_group"] = adata.obs["age_group"].astype(str).str.lower()
            elif "age_month" in adata.obs.columns:
                adata.obs["Age_group"] = adata.obs["age_month"].apply(
                    lambda x: "young" if pd.notna(x) and x < 18 else "old" if pd.notna(x) else "unknown"
                )

    return adatas


def step_build_mouse_training_matrix(
    adatas: dict[str, sc.AnnData],
    config: MouseTrainingConfig,
) -> dict[str, Any]:
    print("\n[3/6] Filtering MuSC and building combined training matrix")
    musc_adatas: dict[str, sc.AnnData] = {}
    for name, adata in adatas.items():
        musc_adatas[name] = filter_musc(adata, name)

    standardized_by_source: Dict[str, sc.AnnData] = {}
    for name, adata in musc_adatas.items():
        std = standardize_obs(adata, name)
        std.var_names_make_unique(join="first")
        standardized_by_source[name] = std

    all_sources = list(standardized_by_source.keys())
    holdout_sources = set(map(str, config.holdout_sources))
    force_excluded_sources = set(map(str, config.exclude_sources))

    if config.holdout_policy == "study_disjoint":
        excluded_sources = {s for s in all_sources if s in holdout_sources}
    else:
        excluded_sources = set()
    excluded_sources = sorted(excluded_sources.union({s for s in all_sources if s in force_excluded_sources}))
    included_sources = [s for s in all_sources if s not in set(excluded_sources)]
    if not included_sources:
        raise ValueError("No included sources remain after holdout; adjust TRAINING_CONFIG.holdout_sources.")

    print(f"  Holdout policy: {config.holdout_policy}")
    print(f"  Included sources: {included_sources}")
    print(f"  Excluded sources: {excluded_sources}")

    adata_all_sources = concat_on_common_genes([standardized_by_source[s] for s in all_sources])
    print(f"  Combined (all candidate sources) shape: {adata_all_sources.shape}")
    print(f"  Age distribution (all ages):\n{adata_all_sources.obs['Age_group_std'].value_counts()}")

    train_age_mask = adata_all_sources.obs["Age_group_std"].isin(["young", "old"])
    adata_all = adata_all_sources[train_age_mask].copy()
    adata_all, linkage_map = add_linkage_aware_donor_ids(adata_all)

    excluded_source_mask = adata_all.obs["source"].astype(str).isin(set(excluded_sources))
    excluded_donor_ids = set(adata_all.obs.loc[excluded_source_mask, "donor_split_id"].astype(str))
    overlap_donor_ids = set(adata_all.obs.loc[~excluded_source_mask, "donor_split_id"].astype(str)).intersection(
        excluded_donor_ids
    )
    train_mask = (~excluded_source_mask) & (~adata_all.obs["donor_split_id"].astype(str).isin(overlap_donor_ids))
    adata_combined = adata_all[train_mask].copy()
    adata_excluded = adata_all[~train_mask].copy()

    print(f"  Combined (young/old only, all sources) shape: {adata_all.shape}")
    print(f"  Training partition shape: {adata_combined.shape}")
    if adata_excluded.n_obs > 0:
        print(f"  Excluded/benchmark partition shape: {adata_excluded.shape}")
    print(f"  Age distribution (training):\n{adata_combined.obs['Age_group_std'].value_counts()}")
    print(f"  Linked sample IDs across sources: {int(linkage_map['is_linked_across_sources'].sum())}")
    print(f"  Cross-partition linked split groups moved out of training: {len(overlap_donor_ids)}")
    print(f"  Unique split groups (training): {adata_combined.obs['donor_split_id'].nunique()}")
    print(f"  Unique bootstrap donors (training): {adata_combined.obs['donor_bootstrap_id'].nunique()}")

    return {
        "adata_all": adata_all,
        "adata_combined": adata_combined,
        "adata_excluded": adata_excluded,
        "linkage_map": linkage_map,
        "included_sources": included_sources,
        "excluded_sources": excluded_sources,
        "cross_partition_linked_groups": sorted(overlap_donor_ids),
    }


def step_save_mouse_training_atlas(
    adata_combined: sc.AnnData,
    linkage_map: pd.DataFrame,
    paths: MouseTrainingPaths,
    config: MouseTrainingConfig,
) -> dict[str, Any]:
    print("\n[3.5/6] Saving train-only raw-count atlas")
    artifact_dir = Path(paths.artifact_dir)
    training_atlas_path = Path(paths.training_atlas_path)
    linkage_map_path = artifact_dir / "donor_linkage_map.tsv"
    linkage_map.to_csv(linkage_map_path, sep="\t", index=False)

    adata_training_atlas = None
    if config.artifact_mode == "full":
        raw_counts = adata_combined.layers["counts"] if "counts" in adata_combined.layers else adata_combined.X
        adata_training_atlas = sc.AnnData(
            X=raw_counts,
            obs=adata_combined.obs.copy(),
            var=adata_combined.var.copy(),
        )
        adata_training_atlas.layers["counts"] = raw_counts
        safe_write_h5ad(adata_training_atlas, training_atlas_path)
        print(f"  Saved training atlas: {training_atlas_path}")
    else:
        print("  Skipping training atlas write (artifact_mode=metrics_only)")

    return {
        "linkage_map_path": linkage_map_path,
        "training_atlas_path": training_atlas_path,
        "adata_training_atlas": adata_training_atlas,
    }


def step_build_mouse_metacell_matrices(
    adata_combined: sc.AnnData,
    config: MouseTrainingConfig,
) -> dict[str, Any]:
    print("\n[4/6] Bootstrap metacells + preprocessing")
    adata_musc_bs = abclock_metacells.generate_bootstrap_cells(
        adata_combined,
        n_cells_per_bin=15,
        n_iter=config.bootstrap_iters,
        donor_col="donor_bootstrap_id",
        age_col="Age_group_std",
        state_col="celltype_std",
        extra_meta_cols=[
            "Sex_std",
            "celltype_std",
            "Age_group_std",
            "source",
            "sample_id_std",
            "donor_bootstrap_id",
            "donor_split_id",
        ],
        balance_classes=True,
        n_iter_minority=config.bootstrap_iters,
        random_state=config.random_state,
    )
    adata_musc_bs_pre_balance = adata_musc_bs.copy()
    if config.balance_mode == "donor_stratified":
        adata_musc_bs = rebalance_metacells_by_donor(
            adata_musc_bs,
            target_old_to_young=config.target_class_ratio,
            age_col="Age",
            donor_col="donor_split_id",
            random_state=config.random_state,
        )
        print("  Rebalanced metacells (donor_stratified) age counts:\n" f"{adata_musc_bs.obs['Age'].value_counts()}")
    else:
        print("  Skipping metacell balancing (balance_mode=none)")

    sc.pp.filter_genes(adata_musc_bs, min_counts=50)
    mt_genes = [g for g in adata_musc_bs.var_names if g.lower().startswith("mt-")]
    adata_musc_bs = adata_musc_bs[:, ~adata_musc_bs.var_names.isin(mt_genes)].copy()
    sc.pp.normalize_total(adata_musc_bs, target_sum=1e4)
    sc.pp.log1p(adata_musc_bs)

    adata_musc_full = adata_musc_bs.copy() if config.artifact_mode == "full" else None
    sc.pp.highly_variable_genes(adata_musc_bs, n_top_genes=config.n_training_hvg, subset=True)
    adata_musc_combined = adata_musc_bs

    adata_musc_pyscenic = None
    if config.artifact_mode == "full" and adata_musc_full is not None:
        adata_musc_pyscenic = adata_musc_full.copy()
        sc.pp.highly_variable_genes(adata_musc_pyscenic, n_top_genes=config.n_pyscenic_hvg, subset=True)

    print(f"  Training matrix shape: {adata_musc_combined.shape}")
    if adata_musc_pyscenic is not None:
        print(f"  pySCENIC matrix shape: {adata_musc_pyscenic.shape}")
    else:
        print("  Skipping pySCENIC matrix materialization (artifact_mode=metrics_only)")

    return {
        "adata_musc_bs_pre_balance": adata_musc_bs_pre_balance,
        "adata_musc_bs": adata_musc_bs,
        "adata_musc_combined": adata_musc_combined,
        "adata_musc_pyscenic": adata_musc_pyscenic,
    }


def step_train_mouse_model(
    matrix_info: dict[str, Any],
    config: MouseTrainingConfig,
) -> dict[str, Any]:
    if str(config.execution_mode) != "train_and_evaluate":
        raise ValueError(f"Unsupported execution_mode: {config.execution_mode}")

    print("\n[4/6] Running donor-aware nested evaluation")
    outer_cv = run_mouse_outer_cv(matrix_info["adata_combined"], config)
    outer_summary = outer_cv["summary"]
    print(
        "  Outer donor CV:"
        f" mean donor AUC={outer_summary['mean_donor_auc']:.4f},"
        f" mean donor balanced accuracy={outer_summary['mean_donor_balanced_accuracy']:.4f}"
    )

    print("\n[5/6] Fitting final age-state classifier on full training partition")
    full_artifacts = build_mouse_full_model_artifacts(matrix_info["adata_combined"], config)
    selection = full_artifacts["selection"]
    selected_candidate = selection["selected_candidate"]
    print(
        "  Final selected parameters:"
        f" C={selected_candidate['C']},"
        f" l1_ratio={selected_candidate['l1_ratio']},"
        f" calibration={selected_candidate['calibration']},"
        f" threshold={selected_candidate['selected_threshold']:.4f}"
    )

    print("  Running source-holdout benchmarks")
    study_holdouts = evaluate_mouse_source_holdouts(matrix_info["adata_all"], config)

    split_df = outer_cv["split_results"]
    best_split_row = (
        split_df.sort_values("holdout_donor_auc", ascending=False).iloc[0].to_dict() if not split_df.empty else {}
    )
    test_donors = []
    if best_split_row.get("test_donors"):
        test_donors = [item for item in str(best_split_row["test_donors"]).split("|") if item]

    result = {
        "model": full_artifacts["model_bundle"],
        "gene_weights": full_artifacts["gene_weights"],
        "train_metrics": {
            "auc": float(full_artifacts["train_metrics"]["auc"]),
            "split_mean_auc": float(split_df["train_auc_train_only"].mean()) if not split_df.empty else float("nan"),
            "split_std_auc": float(split_df["train_auc_train_only"].std(ddof=0))
            if not split_df.empty
            else float("nan"),
            "split_mean_balanced_accuracy": float(outer_summary["mean_donor_balanced_accuracy"]),
            "split_mean_recall_old": float(outer_summary["mean_donor_recall_old"]),
            "split_mean_recall_young": float(outer_summary["mean_donor_recall_young"]),
            "y_prob": abclock_model.predict_age(full_artifacts["model_bundle"], full_artifacts["adata_musc_combined"]),
            "y_true": (full_artifacts["adata_musc_combined"].obs["Age"].astype(str) == "old").astype(int).to_numpy(),
        },
        "test_metrics": {
            "auc": float(outer_summary["mean_donor_auc"]),
            "auc_std": float(outer_summary["std_donor_auc"]),
            "pr_auc_old_mean": float(split_df["holdout_donor_pr_auc_old"].mean()) if not split_df.empty else float("nan"),
            "recall_old_mean": float(outer_summary["mean_donor_recall_old"]),
            "recall_young_mean": float(outer_summary["mean_donor_recall_young"]),
            "balanced_accuracy_mean": float(outer_summary["mean_donor_balanced_accuracy"]),
            "best_split_pr_auc_old": float(best_split_row.get("holdout_donor_pr_auc_old", float("nan"))),
            "best_split_recall_old": float(best_split_row.get("holdout_donor_recall_old_thr_opt", float("nan"))),
            "best_split_recall_young": float(best_split_row.get("holdout_donor_recall_young_thr_opt", float("nan"))),
            "best_split_balanced_accuracy": float(
                best_split_row.get("holdout_donor_balanced_accuracy_thr_opt", float("nan"))
            ),
            "y_prob": np.asarray([], dtype=float),
            "y_true": np.asarray([], dtype=int),
        },
        "test_donors": test_donors,
        "split_policy": str(config.split_policy),
        "split_diagnostics": split_df,
        "inner_selection_summary": pd.concat(
            [
                outer_cv["inner_selection_summary"],
                selection["selection_summary"],
                study_holdouts["selection_summary"],
            ],
            ignore_index=True,
        ),
        "threshold_diagnostics": pd.concat(
            [
                outer_cv["threshold_diagnostics"],
                selection["threshold_diagnostics"],
                study_holdouts["threshold_diagnostics"],
            ],
            ignore_index=True,
        ),
        "study_holdout_metrics": study_holdouts["metrics"],
        "study_holdout_not_evaluable": study_holdouts["not_evaluable"],
        "outer_cv_summary": outer_summary,
        "full_data_selected_params": {
            "C": float(selected_candidate["C"]),
            "l1_ratio": float(selected_candidate["l1_ratio"]),
            "calibration": str(selected_candidate["calibration"]),
        },
        "selected_threshold": float(selected_candidate["selected_threshold"]),
        "selected_calibration": str(selected_candidate["calibration"]),
        "selected_threshold_source": "inner_oof_donor",
    }
    return {
        "result": result,
        "coarse_donor_ids": list(map(str, config.coarse_donor_ids)),
        "adata_musc_bs_pre_balance": full_artifacts["adata_musc_bs_pre_balance"],
        "adata_musc_bs": full_artifacts["adata_musc_bs"],
        "adata_musc_combined": full_artifacts["adata_musc_combined"],
        "adata_musc_pyscenic": full_artifacts["adata_musc_pyscenic"],
    }


def step_save_mouse_training_artifacts(
    paths: MouseTrainingPaths,
    config: MouseTrainingConfig,
    matrix_info: dict[str, Any],
    atlas_info: dict[str, Any],
    model_info: dict[str, Any],
) -> dict[str, Any]:
    print("\n[6/6] Saving artifacts")
    artifact_dir = Path(paths.artifact_dir)
    result = model_info["result"]
    coarse_donor_ids = model_info["coarse_donor_ids"]
    adata_combined = matrix_info["adata_combined"]
    linkage_map = matrix_info["linkage_map"]
    included_sources = matrix_info["included_sources"]
    excluded_sources = matrix_info["excluded_sources"]
    adata_musc_bs_pre_balance = model_info["adata_musc_bs_pre_balance"]
    adata_musc_bs = model_info["adata_musc_bs"]
    adata_musc_combined = model_info["adata_musc_combined"]
    adata_musc_pyscenic = model_info["adata_musc_pyscenic"]
    training_atlas_path = atlas_info["training_atlas_path"]
    linkage_map_path = atlas_info["linkage_map_path"]

    model_path = artifact_dir / "final_model.joblib"
    genes_path = artifact_dir / "model_genes.txt"
    weights_path = artifact_dir / "gene_weights.tsv"
    metrics_path = artifact_dir / "training_metrics.json"
    split_diag_path = artifact_dir / "split_diagnostics.tsv"
    inner_selection_path = artifact_dir / "inner_selection_summary.tsv"
    threshold_diag_path = artifact_dir / "threshold_diagnostics.tsv"
    study_holdout_path = artifact_dir / "study_holdout_metrics.tsv"
    balance_diag_path = artifact_dir / "class_balance_diagnostics.tsv"
    pyscenic_path = artifact_dir / "adata_musc_pyscenic.h5ad"
    train_hvg_path = artifact_dir / "adata_musc_combined_hvg.h5ad"

    if config.artifact_mode == "full":
        joblib.dump(result["model"], model_path)
        result["gene_weights"].to_csv(weights_path, sep="\t", index=False)

        with open(genes_path, "w", encoding="utf-8") as f:
            for gene in adata_musc_combined.var_names.astype(str):
                f.write(f"{gene}\n")

        safe_write_h5ad(adata_musc_pyscenic, pyscenic_path)
        safe_write_h5ad(adata_musc_combined, train_hvg_path)
    else:
        print("  Skipping model/gene/h5ad artifact writes (artifact_mode=metrics_only)")

    if isinstance(result.get("split_diagnostics"), pd.DataFrame):
        result["split_diagnostics"].to_csv(split_diag_path, sep="\t", index=False)
    if isinstance(result.get("inner_selection_summary"), pd.DataFrame):
        result["inner_selection_summary"].to_csv(inner_selection_path, sep="\t", index=False)
    if isinstance(result.get("threshold_diagnostics"), pd.DataFrame):
        result["threshold_diagnostics"].to_csv(threshold_diag_path, sep="\t", index=False)
    if isinstance(result.get("study_holdout_metrics"), pd.DataFrame):
        result["study_holdout_metrics"].to_csv(study_holdout_path, sep="\t", index=False)

    balance_diag = build_class_balance_diagnostics(adata_combined, adata_musc_bs_pre_balance, adata_musc_bs)
    balance_diag.to_csv(balance_diag_path, sep="\t", index=False)

    metrics = {
        "train_auc": float(result["train_metrics"]["auc"]),
        "split_train_auc_mean": float(result["train_metrics"].get("split_mean_auc", np.nan)),
        "split_train_auc_std": float(result["train_metrics"].get("split_std_auc", np.nan)),
        "split_balanced_accuracy_mean": float(result["train_metrics"].get("split_mean_balanced_accuracy", np.nan)),
        "split_recall_old_mean": float(result["train_metrics"].get("split_mean_recall_old", np.nan)),
        "split_recall_young_mean": float(result["train_metrics"].get("split_mean_recall_young", np.nan)),
        "test_auc": float(result["test_metrics"]["auc"]),
        "test_auc_std": float(result["test_metrics"].get("auc_std", 0.0)),
        "test_pr_auc_old_mean": float(result["test_metrics"].get("pr_auc_old_mean", np.nan)),
        "test_balanced_accuracy_mean": float(result["test_metrics"].get("balanced_accuracy_mean", np.nan)),
        "test_recall_old_mean": float(result["test_metrics"].get("recall_old_mean", np.nan)),
        "test_recall_young_mean": float(result["test_metrics"].get("recall_young_mean", np.nan)),
        "test_donors": list(map(str, result["test_donors"])),
        "n_training_metacells": int(adata_musc_combined.n_obs),
        "n_training_genes": int(adata_musc_combined.n_vars),
        "n_pyscenic_genes": int(adata_musc_pyscenic.n_vars) if adata_musc_pyscenic is not None else 0,
        "n_training_hvg": int(config.n_training_hvg),
        "n_pyscenic_hvg": int(config.n_pyscenic_hvg),
        "bootstrap_iters": int(config.bootstrap_iters),
        "random_state": int(config.random_state),
        "sklearn_version": str(sklearn.__version__),
        "runtime_versions": collect_runtime_versions({"scanpy": sc}),
        "artifact_mode": str(config.artifact_mode),
        "holdout_policy": str(config.holdout_policy),
        "validation_design": "study_disjoint_holdout" if excluded_sources else "repeated_nested_donor_cv_internal",
        "study_holdout_name": "|".join(excluded_sources) if excluded_sources else None,
        "model_params": {
            "C": float(result["full_data_selected_params"]["C"]),
            "l1_ratio": float(result["full_data_selected_params"]["l1_ratio"]),
            "penalty": "elasticnet",
            "solver": "saga",
            "max_iter": 5000,
            "class_weight": "balanced",
        },
        "split_policy": str(result.get("split_policy", config.split_policy)),
        "test_fraction": float(config.test_fraction),
        "coarse_donor_ids": coarse_donor_ids,
        "n_split_repeats": int(resolve_outer_split_repeats(config)),
        "outer_split_repeats": int(resolve_outer_split_repeats(config)),
        "inner_split_repeats": int(config.inner_split_repeats),
        "balance_mode": str(config.balance_mode),
        "target_class_ratio_old_to_young": float(config.target_class_ratio),
        "min_test_donors_per_class": int(config.min_test_donors_per_class),
        "split_diagnostics_path": str(split_diag_path),
        "inner_selection_summary_path": str(inner_selection_path),
        "threshold_diagnostics_path": str(threshold_diag_path),
        "study_holdout_metrics_path": str(study_holdout_path),
        "class_balance_diagnostics_path": str(balance_diag_path),
        "donor_linkage_map_path": str(linkage_map_path),
        "n_linked_sample_ids": int(linkage_map["is_linked_across_sources"].sum()),
        "n_unique_split_groups": int(adata_combined.obs["donor_split_id"].nunique()),
        "included_sources": included_sources,
        "excluded_sources": excluded_sources,
        "cross_partition_linked_groups": matrix_info["cross_partition_linked_groups"],
        "n_cells_training_atlas": int(adata_combined.n_obs),
        "n_genes_training_atlas": int(adata_combined.n_vars),
        "training_atlas_path": str(training_atlas_path) if config.artifact_mode == "full" else None,
        "input_files": {name: str(path) for name, path in paths.input_files.items()},
        "artifact_dir": str(artifact_dir),
        "execution_mode": str(config.execution_mode),
        "primary_metric_level": "donor",
        "selected_threshold": float(result["selected_threshold"]),
        "selected_threshold_source": str(result["selected_threshold_source"]),
        "selected_calibration": str(result["selected_calibration"]),
        "outer_cv_summary": result["outer_cv_summary"],
        "full_data_selected_params": result["full_data_selected_params"],
        "study_holdout_not_evaluable": result.get("study_holdout_not_evaluable", []),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved:")
    print(f"  - {metrics_path}")
    print(f"  - {split_diag_path}")
    print(f"  - {inner_selection_path}")
    print(f"  - {threshold_diag_path}")
    print(f"  - {study_holdout_path}")
    print(f"  - {balance_diag_path}")
    print(f"  - {linkage_map_path}")
    if config.artifact_mode == "full":
        print(f"  - {model_path}")
        print(f"  - {genes_path}")
        print(f"  - {weights_path}")
        print(f"  - {pyscenic_path}")
        print(f"  - {train_hvg_path}")
        print(f"  - {training_atlas_path}")

    return metrics


def run_mouse_training(
    paths: MouseTrainingPaths = TRAINING_PATHS,
    config: MouseTrainingConfig = TRAINING_CONFIG,
) -> dict[str, Any]:
    validate_required_paths(
        {
            **{f"mouse training input {name}": path for name, path in paths.input_files.items()},
            "mouse Ensembl mapping": paths.mapping_path,
        }
    )

    artifact_dir = Path(paths.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    step_setup_runtime()
    adatas = step_load_mouse_datasets(paths)
    adatas = step_harmonize_mouse_datasets(adatas, paths)
    matrix_info = step_build_mouse_training_matrix(adatas, config)
    atlas_info = step_save_mouse_training_atlas(
        matrix_info["adata_combined"],
        matrix_info["linkage_map"],
        paths,
        config,
    )
    model_info = step_train_mouse_model(matrix_info, config)
    return step_save_mouse_training_artifacts(paths, config, matrix_info, atlas_info, model_info)


def _require_file(path: Path, desc: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {desc}: {path}")


def step_load_mouse_post_inputs(paths: MousePostPaths) -> tuple[Path, pd.DataFrame]:
    weights_path = paths.artifact_dir / "gene_weights.tsv"
    validate_required_paths(
        {
            "gene_weights artifact": weights_path,
            "verification input h5ad": paths.verification_input,
        }
    )
    outdir = Path(paths.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    gene_weights = pd.read_csv(weights_path, sep="\t")
    return outdir, gene_weights


def run_grn(paths: MousePostPaths, outdir: Path, gene_weights: pd.DataFrame, config: MousePostConfig) -> dict[str, Any]:
    sys.path.insert(0, str(ABCLOCK_ROOT))
    import grn as abclock_grn  # noqa: E402

    pyscenic_path = paths.artifact_dir / "adata_musc_pyscenic.h5ad"
    _require_file(pyscenic_path, "pySCENIC AnnData artifact")
    _require_file(paths.motif_anno_path, "motif annotation file")
    _require_file(paths.tf_list_path, "mouse TF list")

    db_paths = sorted(paths.db_root.glob("*.feather"))
    if not db_paths:
        raise FileNotFoundError(f"No cisTarget .feather databases found in {paths.db_root}")

    adata_musc_pyscenic = sc.read_h5ad(pyscenic_path)
    adata_musc_pyscenic.var_names = adata_musc_pyscenic.var_names.str.capitalize()
    tf_genes_list = pd.read_csv(paths.tf_list_path, header=None)[0].astype(str).tolist()

    grn_result = abclock_grn.run_regdiffusion_pyscenic_pipeline(
        adata=adata_musc_pyscenic,
        tf_list=tf_genes_list,
        db_paths=db_paths,
        motif_anno_path=str(paths.motif_anno_path),
        num_workers=config.grn_num_workers,
    )

    adj = grn_result.get("adjacencies", pd.DataFrame())
    regulons = grn_result.get("regulons", [])
    auc_mtx = grn_result.get("auc_mtx", pd.DataFrame())

    if isinstance(adj, pd.DataFrame) and not adj.empty:
        adj.to_csv(outdir / "grn_adjacencies.tsv", sep="\t", index=False)
    if hasattr(auc_mtx, "to_csv") and not auc_mtx.empty:
        auc_mtx.to_csv(outdir / "grn_auc_matrix.tsv", sep="\t")

    regulon_rows = []
    for regulon in regulons:
        regulon_rows.append(
            {
                "regulon": getattr(regulon, "name", "unknown"),
                "n_genes": len(getattr(regulon, "genes", []) or []),
                "tf": getattr(regulon, "transcription_factor", ""),
            }
        )
    pd.DataFrame(regulon_rows).to_csv(outdir / "grn_regulons_summary.tsv", sep="\t", index=False)

    top_young = gene_weights.tail(10)["Gene"].astype(str).tolist()
    top_old = gene_weights.head(10)["Gene"].astype(str).tolist()
    clock_genes = {gene.capitalize() for gene in (top_young + top_old)}
    overlap_rows = abclock_grn.find_clock_gene_regulons(regulons, clock_genes)
    pd.DataFrame(overlap_rows, columns=["regulon", "n_genes", "overlap_clock_genes"]).to_csv(
        outdir / "grn_clock_regulon_overlap.tsv",
        sep="\t",
        index=False,
    )

    return {
        "n_regulons": len(regulons),
        "n_adjacencies": int(len(adj)) if isinstance(adj, pd.DataFrame) else 0,
    }


def run_enrichment(paths: MousePostPaths, outdir: Path, gene_weights: pd.DataFrame) -> dict[str, Any]:
    if not paths.gmt_path.exists():
        raise FileNotFoundError(
            "Missing mouse GO GMT file: "
            f"{paths.gmt_path}. Add this file before running clock_system_mouse_post.py."
        )

    sys.path.insert(0, str(ABCLOCK_ROOT))
    import enrichment as abclock_enrichment  # noqa: E402

    gene_weights_df = gene_weights.copy()
    gene_weights_df["Gene_title"] = gene_weights_df["Gene"].astype(str).str.capitalize()

    results = abclock_enrichment.run_age_gene_enrichment(
        gene_weights_df,
        top_n=100,
        gmt_path=str(paths.gmt_path),
        organism="Mouse",
    )

    aging_df = results.get("aging_enrichment", pd.DataFrame())
    youth_df = results.get("youth_enrichment", pd.DataFrame())

    aging_df.to_csv(outdir / "enrichment_aging.tsv", sep="\t", index=False)
    youth_df.to_csv(outdir / "enrichment_youth.tsv", sep="\t", index=False)

    if not aging_df.empty or not youth_df.empty:
        fig = abclock_enrichment.plot_age_enrichment_comparison(
            aging_df,
            youth_df,
            top_n=10,
            figsize=(14, 10),
            aging_color="#ff9f9b",
            youth_color="#8ECFC9",
        )
        _ = fig
        plt.title("Mouse MuSC Aging Clock: Pathway Enrichment", fontsize=14)
        plt.tight_layout()
        plt.savefig(outdir / "mouse_clock_enrichment_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    if not aging_df.empty:
        abclock_enrichment.plot_enrichment_results(
            aging_df,
            title="Aging-Associated Pathways (Mouse MuSC)",
            color="#ff9f9b",
        )
        plt.savefig(outdir / "mouse_clock_aging_pathways.png", dpi=300, bbox_inches="tight")
        plt.close()

    if not youth_df.empty:
        abclock_enrichment.plot_enrichment_results(
            youth_df,
            title="Youth-Associated Pathways (Mouse MuSC)",
            color="#8ECFC9",
        )
        plt.savefig(outdir / "mouse_clock_youth_pathways.png", dpi=300, bbox_inches="tight")
        plt.close()

    return {
        "aging_terms": int(aging_df.shape[0]),
        "youth_terms": int(youth_df.shape[0]),
    }


def run_verification_apply(paths: MousePostPaths, outdir: Path) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_verification_clock as verification_workflow  # noqa: E402

    model_path = paths.artifact_dir / "final_model.joblib"
    genes_path = paths.artifact_dir / "model_genes.txt"
    metrics_path = paths.artifact_dir / "training_metrics.json"

    validate_required_paths(
        {
            "trained model artifact": model_path,
            "model genes artifact": genes_path,
            "verification input h5ad": paths.verification_input,
        }
    )

    verify_outdir = outdir / "verification_results"
    inference_metrics = verification_workflow.run_verification_clock(
        model_path=model_path,
        model_genes_path=genes_path,
        input_h5ad=paths.verification_input,
        outdir=verify_outdir,
        training_metrics_path=metrics_path if metrics_path.exists() else None,
    )

    return {
        "verification_outdir": str(verify_outdir),
        "analysis_mode": inference_metrics.get("analysis_mode"),
        "validation_tier": inference_metrics.get("validation_tier"),
        "verification_sample_overlap_count_exact": inference_metrics.get("verification_sample_overlap_count_exact"),
        "verification_sample_overlap_count_canonical": inference_metrics.get("verification_sample_overlap_count_canonical"),
        "baseline_d0_age_order": inference_metrics.get("baseline_d0_age_order"),
    }


def step_write_mouse_post_summary(outdir: Path, summary: Mapping[str, Any]) -> None:
    with open(outdir / "post_summary.json", "w", encoding="utf-8") as f:
        json.dump(dict(summary), f, indent=2)


def run_mouse_post(
    paths: MousePostPaths = POST_PATHS,
    config: MousePostConfig = POST_CONFIG,
) -> dict[str, Any]:
    outdir, gene_weights = step_load_mouse_post_inputs(paths)

    summary: dict[str, Any] = {
        "artifact_dir": str(paths.artifact_dir),
        "outdir": str(outdir),
        "ran_grn": True,
        "ran_enrichment": True,
        "ran_verification": True,
    }

    print("[1/3] Running GRN analysis")
    summary["grn"] = run_grn(paths, outdir, gene_weights, config)

    print("[2/3] Running enrichment analysis")
    summary["enrichment"] = run_enrichment(paths, outdir, gene_weights)

    print("[3/3] Running external verification apply")
    summary["verification"] = run_verification_apply(paths, outdir)

    step_write_mouse_post_summary(outdir, summary)

    print("Post-analysis complete.")
    print(f"Outputs: {outdir}")
    return summary
