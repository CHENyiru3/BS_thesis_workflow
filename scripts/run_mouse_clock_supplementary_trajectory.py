#!/usr/bin/env python3
"""
Generate supplementary trajectory figures for the baseline mouse clock.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from path_config import ARTIFACT_ROOT, MOUSE_VERIFICATION_DIR, PROJECT_ROOT


ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT / "scripts"))

import run_verification_clock as verification_workflow  # noqa: E402


MODEL_PATH = ARTIFACT_ROOT / "clock_artifacts" / "0310_results" / "expression_only" / "final_model.joblib"
MODEL_GENES_PATH = ARTIFACT_ROOT / "clock_artifacts" / "0310_results" / "expression_only" / "model_genes.txt"
TRAINING_METRICS_PATH = ARTIFACT_ROOT / "clock_artifacts" / "0310_results" / "expression_only" / "training_metrics.json"
INPUT_H5AD = MOUSE_VERIFICATION_DIR / "Myo_Aged_SkM_mm10_v1-1_MuSC.h5ad"
OUTDIR = ARTIFACT_ROOT / "clock_artifacts" / "0310_results" / "expression_only" / "supplementary_trajectory"

YOUNG_COLOR = "#4F7A5A"
OLD_COLOR = "#C65D3B"
NEUTRAL_COLOR = "#2F4858"
N_PSEUDOTIME_BINS = 25


def _validate_paths() -> None:
    required = {
        "baseline model": MODEL_PATH,
        "baseline model genes": MODEL_GENES_PATH,
        "baseline training metrics": TRAINING_METRICS_PATH,
        "verification input": INPUT_H5AD,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n- " + "\n- ".join(missing))


def _load_filtered_cell_scores(outdir: Path) -> pd.DataFrame:
    cell_scores = pd.read_csv(outdir / "cell_scores.tsv", sep="\t")
    keep = cell_scores["age_group_std"].astype(str).isin(["young", "old"])
    cell_scores = cell_scores.loc[keep].copy()
    cell_scores["pseudotime_std"] = pd.to_numeric(cell_scores["pseudotime_std"], errors="coerce")
    cell_scores["p_old"] = pd.to_numeric(cell_scores["p_old"], errors="coerce")
    cell_scores = cell_scores.dropna(subset=["pseudotime_std", "p_old"])
    if cell_scores.empty:
        raise ValueError("No young/old cells with valid pseudotime and p_old remain after filtering.")
    return cell_scores


def _assign_shared_pseudotime_bins(cell_scores: pd.DataFrame, n_bins: int = N_PSEUDOTIME_BINS) -> pd.DataFrame:
    work = cell_scores.copy()
    work["pseudotime_bin"] = pd.qcut(
        work["pseudotime_std"],
        q=int(n_bins),
        labels=False,
        duplicates="drop",
    )
    work = work.dropna(subset=["pseudotime_bin"]).copy()
    work["pseudotime_bin"] = work["pseudotime_bin"].astype(int)
    return work


def build_trajectory_age_proportion(cell_scores: pd.DataFrame) -> pd.DataFrame:
    work = _assign_shared_pseudotime_bins(cell_scores)
    counts = (
        work.groupby(["pseudotime_bin", "age_group_std"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["young", "old"]:
        if col not in counts.columns:
            counts[col] = 0
    counts["n_total"] = counts["young"] + counts["old"]
    counts["prop_young"] = counts["young"] / counts["n_total"].clip(lower=1)
    counts["prop_old"] = counts["old"] / counts["n_total"].clip(lower=1)
    centers = (
        work.groupby("pseudotime_bin")
        .agg(mean_pseudotime=("pseudotime_std", "mean"))
        .reset_index()
    )
    out = counts.merge(centers, on="pseudotime_bin", how="left").sort_values("mean_pseudotime").reset_index(drop=True)
    return out[["pseudotime_bin", "mean_pseudotime", "young", "old", "n_total", "prop_young", "prop_old"]]


def build_trajectory_score_summary(cell_scores: pd.DataFrame) -> pd.DataFrame:
    work = _assign_shared_pseudotime_bins(cell_scores)
    out = (
        work.groupby("pseudotime_bin")
        .agg(
            n_cells=("p_old", "size"),
            mean_p_old=("p_old", "mean"),
            median_p_old=("p_old", "median"),
            mean_pseudotime=("pseudotime_std", "mean"),
            std_p_old=("p_old", "std"),
        )
        .reset_index()
        .sort_values("mean_pseudotime")
        .reset_index(drop=True)
    )
    out["std_p_old"] = out["std_p_old"].fillna(0.0)
    out["se_p_old"] = out["std_p_old"] / np.sqrt(out["n_cells"].clip(lower=1))
    out["ci95_low"] = out["mean_p_old"] - 1.96 * out["se_p_old"]
    out["ci95_high"] = out["mean_p_old"] + 1.96 * out["se_p_old"]
    return out


def build_trajectory_score_by_age(cell_scores: pd.DataFrame) -> pd.DataFrame:
    work = _assign_shared_pseudotime_bins(cell_scores)
    out = (
        work.groupby(["pseudotime_bin", "age_group_std"])
        .agg(
            n_cells=("p_old", "size"),
            mean_p_old=("p_old", "mean"),
            median_p_old=("p_old", "median"),
            mean_pseudotime=("pseudotime_std", "mean"),
            std_p_old=("p_old", "std"),
        )
        .reset_index()
        .sort_values(["age_group_std", "mean_pseudotime"])
        .reset_index(drop=True)
    )
    out["std_p_old"] = out["std_p_old"].fillna(0.0)
    out["se_p_old"] = out["std_p_old"] / np.sqrt(out["n_cells"].clip(lower=1))
    out["ci95_low"] = out["mean_p_old"] - 1.96 * out["se_p_old"]
    out["ci95_high"] = out["mean_p_old"] + 1.96 * out["se_p_old"]
    return out


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_age_proportion_curve(df: pd.DataFrame, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(df["mean_pseudotime"], df["prop_young"], color=YOUNG_COLOR, linewidth=2.6, label="Young")
    ax.plot(df["mean_pseudotime"], df["prop_old"], color=OLD_COLOR, linewidth=2.6, label="Old")
    ax.fill_between(df["mean_pseudotime"], 0, df["prop_young"], color=YOUNG_COLOR, alpha=0.10)
    ax.fill_between(df["mean_pseudotime"], 0, df["prop_old"], color=OLD_COLOR, alpha=0.08)
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("Cell proportion")
    ax.set_ylim(0, 1.02)
    ax.set_title("Young and Old Cell Proportion Across Pseudotime", fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="upper right")
    return _save_figure(fig, outdir / "01_age_proportion_across_pseudotime_curve.pdf")


def plot_aging_score_curve(df: pd.DataFrame, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(df["mean_pseudotime"], df["mean_p_old"], color=NEUTRAL_COLOR, linewidth=2.8)
    ax.fill_between(df["mean_pseudotime"], df["ci95_low"], df["ci95_high"], color=NEUTRAL_COLOR, alpha=0.18)
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("Mean aging score (p_old)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Aging Score Across Pseudotime", fontweight="bold", pad=12)
    return _save_figure(fig, outdir / "02_aging_score_across_pseudotime_curve.pdf")


def plot_aging_score_by_age_curve(df: pd.DataFrame, outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for age_group, color in [("young", YOUNG_COLOR), ("old", OLD_COLOR)]:
        sub = df.loc[df["age_group_std"] == age_group].copy().sort_values("mean_pseudotime")
        if sub.empty:
            continue
        ax.plot(sub["mean_pseudotime"], sub["mean_p_old"], color=color, linewidth=2.8, label=age_group.capitalize())
        ax.fill_between(sub["mean_pseudotime"], sub["ci95_low"], sub["ci95_high"], color=color, alpha=0.16)
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("Mean aging score (p_old)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Aging Score Across Pseudotime by Age", fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="upper left")
    return _save_figure(fig, outdir / "03_aging_score_by_age_across_pseudotime_curve.pdf")


def main() -> None:
    _validate_paths()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    inference_metrics = verification_workflow.run_verification_clock(
        model_path=MODEL_PATH,
        model_genes_path=MODEL_GENES_PATH,
        input_h5ad=INPUT_H5AD,
        outdir=OUTDIR,
        training_metrics_path=TRAINING_METRICS_PATH,
    )

    cell_scores = _load_filtered_cell_scores(OUTDIR)
    age_prop = build_trajectory_age_proportion(cell_scores)
    traj_summary = build_trajectory_score_summary(cell_scores)
    traj_by_age = build_trajectory_score_by_age(cell_scores)

    age_prop.to_csv(OUTDIR / "trajectory_age_proportion.tsv", sep="\t", index=False)
    traj_summary.to_csv(OUTDIR / "trajectory_summary.tsv", sep="\t", index=False)
    traj_by_age.to_csv(OUTDIR / "trajectory_age_separate_summary.tsv", sep="\t", index=False)

    outputs = [
        plot_age_proportion_curve(age_prop, OUTDIR),
        plot_aging_score_curve(traj_summary, OUTDIR),
        plot_aging_score_by_age_curve(traj_by_age, OUTDIR),
    ]

    with open(OUTDIR / "supplementary_trajectory_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_path": str(MODEL_PATH),
                "model_genes_path": str(MODEL_GENES_PATH),
                "training_metrics_path": str(TRAINING_METRICS_PATH),
                "input_h5ad": str(INPUT_H5AD),
                "output_dir": str(OUTDIR),
                "n_cells_young_old_only": int(cell_scores.shape[0]),
                "n_bins": int(age_prop["pseudotime_bin"].nunique()),
                "verification_inference_metrics": inference_metrics,
                "figure_paths": [str(path) for path in outputs],
            },
            handle,
            indent=2,
        )

    print("Supplementary trajectory analysis complete.")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
