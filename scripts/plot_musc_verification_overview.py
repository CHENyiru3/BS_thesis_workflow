#!/usr/bin/env python3
"""
Create compact donor/sample overview figures for the MuSC verification dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from path_config import ARTIFACT_ROOT, MOUSE_VERIFICATION_DIR, PROJECT_ROOT

ROOT = PROJECT_ROOT
INPUT_H5AD = MOUSE_VERIFICATION_DIR / "Myo_Aged_SkM_mm10_v1-1_MuSC.h5ad"
OUTDIR = ARTIFACT_ROOT / "figures" / "verification_overview"
SUPP_TRAJ_DIR = ARTIFACT_ROOT / "clock_artifacts" / "0310_results" / "expression_only" / "supplementary_trajectory"

YOUNG_COLOR = "#4F7A5A"
OLD_COLOR = "#C65D3B"
GERIATRIC_COLOR = "#7A4C5D"
NEUTRAL_COLOR = "#2F4858"
TIMEPOINT_PALETTE = {
    "d0": "#4F7A5A",
    "d1": "#D17C48",
    "d2": "#C65D3B",
    "d3.5": "#8F4C7A",
    "d5": "#4C78A8",
    "d7": "#2F4858",
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


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def load_dataset() -> pd.DataFrame:
    adata = ad.read_h5ad(INPUT_H5AD)
    obs = adata.obs.copy()
    obs["sample_id_std"] = obs["Sample.ID"].astype(str)
    obs["age_group_std"] = obs["Age.Word"].map(map_age_group).astype(str)
    obs["timepoint_std"] = obs["Time.Point"].map(map_timepoint).astype(str)
    obs["pseudotime_std"] = pd.to_numeric(obs["Pseudotime"], errors="coerce")
    return obs


def plot_sample_cellcount_heatmap(obs: pd.DataFrame) -> Path:
    age_counts = obs["age_group_std"].value_counts().reindex(["young", "old", "geriatric"]).fillna(0)
    time_counts = obs["timepoint_std"].value_counts().reindex(["d0", "d1", "d2", "d3.5", "d5", "d7"]).fillna(0)

    age_colors = [YOUNG_COLOR, OLD_COLOR, GERIATRIC_COLOR]
    time_colors = [TIMEPOINT_PALETTE[tp] for tp in ["d0", "d1", "d2", "d3.5", "d5", "d7"]]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.8))
    axes[0].pie(
        age_counts.values,
        labels=["Young", "Old", "26mo"],
        colors=age_colors,
        autopct=lambda p: f"{p:.1f}%" if p >= 5 else "",
        startangle=90,
        textprops={"fontsize": 10},
        wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
    )
    axes[0].set_title("Cells by age group", fontweight="bold", pad=12)

    axes[1].pie(
        time_counts.values,
        labels=["d0", "d1", "d2", "d3.5", "d5", "d7"],
        colors=time_colors,
        autopct=lambda p: f"{p:.1f}%" if p >= 5 else "",
        startangle=90,
        textprops={"fontsize": 10},
        wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
    )
    axes[1].set_title("Cells by timepoint", fontweight="bold", pad=12)

    fig.suptitle("MuSC verification dataset: cell-count overview", fontweight="bold", y=0.98)
    fig.tight_layout()
    return _save_figure(fig, OUTDIR / "01_sample_cellcount_overview_pies.pdf")


def plot_donor_pseudotime_distribution(obs: pd.DataFrame) -> Path:
    plot_df = obs.dropna(subset=["pseudotime_std"]).copy()
    donor_meta = (
        plot_df.groupby("sample_id_std")
        .agg(
            age_group_std=("age_group_std", lambda x: x.iloc[0]),
            timepoint_std=("timepoint_std", lambda x: x.iloc[0]),
            mean_pt=("pseudotime_std", "mean"),
            n_cells=("sample_id_std", "size"),
        )
        .reset_index()
    )
    summary = (
        donor_meta.groupby(["age_group_std", "timepoint_std"])
        .agg(
            n_donors=("sample_id_std", "size"),
            mean_pt=("mean_pt", "mean"),
            q25_pt=("mean_pt", lambda x: float(np.quantile(x, 0.25))),
            q75_pt=("mean_pt", lambda x: float(np.quantile(x, 0.75))),
        )
        .reset_index()
    )
    age_rank = {"young": 0, "old": 1, "geriatric": 2}
    time_order = ["d0", "d1", "d2", "d3.5", "d5", "d7"]
    time_rank = {tp: i for i, tp in enumerate(time_order)}
    summary["y_base"] = summary["timepoint_std"].map(time_rank).astype(float)
    offsets = {"young": -0.22, "old": 0.0, "geriatric": 0.22}
    age_colors = {"young": YOUNG_COLOR, "old": OLD_COLOR, "geriatric": GERIATRIC_COLOR}
    summary["y"] = summary["y_base"] + summary["age_group_std"].map(offsets).astype(float)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for _, row in summary.sort_values(["y_base", "age_group_std"], key=lambda s: s.map(age_rank) if s.name == "age_group_std" else s).iterrows():
        color = age_colors[row["age_group_std"]]
        ax.hlines(
            y=row["y"],
            xmin=row["q25_pt"],
            xmax=row["q75_pt"],
            color=color,
            linewidth=2.4,
            alpha=0.85,
        )
        ax.scatter(
            row["mean_pt"],
            row["y"],
            s=55 + (row["n_donors"] * 10),
            color=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=age_colors[age], label=label, markersize=8)
        for age, label in [("young", "Young"), ("old", "Old"), ("geriatric", "26mo")]
    ]
    ax.legend(handles=handles, title="Age group", frameon=False, loc="lower right")
    ax.set_yticks(range(len(time_order)))
    ax.set_yticklabels(time_order)
    ax.set_xlabel("Pseudotime")
    ax.set_ylabel("Timepoint")
    ax.set_title("MuSC verification dataset: mean donor pseudotime by group", fontweight="bold", pad=12)
    return _save_figure(fig, OUTDIR / "02_donor_pseudotime_distribution.pdf")


def write_sample_summary(obs: pd.DataFrame) -> Path:
    summary = (
        obs.groupby(["sample_id_std", "age_group_std", "timepoint_std"])
        .agg(
            n_cells=("sample_id_std", "size"),
            mean_pseudotime=("pseudotime_std", "mean"),
            median_pseudotime=("pseudotime_std", "median"),
        )
        .reset_index()
    )
    out = OUTDIR / "verification_sample_overview.tsv"
    summary.to_csv(out, sep="\t", index=False)
    return out


def load_injury_clock_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    by_sample = pd.read_csv(SUPP_TRAJ_DIR / "sample_scores.tsv", sep="\t")
    summary = pd.read_csv(SUPP_TRAJ_DIR / "injury_summary_by_sample.tsv", sep="\t")
    by_sample["timepoint_std"] = by_sample["timepoint_std"].astype(str)
    by_sample["age_group_std"] = by_sample["age_group_std"].astype(str)
    summary["timepoint_std"] = summary["timepoint_std"].astype(str)
    summary["age_group_std"] = summary["age_group_std"].astype(str)
    return by_sample, summary


def plot_post_injury_p_old_three_curves(sample_scores: pd.DataFrame, summary: pd.DataFrame) -> Path:
    age_colors = {
        "young": YOUNG_COLOR,
        "old": OLD_COLOR,
        "geriatric": GERIATRIC_COLOR,
    }
    age_labels = {
        "young": "Young",
        "old": "Old",
        "geriatric": "26mo",
    }
    time_order = ["d0", "d1", "d2", "d3.5", "d5", "d7"]
    x_map = {tp: i for i, tp in enumerate(time_order)}

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for age in ["young", "old", "geriatric"]:
        sub_sum = summary.loc[summary["age_group_std"] == age].copy()
        sub_sum["x"] = sub_sum["timepoint_std"].map(x_map)
        sub_sum = sub_sum.sort_values("x")
        color = age_colors[age]

        ax.plot(
            sub_sum["x"],
            sub_sum["mean_sample_p_old"],
            color=color,
            linewidth=2.8,
            marker="o",
            markersize=5.5,
            label=age_labels[age],
        )

        sub_pts = sample_scores.loc[sample_scores["age_group_std"] == age].copy()
        sub_pts["x"] = sub_pts["timepoint_std"].map(x_map).astype(float)
        offsets = np.linspace(-0.12, 0.12, max(1, len(sub_pts)))
        sub_pts = sub_pts.sort_values(["timepoint_std", "sample_id_std"]).reset_index(drop=True)
        sub_pts["x_jitter"] = sub_pts["x"] + np.resize(offsets, len(sub_pts))
        ax.scatter(
            sub_pts["x_jitter"],
            sub_pts["mean_p_old"],
            s=36,
            color=color,
            alpha=0.45,
            linewidths=0,
        )

    ax.set_xticks(range(len(time_order)))
    ax.set_xticklabels(time_order)
    ax.set_xlabel("Post-injury time")
    ax.set_ylabel("Predicted p_old")
    ax.set_ylim(0, 1.02)
    ax.set_title("Predicted age across post-injury time", fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="best")
    return _save_figure(fig, OUTDIR / "03_post_injury_p_old_three_curves.pdf")


def plot_post_injury_p_old_by_age_panels(sample_scores: pd.DataFrame, summary: pd.DataFrame) -> Path:
    age_colors = {
        "young": YOUNG_COLOR,
        "old": OLD_COLOR,
        "geriatric": GERIATRIC_COLOR,
    }
    age_labels = {
        "young": "Young",
        "old": "Old",
        "geriatric": "26mo",
    }
    time_order = ["d0", "d1", "d2", "d3.5", "d5", "d7"]
    x_map = {tp: i for i, tp in enumerate(time_order)}

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 9.2), sharex=True)
    for ax, age in zip(axes, ["young", "old", "geriatric"]):
        color = age_colors[age]
        sub_sum = summary.loc[summary["age_group_std"] == age].copy()
        sub_sum["x"] = sub_sum["timepoint_std"].map(x_map)
        sub_sum = sub_sum.sort_values("x")
        ax.plot(
            sub_sum["x"],
            sub_sum["mean_sample_p_old"],
            color=color,
            linewidth=2.8,
            marker="o",
            markersize=5.5,
        )

        sub_pts = sample_scores.loc[sample_scores["age_group_std"] == age].copy()
        sub_pts["x"] = sub_pts["timepoint_std"].map(x_map).astype(float)
        offsets = np.linspace(-0.12, 0.12, max(1, len(sub_pts)))
        sub_pts = sub_pts.sort_values(["timepoint_std", "sample_id_std"]).reset_index(drop=True)
        sub_pts["x_jitter"] = sub_pts["x"] + np.resize(offsets, len(sub_pts))
        ax.scatter(
            sub_pts["x_jitter"],
            sub_pts["mean_p_old"],
            s=36,
            color=color,
            alpha=0.45,
            linewidths=0,
        )
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("p_old")
        ax.set_title(age_labels[age], loc="left", fontweight="bold", pad=8)

    axes[-1].set_xticks(range(len(time_order)))
    axes[-1].set_xticklabels(time_order)
    axes[-1].set_xlabel("Post-injury time")
    fig.suptitle("Predicted age across post-injury time by age group", fontweight="bold", y=0.995)
    fig.tight_layout()
    return _save_figure(fig, OUTDIR / "04_post_injury_p_old_by_age_panels.pdf")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    obs = load_dataset()
    sample_scores, injury_summary = load_injury_clock_summaries()
    outputs = [
        plot_sample_cellcount_heatmap(obs),
        plot_donor_pseudotime_distribution(obs),
        write_sample_summary(obs),
        plot_post_injury_p_old_three_curves(sample_scores, injury_summary),
        plot_post_injury_p_old_by_age_panels(sample_scores, injury_summary),
    ]
    print("Verification overview complete.")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
