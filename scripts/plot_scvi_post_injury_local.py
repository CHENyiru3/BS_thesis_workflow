#!/usr/bin/env python3
"""
Summarize post-injury local pseudotime clock outputs and generate figures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from path_config import ARTIFACT_ROOT, PROJECT_ROOT

ROOT = PROJECT_ROOT
RUN_ROOT = ARTIFACT_ROOT / "scvi_reference_verify_post_injury_local"
SUMMARY_DIR = RUN_ROOT / "summary" / "atlas_fixed"
BASELINE_DIR = ARTIFACT_ROOT / "clock_artifacts" / "0310_results" / "expression_only" / "supplementary_trajectory"

YOUNG_COLOR = "#4F7A5A"
OLD_COLOR = "#C65D3B"
GERIATRIC_COLOR = "#7A4C5D"
NEUTRAL_COLOR = "#2F4858"
TIME_ORDER = ["d0", "d1", "d2", "d3.5", "d5", "d7"]
AGE_COLORS = {
    "young": YOUNG_COLOR,
    "old": OLD_COLOR,
    "geriatric": GERIATRIC_COLOR,
}
AGE_LABELS = {
    "young": "Young",
    "old": "Old",
    "geriatric": "26mo",
}


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def load_sample_summary() -> pd.DataFrame:
    summary = pd.read_csv(SUMMARY_DIR / "sample_summary.tsv", sep="\t")
    summary["sample_id_std"] = summary["sample_id_std"].astype(str)
    summary["age_group_std"] = summary["age_group_std"].astype(str)
    summary["timepoint_std"] = summary["timepoint_std"].astype(str)
    return summary


def load_metacell_scores(sample_summary: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for sample_id in sample_summary["sample_id_std"].astype(str):
        path = RUN_ROOT / sample_id / "query_metacell_scores.tsv"
        if not path.exists():
            raise FileNotFoundError(f"Missing metacell score file: {path}")
        frame = pd.read_csv(path, sep="\t")
        frame["sample_id_std"] = frame["sample_id_std"].astype(str)
        frames.append(frame)
    metacells = pd.concat(frames, ignore_index=True)
    meta = sample_summary[
        [
            "sample_id_std",
            "age_group_std",
            "timepoint_std",
            "group_std",
            "condition_std",
            "n_query_cells",
            "n_query_metacells",
        ]
    ].drop_duplicates()
    return metacells.merge(meta, on=["sample_id_std", "age_group_std"], how="left", validate="many_to_one")


def build_sample_scores(sample_summary: pd.DataFrame, metacells: pd.DataFrame) -> pd.DataFrame:
    by_sample = (
        metacells.groupby(["sample_id_std", "age_group_std", "timepoint_std"], as_index=False)
        .agg(
            mean_p_old=("p_old", "mean"),
            median_p_old=("p_old", "median"),
            young_like_fraction=("pred_old_thr_local", lambda x: float(np.mean(1 - np.asarray(x, dtype=float)))),
            old_like_fraction=("pred_old_thr_local", lambda x: float(np.mean(np.asarray(x, dtype=float)))),
            mean_pseudotime=("reference_pseudotime_transferred_mean", "mean"),
            mean_local_threshold=("local_threshold", "mean"),
            mean_transfer_confidence=("reference_pseudotime_transfer_confidence_mean", "mean"),
            n_metacells=("p_old", "size"),
        )
    )
    out = by_sample.merge(
        sample_summary[
            [
                "sample_id_std",
                "age_group_std",
                "timepoint_std",
                "group_std",
                "condition_std",
                "n_query_cells",
            ]
        ].drop_duplicates(),
        on=["sample_id_std", "age_group_std", "timepoint_std"],
        how="left",
        validate="one_to_one",
    )
    out = out.rename(columns={"n_query_cells": "n_cells"})
    out["injury_status_std"] = np.where(out["timepoint_std"].eq("d0"), "uninjured", "injured")
    out["analysis_mode"] = "sample_level"
    out["validation_tier"] = "scvi_reference_local_clock"
    cols = [
        "sample_id_std",
        "age_group_std",
        "timepoint_std",
        "injury_status_std",
        "n_cells",
        "n_metacells",
        "mean_p_old",
        "median_p_old",
        "young_like_fraction",
        "old_like_fraction",
        "mean_pseudotime",
        "mean_local_threshold",
        "mean_transfer_confidence",
        "analysis_mode",
        "validation_tier",
    ]
    return out[cols].sort_values(["age_group_std", "timepoint_std", "sample_id_std"]).reset_index(drop=True)


def build_injury_summary_by_sample(sample_scores: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        sample_scores.groupby(["age_group_std", "timepoint_std"], as_index=False)
        .agg(
            n_samples=("sample_id_std", "size"),
            total_cells=("n_cells", "sum"),
            total_metacells=("n_metacells", "sum"),
            mean_sample_p_old=("mean_p_old", "mean"),
            median_sample_p_old=("mean_p_old", "median"),
            mean_sample_young_like_fraction=("young_like_fraction", "mean"),
            mean_sample_pseudotime=("mean_pseudotime", "mean"),
            mean_sample_local_threshold=("mean_local_threshold", "mean"),
            mean_transfer_confidence=("mean_transfer_confidence", "mean"),
        )
    )
    grouped["timepoint_std"] = pd.Categorical(grouped["timepoint_std"], categories=TIME_ORDER, ordered=True)
    grouped = grouped.sort_values(["age_group_std", "timepoint_std"]).reset_index(drop=True)
    d0 = grouped.loc[grouped["timepoint_std"] == "d0", ["age_group_std", "mean_sample_p_old"]].rename(
        columns={"mean_sample_p_old": "mean_sample_p_old_d0"}
    )
    grouped = grouped.merge(d0, on="age_group_std", how="left")
    grouped["delta_mean_sample_vs_d0"] = grouped["mean_sample_p_old"] - grouped["mean_sample_p_old_d0"]
    grouped["analysis_mode"] = "sample_level"
    grouped["validation_tier"] = "scvi_reference_local_clock"
    grouped["timepoint_std"] = grouped["timepoint_std"].astype(str)
    return grouped


def build_injury_summary(metacells: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        metacells.groupby(["age_group_std", "timepoint_std"], as_index=False)
        .agg(
            n_samples=("sample_id_std", "nunique"),
            n_metacells=("p_old", "size"),
            mean_p_old=("p_old", "mean"),
            median_p_old=("p_old", "median"),
            mean_local_threshold=("local_threshold", "mean"),
            mean_pseudotime=("reference_pseudotime_transferred_mean", "mean"),
            mean_transfer_confidence=("reference_pseudotime_transfer_confidence_mean", "mean"),
        )
    )
    grouped["analysis_mode"] = "metacell_level"
    grouped["validation_tier"] = "scvi_reference_local_clock"
    grouped["timepoint_std"] = pd.Categorical(grouped["timepoint_std"], categories=TIME_ORDER, ordered=True)
    grouped = grouped.sort_values(["age_group_std", "timepoint_std"]).reset_index(drop=True)
    grouped["timepoint_std"] = grouped["timepoint_std"].astype(str)
    return grouped


def plot_post_injury_p_old_three_curves(sample_scores: pd.DataFrame, summary: pd.DataFrame) -> Path:
    x_map = {tp: i for i, tp in enumerate(TIME_ORDER)}
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for age in ["young", "old", "geriatric"]:
        color = AGE_COLORS[age]
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
            label=AGE_LABELS[age],
        )

        sub_pts = sample_scores.loc[sample_scores["age_group_std"] == age].copy()
        sub_pts["x"] = sub_pts["timepoint_std"].map(x_map).astype(float)
        offsets = np.linspace(-0.12, 0.12, max(1, len(sub_pts)))
        sub_pts = sub_pts.sort_values(["timepoint_std", "sample_id_std"]).reset_index(drop=True)
        sub_pts["x_jitter"] = sub_pts["x"] + np.resize(offsets, len(sub_pts))
        ax.scatter(sub_pts["x_jitter"], sub_pts["mean_p_old"], s=36, color=color, alpha=0.45, linewidths=0)

    ax.set_xticks(range(len(TIME_ORDER)))
    ax.set_xticklabels(TIME_ORDER)
    ax.set_xlabel("Post-injury time")
    ax.set_ylabel("Predicted p_old")
    ax.set_ylim(0, 1.02)
    ax.set_title("Pseudotime-aware clock across post-injury time", fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="best")
    return _save_figure(fig, SUMMARY_DIR / "03_post_injury_p_old_three_curves_local.pdf")


def plot_post_injury_p_old_by_age_panels(sample_scores: pd.DataFrame, summary: pd.DataFrame) -> Path:
    x_map = {tp: i for i, tp in enumerate(TIME_ORDER)}
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 9.2), sharex=True)
    for ax, age in zip(axes, ["young", "old", "geriatric"]):
        color = AGE_COLORS[age]
        sub_sum = summary.loc[summary["age_group_std"] == age].copy()
        sub_sum["x"] = sub_sum["timepoint_std"].map(x_map)
        sub_sum = sub_sum.sort_values("x")
        ax.plot(sub_sum["x"], sub_sum["mean_sample_p_old"], color=color, linewidth=2.8, marker="o", markersize=5.5)

        sub_pts = sample_scores.loc[sample_scores["age_group_std"] == age].copy()
        sub_pts["x"] = sub_pts["timepoint_std"].map(x_map).astype(float)
        offsets = np.linspace(-0.12, 0.12, max(1, len(sub_pts)))
        sub_pts = sub_pts.sort_values(["timepoint_std", "sample_id_std"]).reset_index(drop=True)
        sub_pts["x_jitter"] = sub_pts["x"] + np.resize(offsets, len(sub_pts))
        ax.scatter(sub_pts["x_jitter"], sub_pts["mean_p_old"], s=36, color=color, alpha=0.45, linewidths=0)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("p_old")
        ax.set_title(AGE_LABELS[age], loc="left", fontweight="bold", pad=8)

    axes[-1].set_xticks(range(len(TIME_ORDER)))
    axes[-1].set_xticklabels(TIME_ORDER)
    axes[-1].set_xlabel("Post-injury time")
    fig.suptitle("Pseudotime-aware clock by age group", fontweight="bold", y=0.995)
    fig.tight_layout()
    return _save_figure(fig, SUMMARY_DIR / "04_post_injury_p_old_by_age_panels_local.pdf")


def plot_baseline_vs_local_comparison(local_summary: pd.DataFrame) -> Path:
    baseline = pd.read_csv(BASELINE_DIR / "injury_summary_by_sample.tsv", sep="\t")
    baseline["timepoint_std"] = baseline["timepoint_std"].astype(str)
    baseline["age_group_std"] = baseline["age_group_std"].astype(str)
    local_summary = local_summary.copy()
    local_summary["timepoint_std"] = local_summary["timepoint_std"].astype(str)

    x_map = {tp: i for i, tp in enumerate(TIME_ORDER)}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=True)
    for ax, title, frame, value_col in [
        (axes[0], "Baseline direct apply", baseline, "mean_sample_p_old"),
        (axes[1], "Local clock via scArches", local_summary, "mean_sample_p_old"),
    ]:
        for age in ["young", "old", "geriatric"]:
            sub = frame.loc[frame["age_group_std"] == age].copy()
            sub["x"] = sub["timepoint_std"].map(x_map)
            sub = sub.sort_values("x")
            ax.plot(sub["x"], sub[value_col], color=AGE_COLORS[age], linewidth=2.6, marker="o", markersize=5.2, label=AGE_LABELS[age])
        ax.set_xticks(range(len(TIME_ORDER)))
        ax.set_xticklabels(TIME_ORDER)
        ax.set_xlabel("Post-injury time")
        ax.set_title(title, fontweight="bold", pad=10)
    axes[0].set_ylabel("Mean donor p_old")
    axes[0].set_ylim(0, 1.02)
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle("Baseline vs pseudotime-aware post-injury comparison", fontweight="bold", y=0.995)
    fig.tight_layout()
    return _save_figure(fig, SUMMARY_DIR / "05_baseline_vs_local_post_injury_comparison.pdf")


def plot_shared_axis_support(metacells: pd.DataFrame) -> Path:
    work = metacells.copy()
    work["pt_bin"] = pd.qcut(work["reference_pseudotime_transferred_mean"], q=20, labels=False, duplicates="drop")
    work = work.dropna(subset=["pt_bin"]).copy()
    work["pt_bin"] = work["pt_bin"].astype(int)
    summary = (
        work.groupby(["age_group_std", "pt_bin"], as_index=False)
        .agg(
            mean_pseudotime=("reference_pseudotime_transferred_mean", "mean"),
            mean_p_old=("p_old", "mean"),
            std_p_old=("p_old", "std"),
            n_metacells=("p_old", "size"),
        )
    )
    summary["std_p_old"] = summary["std_p_old"].fillna(0.0)
    summary["se_p_old"] = summary["std_p_old"] / np.sqrt(summary["n_metacells"].clip(lower=1))
    summary["ci95_low"] = summary["mean_p_old"] - 1.96 * summary["se_p_old"]
    summary["ci95_high"] = summary["mean_p_old"] + 1.96 * summary["se_p_old"]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for age in ["young", "old", "geriatric"]:
        sub = summary.loc[summary["age_group_std"] == age].copy().sort_values("mean_pseudotime")
        ax.plot(sub["mean_pseudotime"], sub["mean_p_old"], color=AGE_COLORS[age], linewidth=2.6, label=AGE_LABELS[age])
        ax.fill_between(sub["mean_pseudotime"], sub["ci95_low"], sub["ci95_high"], color=AGE_COLORS[age], alpha=0.12)
    ax.set_xlabel("Transferred reference pseudotime")
    ax.set_ylabel("Mean metacell p_old")
    ax.set_ylim(0, 1.02)
    ax.set_title("Local clock on shared reference pseudotime", fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="best")
    summary.to_csv(SUMMARY_DIR / "shared_axis_age_group_summary.tsv", sep="\t", index=False)
    return _save_figure(fig, SUMMARY_DIR / "06_shared_axis_p_old_by_age_group_local.pdf")


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    sample_summary = load_sample_summary()
    metacells = load_metacell_scores(sample_summary)
    sample_scores = build_sample_scores(sample_summary, metacells)
    injury_summary_by_sample = build_injury_summary_by_sample(sample_scores)
    injury_summary = build_injury_summary(metacells)

    sample_scores.to_csv(SUMMARY_DIR / "sample_scores.tsv", sep="\t", index=False)
    injury_summary_by_sample.to_csv(SUMMARY_DIR / "injury_summary_by_sample.tsv", sep="\t", index=False)
    injury_summary.to_csv(SUMMARY_DIR / "injury_summary.tsv", sep="\t", index=False)

    outputs = [
        SUMMARY_DIR / "sample_scores.tsv",
        SUMMARY_DIR / "injury_summary_by_sample.tsv",
        SUMMARY_DIR / "injury_summary.tsv",
        plot_post_injury_p_old_three_curves(sample_scores, injury_summary_by_sample),
        plot_post_injury_p_old_by_age_panels(sample_scores, injury_summary_by_sample),
        plot_baseline_vs_local_comparison(injury_summary_by_sample),
        plot_shared_axis_support(metacells),
    ]
    print("Local post-injury scVI/scArches summary complete.")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
