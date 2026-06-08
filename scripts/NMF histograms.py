#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exploratory plots and correlations for MS NMF phenotype modules.

Purpose:
    1. Read patient-level MS phenotype data.
    2. Plot histograms for dominant module and selected continuous variables.
    3. Create Pearson and Spearman correlation matrices.
    4. Write plots, correlation matrices, and summary statistics to files.

Example:
    python exploratory_nmf_summary.py \
        --input data/ms_patient_summary.csv \
        --output-dir figures
"""

from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================================
# Column configuration
# ==========================================================

EFFECTIVE_MODULES_COL = "effective_modules"
DOMINANT_MODULE_COL = "dominant_module"
DOMINANT_MODULE_PCT_COL = "dominant_module_pct"
TOTAL_PHENOTYPE_BURDEN_COL = "total_phenotype_burden"
NOTE_COUNT_COL = "note_count"
TOTAL_DAYS_COL = "t_days"
DISEASE_BURDEN_COL = "disease_burden"

MODULE_NAME_MAP = {
    "Sensory-visual": "Sensory-visual-pain",
    "Bowel-bladder": "Autonomic-bowel-bladder",
    "Spastic-ataxic": "Ataxic-spastic-falls",
    "Cognitive-psych": "Cognitive-psychologic-fatigue",
}


# ==========================================================
# Command-line arguments
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create histograms and correlation matrices for MS NMF phenotype modules."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input patient-level CSV file."
    )

    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Directory where output files will be saved."
    )

    return parser.parse_args()


# ==========================================================
# Plotting functions
# ==========================================================

def save_histogram(
    dataframe,
    column,
    output_dir,
    filename,
    title=None,
    xlabel=None,
    bins=30,
):
    if column not in dataframe.columns:
        print(f"Skipping {column}: column not found.")
        return

    series = dataframe[column].dropna()

    plt.figure(figsize=(8, 6))
    plt.hist(series, bins=bins, edgecolor="black")
    plt.title(title or f"Histogram of {column}")
    plt.xlabel(xlabel or column)
    plt.ylabel("Number of patients")
    plt.tight_layout()

    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved histogram: {output_path}")


def save_correlation_heatmap(corr_matrix, output_dir, filename, title):
    plt.figure(figsize=(9, 8))

    ax = sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Correlation"},
        annot_kws={"fontsize": 14, "fontweight": "bold"},
    )

    for text in ax.texts:
        value = float(text.get_text())

        if abs(value) >= 0.50:
            text.set_color("white")
        else:
            text.set_color("black")

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=12, fontweight="bold")
    plt.yticks(rotation=0, fontsize=12, fontweight="bold")

    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=12)
    colorbar.set_label("Correlation", fontsize=12, fontweight="bold")

    plt.tight_layout()

    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved correlation heatmap: {output_path}")


# ==========================================================
# Main
# ==========================================================

def main():
    args = parse_args()

    input_file = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)

    print("\nDataframe information:")
    print(df.info())

    print("\nFirst few rows:")
    print(df.head())

    required_cols = [
        DOMINANT_MODULE_COL,
        TOTAL_PHENOTYPE_BURDEN_COL,
        DISEASE_BURDEN_COL,
        EFFECTIVE_MODULES_COL,
        DOMINANT_MODULE_PCT_COL,
        NOTE_COUNT_COL,
        TOTAL_DAYS_COL,
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["dominant_module_clean"] = df[DOMINANT_MODULE_COL].replace(MODULE_NAME_MAP)

    dominant_counts = (
        df["dominant_module_clean"]
        .value_counts()
        .rename_axis("dominant_module")
        .reset_index(name="count")
    )

    dominant_counts["percentage"] = (
        100 * dominant_counts["count"] / dominant_counts["count"].sum()
    )

    dominant_counts_file = output_dir / "dominant_module_counts.csv"
    dominant_counts.to_csv(dominant_counts_file, index=False)

    print("\nDominant module counts:")
    print(dominant_counts)
    print(f"\nSaved dominant module counts: {dominant_counts_file}")

    plt.figure(figsize=(8, 6))
    plt.bar(
        dominant_counts["dominant_module"],
        dominant_counts["count"],
        edgecolor="black",
    )
    plt.title("Dominant phenotype module")
    plt.xlabel("Dominant module")
    plt.ylabel("Number of patients")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    dominant_plot_file = output_dir / "dominant_module_histogram.png"
    plt.savefig(dominant_plot_file, dpi=300)
    plt.close()

    print(f"Saved dominant module plot: {dominant_plot_file}")

    save_histogram(
        df,
        TOTAL_PHENOTYPE_BURDEN_COL,
        output_dir,
        "total_phenotype_burden_histogram.png",
        title="Total phenotype burden",
        xlabel="Total phenotype burden",
        bins=17,
    )

    save_histogram(
        df,
        NOTE_COUNT_COL,
        output_dir,
        "note_count_histogram.png",
        title="Number of notes per patient",
        xlabel="Number of notes",
        bins=30,
    )

    save_histogram(
        df,
        DISEASE_BURDEN_COL,
        output_dir,
        "disease_burden_histogram.png",
        title="Disease burden",
        xlabel="Disease burden",
        bins=30,
    )

    save_histogram(
        df,
        TOTAL_DAYS_COL,
        output_dir,
        "total_days_histogram.png",
        title="Total days of observation",
        xlabel="Days from first to last note",
        bins=30,
    )

    save_histogram(
        df,
        EFFECTIVE_MODULES_COL,
        output_dir,
        "effective_modules_histogram.png",
        title="Effective number of phenotype modules",
        xlabel="Effective number of phenotype modules",
        bins=30,
    )

    save_histogram(
        df,
        DOMINANT_MODULE_PCT_COL,
        output_dir,
        "dominant_module_pct_histogram.png",
        title="Dominant module percentage",
        xlabel="Dominant module percentage",
        bins=30,
    )

    plt.figure(figsize=(9, 6))

    plt.hist(
        df[EFFECTIVE_MODULES_COL].dropna(),
        bins=30,
        edgecolor="black",
    )

    plt.axvline(1, linestyle="--", linewidth=2, label="1 module")
    plt.axvline(2, linestyle="--", linewidth=2, label="2 modules")
    plt.axvline(3, linestyle="--", linewidth=2, label="3 modules")
    plt.axvline(4, linestyle=":", linewidth=2, label="4 modules")

    plt.title("Effective number of phenotype modules")
    plt.xlabel("Effective number of phenotype modules")
    plt.ylabel("Number of patients")
    plt.legend()
    plt.tight_layout()

    effective_modules_plot_file = (
        output_dir / "effective_modules_histogram_with_reference_lines.png"
    )

    plt.savefig(effective_modules_plot_file, dpi=300)
    plt.close()

    print(f"Saved effective modules plot: {effective_modules_plot_file}")

    correlation_cols = [
        DOMINANT_MODULE_PCT_COL,
        TOTAL_PHENOTYPE_BURDEN_COL,
        DISEASE_BURDEN_COL,
        EFFECTIVE_MODULES_COL,
        NOTE_COUNT_COL,
        TOTAL_DAYS_COL,
    ]

    corr_df = df[correlation_cols].copy()

    pearson_corr = corr_df.corr(method="pearson")
    spearman_corr = corr_df.corr(method="spearman")

    pearson_file = output_dir / "pearson_correlation_matrix.csv"
    spearman_file = output_dir / "spearman_correlation_matrix.csv"

    pearson_corr.to_csv(pearson_file)
    spearman_corr.to_csv(spearman_file)

    print(f"\nSaved Pearson correlation matrix: {pearson_file}")
    print(f"Saved Spearman correlation matrix: {spearman_file}")

    print("\nPearson correlation matrix:")
    print(pearson_corr.round(3))

    print("\nSpearman correlation matrix:")
    print(spearman_corr.round(3))

    save_correlation_heatmap(
        pearson_corr,
        output_dir,
        "pearson_correlation_heatmap.png",
        "Pearson correlation matrix",
    )

    save_correlation_heatmap(
        spearman_corr,
        output_dir,
        "spearman_correlation_heatmap.png",
        "Spearman correlation matrix",
    )

    summary_stats = df[correlation_cols].describe().T
    summary_stats_file = output_dir / "summary_statistics.csv"
    summary_stats.to_csv(summary_stats_file)

    print(f"\nSaved summary statistics: {summary_stats_file}")
    print("\nDone.")


if __name__ == "__main__":
    main()

