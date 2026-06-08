#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:37:28 2026

@author: danielhier
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot effective number of modules by dominant NMF module.

This script creates a box-and-jitter plot showing the entropy-derived
effective number of phenotype modules for each dominant NMF module.

Input:
    A patient-level CSV file containing:
        - dominant_module
        - effective_modules

Output:
    PNG and PDF figures.

Example:
    python scripts/plot_effective_modules_by_dominant_module.py \
        --input data/ms_patient_level_module_scores.csv \
        --output-prefix figures/effective_modules_by_dominant_module
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# Constants
# ==========================================================

DEFAULT_CATEGORY_COL = "dominant_module"
DEFAULT_Y_COL = "effective_modules"

MODULE_ORDER = [
    "Sensory_Visual_Pain",
    "Ataxic_Spastic_Falls",
    "Cognitive_Psychologic_Fatigue",
    "Autonomic_Bladder_Bowel",
]

MODULE_LABELS = [
    "Sensory\nvision\npain",
    "Ataxic\nspastic\nfalls",
    "Cognitive\npsychologic\nfatigue",
    "Autonomic\nbowel\nbladder",
]

# Map older or alternate names to final internal names.
MODULE_NAME_MAP = {
    # Sensory-visual-pain
    "Sensory-visual": "Sensory_Visual_Pain",
    "Sensory-visual-pain": "Sensory_Visual_Pain",
    "Sensory-Visual-Pain": "Sensory_Visual_Pain",
    "Sensory_Pain_Vision": "Sensory_Visual_Pain",
    "Sensory_Visual_Pain": "Sensory_Visual_Pain",

    # Ataxic-spastic-falls
    "Spastic-ataxic": "Ataxic_Spastic_Falls",
    "Ataxic-spastic-falls": "Ataxic_Spastic_Falls",
    "Ataxic-Spastic-Falls": "Ataxic_Spastic_Falls",
    "Motor_Cerebellar_Spasticity": "Ataxic_Spastic_Falls",
    "Ataxic_Spastic_Falls": "Ataxic_Spastic_Falls",

    # Cognitive-psychologic-fatigue
    "Cognitive-psych": "Cognitive_Psychologic_Fatigue",
    "Cognitive-fatigue-psychologic": "Cognitive_Psychologic_Fatigue",
    "Cognitive-psychologic-fatigue": "Cognitive_Psychologic_Fatigue",
    "Cognitive-Psychologic-Fatigue": "Cognitive_Psychologic_Fatigue",
    "Cognitive_Fatigue_Psychologic": "Cognitive_Psychologic_Fatigue",
    "Cognitive_Psychologic_Fatigue": "Cognitive_Psychologic_Fatigue",

    # Autonomic-bladder-bowel
    "Bowel-bladder": "Autonomic_Bladder_Bowel",
    "Autonomic-bladder-bowel": "Autonomic_Bladder_Bowel",
    "Autonomic-Bladder-Bowel": "Autonomic_Bladder_Bowel",
    "Autonomic_Bowel_Bladder": "Autonomic_Bladder_Bowel",
    "Autonomic_Bladder_Bowel": "Autonomic_Bladder_Bowel",
}

MODULE_COLORS = {
    "Sensory_Visual_Pain": "purple",
    "Ataxic_Spastic_Falls": "red",
    "Cognitive_Psychologic_Fatigue": "blue",
    "Autonomic_Bladder_Bowel": "green",
}


# ==========================================================
# Argument Parsing
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create box-and-jitter plot of effective modules by dominant module."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to patient-level CSV file.",
    )

    parser.add_argument(
        "--output-prefix",
        default="figures/effective_modules_by_dominant_module",
        help="Output path prefix without extension.",
    )

    parser.add_argument(
        "--category-col",
        default=DEFAULT_CATEGORY_COL,
        help="Column containing dominant module labels.",
    )

    parser.add_argument(
        "--y-col",
        default=DEFAULT_Y_COL,
        help="Column containing effective number of modules.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="DPI for PNG output.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for jitter reproducibility.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plot interactively.",
    )

    return parser.parse_args()


# ==========================================================
# Data Preparation
# ==========================================================

def load_and_clean_data(input_path, category_col, y_col):
    """
    Load and clean patient-level module data.

    Parameters
    ----------
    input_path : str or pathlib.Path
        Path to CSV file.

    category_col : str
        Column containing dominant module names.

    y_col : str
        Column containing effective number of modules.

    Returns
    -------
    plot_df : pandas.DataFrame
        Cleaned dataframe with dominant_module_clean and y_col.
    """

    df = pd.read_csv(input_path)

    required_cols = [category_col, y_col]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    plot_df = df[[category_col, y_col]].copy()

    # Ensure y values are numeric.
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")

    # Clean and harmonize module names.
    plot_df[category_col] = plot_df[category_col].astype(str).str.strip()
    plot_df["dominant_module_clean"] = plot_df[category_col].map(MODULE_NAME_MAP)

    # Drop rows that cannot be plotted.
    plot_df = plot_df.dropna(subset=["dominant_module_clean", y_col])

    # Keep only expected module names.
    plot_df = plot_df[
        plot_df["dominant_module_clean"].isin(MODULE_ORDER)
    ].copy()

    if plot_df.empty:
        raise ValueError(
            "No rows left to plot. Check dominant module names and MODULE_NAME_MAP."
        )

    return plot_df


def summarize_plot_data(plot_df, y_col):
    """
    Print module counts and effective-module summaries.
    """

    print("\nRows after cleaning:", len(plot_df))

    print("\nCounts by dominant module:")
    print(plot_df["dominant_module_clean"].value_counts().reindex(MODULE_ORDER))

    print("\nEffective modules summary:")
    print(plot_df[y_col].describe())


# ==========================================================
# Plotting
# ==========================================================

def make_effective_modules_plot(
    plot_df,
    y_col,
    output_prefix,
    dpi=600,
    random_state=0,
    show=False,
):
    """
    Create and save box-and-jitter plot.

    Parameters
    ----------
    plot_df : pandas.DataFrame
        Cleaned dataframe.

    y_col : str
        Column containing effective number of modules.

    output_prefix : str or pathlib.Path
        Output file path prefix without extension.

    dpi : int
        DPI for PNG file.

    random_state : int
        Random seed for jitter.

    show : bool
        If True, display the figure interactively.
    """

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(random_state)

    data_by_module = []

    for module in MODULE_ORDER:
        values = plot_df.loc[
            plot_df["dominant_module_clean"] == module,
            y_col,
        ].to_numpy()

        data_by_module.append(values)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Box plot with unfilled boxes.
    ax.boxplot(
        data_by_module,
        positions=np.arange(1, len(MODULE_ORDER) + 1),
        widths=0.55,
        patch_artist=False,
        showfliers=False,
        medianprops={
            "linewidth": 2,
            "color": "black",
        },
        boxprops={
            "linewidth": 1.2,
            "color": "black",
        },
        whiskerprops={
            "linewidth": 1.2,
            "color": "black",
        },
        capprops={
            "linewidth": 1.2,
            "color": "black",
        },
    )

    # Overlay individual patients with horizontal jitter.
    jitter_width = 0.10

    for i, module in enumerate(MODULE_ORDER, start=1):

        y_values = plot_df.loc[
            plot_df["dominant_module_clean"] == module,
            y_col,
        ].to_numpy()

        x_values = rng.normal(
            loc=i,
            scale=jitter_width,
            size=len(y_values),
        )

        x_values = np.clip(x_values, i - 0.22, i + 0.22)

        ax.scatter(
            x_values,
            y_values,
            s=24,
            alpha=0.55,
            color=MODULE_COLORS[module],
            edgecolors="black",
            linewidths=0.25,
            zorder=3,
        )

    # Axis ticks and labels.
    ax.set_xticks(np.arange(1, len(MODULE_ORDER) + 1))

    labels_with_n = []

    for label, module in zip(MODULE_LABELS, MODULE_ORDER):
        n = (plot_df["dominant_module_clean"] == module).sum()
        labels_with_n.append(f"{label}\n(n={n})")

    ax.set_xticklabels(labels_with_n, fontsize=14)

    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", labelsize=14)

    ax.set_ylabel(
        "Effective number of modules",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Dominant NMF module",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_ylim(0.9, 4.1)

    # Reference lines.
    for y_ref in [1, 2, 3, 4]:
        ax.axhline(
            y_ref,
            linestyle="--",
            linewidth=1,
            alpha=0.35,
            zorder=0,
        )

    ax.text(
        4.55,
        1,
        "pure",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    ax.text(
        4.55,
        4,
        "maximally mixed",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    print("\nSaved figures:")
    print(png_path)
    print(pdf_path)

    if show:
        plt.show()

    plt.close(fig)


# ==========================================================
# Main
# ==========================================================

def main():
    args = parse_args()

    plot_df = load_and_clean_data(
        input_path=args.input,
        category_col=args.category_col,
        y_col=args.y_col,
    )

    summarize_plot_data(
        plot_df=plot_df,
        y_col=args.y_col,
    )

    make_effective_modules_plot(
        plot_df=plot_df,
        y_col=args.y_col,
        output_prefix=args.output_prefix,
        dpi=args.dpi,
        random_state=args.random_state,
        show=args.show,
    )


if __name__ == "__main__":
    main()