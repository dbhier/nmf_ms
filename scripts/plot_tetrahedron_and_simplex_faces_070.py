#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:58:17 2026

@author: danielhier
requirements:
pandas
numpy
matplotlib
scikit-learn
python-ternary
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create tetrahedron and simplex visualizations of MS phenotype modules.

This script:
1. Loads patient-level phenotype data.
2. Fits a four-component NMF model.
3. Optionally reorders NMF components into canonical clinical order.
4. Normalizes patient weights so each patient is represented as a mixture
   of four phenotype modules.
5. Creates:
   - one 3D tetrahedron plot
   - two ternary/simplex face plots

Canonical module order:
    0 = sensory-visual-pain
    1 = ataxic-spastic-falls
    2 = cognitive-psychologic-fatigue
    3 = autonomic-bladder-bowel

Example:
    python scripts/plot_tetrahedron_and_simplex_faces.py \
        --input data/ms_patient_level_feature_matrix.csv \
        --output-dir figures \
        --threshold 0.70

To make the >0.70 version:
    python scripts/plot_tetrahedron_and_simplex_faces.py \
        --input data/ms_patient_level_feature_matrix.csv \
        --output-dir figures \
        --threshold 0.70
"""

from pathlib import Path
from itertools import combinations
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF


# ==========================================================
# Defaults
# ==========================================================

FEATURES = [
    "psychologic",
    "dysarthria",
    "dysphagia",
    "spasticity",
    "hyperreflexia",
    "vision",
    "weakness",
    "sensory",
    "ataxia",
    "tremor",
    "bladder",
    "bowel",
    "cognitive",
    "falls",
    "fatigue",
    "gait",
    "pain",
]

MODULE_NAMES = [
    "sensory-visual-pain",
    "ataxic-spastic-falls",
    "cognitive-psychologic-fatigue",
    "autonomic-bladder-bowel",
]

MODULE_LABELS = [
    "Sensory-visual-pain",
    "Ataxic-spastic-falls",
    "Cognitive-psychologic-fatigue",
    "Autonomic-bladder-bowel",
]

MODULE_COLORS = {
    "sensory-visual-pain": "tab:purple",
    "ataxic-spastic-falls": "tab:red",
    "cognitive-psychologic-fatigue": "tab:blue",
    "autonomic-bladder-bowel": "tab:green",
}

N_COMPONENTS = 4
NMF_RANDOM_STATE = 0
MAX_ITER = 5000

# Default assumes raw NMF order already matches canonical order.
DEFAULT_COMPONENT_ORDER = [0, 1, 2, 3]


# ==========================================================
# Command-line arguments
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create tetrahedron and simplex plots for four-module NMF phenotype space."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to patient-level CSV file.",
    )

    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Directory where figures and CSV outputs will be saved.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Dominance threshold for coloring patient markers.",
    )

    parser.add_argument(
        "--component-order",
        nargs=4,
        type=int,
        default=DEFAULT_COMPONENT_ORDER,
        help=(
            "Order for mapping raw NMF components to canonical module order. "
            "Example: --component-order 2 0 3 1"
        ),
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=NMF_RANDOM_STATE,
        help="Random state for NMF.",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=MAX_ITER,
        help="Maximum number of NMF iterations.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively.",
    )

    return parser.parse_args()


# ==========================================================
# Data and NMF
# ==========================================================

def load_data(path):
    df = pd.read_csv(path)
    print(f"Patients: {len(df)}")
    return df


def get_feature_matrix(df, features):
    missing = [col for col in features if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = df[features].fillna(0).to_numpy(dtype=float)
    print(f"Feature matrix shape: {X.shape}")

    return X


def fit_nmf(X, n_components=4, random_state=0, max_iter=5000):
    model = NMF(
        n_components=n_components,
        init="nndsvd",
        random_state=random_state,
        max_iter=max_iter,
    )

    W_raw = model.fit_transform(X)
    H_raw = model.components_

    print(f"W shape: {W_raw.shape}")
    print(f"H shape: {H_raw.shape}")

    return W_raw, H_raw


def print_top_features(H, features, top_n=8):
    print("\nTop features per raw NMF component")
    print("-----------------------------------")

    for i in range(H.shape[0]):
        idx = np.argsort(H[i])[::-1][:top_n]

        print(f"\nRaw component {i}")

        for j in idx:
            print(f"  {features[j]:<15} {H[i, j]:.3f}")


def reorder_components(W_raw, H_raw, component_order):
    W = W_raw[:, component_order]
    H = H_raw[component_order, :]

    return W, H


def normalize_rows(W):
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0

    return W / row_sums


def build_patient_module_df(W_norm, threshold):
    module_df = pd.DataFrame(W_norm, columns=MODULE_NAMES)

    module_df["dominant_module"] = module_df[MODULE_NAMES].idxmax(axis=1)
    module_df["dominant_module_prop"] = module_df[MODULE_NAMES].max(axis=1)

    eps = 1e-12
    module_df["module_entropy"] = -np.sum(W_norm * np.log(W_norm + eps), axis=1)
    module_df["effective_modules"] = np.exp(module_df["module_entropy"])

    module_df["dominance_class"] = np.where(
        module_df["dominant_module_prop"] >= threshold,
        "dominant",
        "mixed",
    )

    return module_df


def print_model_summary(X, W, H, module_df, threshold):
    reconstruction = W @ H
    relative_error = np.linalg.norm(X - reconstruction) / np.linalg.norm(X)

    print("\nModel summary")
    print("-------------")
    print(f"Relative reconstruction error: {relative_error:.3f}")
    print(f"Dominance threshold: {threshold:.2f}")

    print("\nDominant module counts")
    print(module_df["dominant_module"].value_counts())

    print("\nDominance class counts")
    print(module_df["dominance_class"].value_counts())

    print("\nMean effective number of modules")
    print(round(module_df["effective_modules"].mean(), 3))

    print("\nTop features per canonical module")
    print("---------------------------------")

    for i, module in enumerate(MODULE_NAMES):
        idx = np.argsort(H[i])[::-1][:8]

        print(f"\n{module}")

        for j in idx:
            print(f"  {FEATURES[j]:<15} {H[i, j]:.3f}")


# ==========================================================
# Tetrahedron geometry
# ==========================================================

def tetrahedron_vertices():
    """
    Return coordinates for a tetrahedron embedded in 3D.

    The fourth module is mapped to the origin.
    """
    return np.array([
        [1.0, 0.0, 0.0],  # sensory-visual-pain
        [0.0, 1.0, 0.0],  # ataxic-spastic-falls
        [0.0, 0.0, 1.0],  # cognitive-psychologic-fatigue
        [0.0, 0.0, 0.0],  # autonomic-bladder-bowel
    ])


def module_weights_to_tetrahedron(W_norm):
    vertices = tetrahedron_vertices()
    return W_norm @ vertices


# ==========================================================
# Tetrahedron plot
# ==========================================================

def plot_tetrahedron(W_norm, module_df, threshold, output_path, show=False):
    coords = module_weights_to_tetrahedron(W_norm)
    vertices = tetrahedron_vertices()

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor("white")
        axis.pane.set_alpha(1.0)

    mixed = module_df["dominant_module_prop"] < threshold

    ax.scatter(
        coords[mixed, 0],
        coords[mixed, 1],
        coords[mixed, 2],
        color="lightgray",
        alpha=0.35,
        s=10,
        edgecolor="gray",
        label=f"Admixed (< {threshold:.2f})",
    )

    for module in MODULE_NAMES:
        idx = (
            (module_df["dominant_module"] == module)
            & (module_df["dominant_module_prop"] >= threshold)
        )

        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            coords[idx, 2],
            color=MODULE_COLORS[module],
            alpha=0.85,
            s=60,
            edgecolor="gray",
            linewidth=0.25,
            label=module,
        )

    for i, j in combinations(range(4), 2):
        ax.plot(
            [vertices[i, 0], vertices[j, 0]],
            [vertices[i, 1], vertices[j, 1]],
            [vertices[i, 2], vertices[j, 2]],
            color="black",
            linewidth=2,
            alpha=0.75,
        )

    label_offsets = np.array([
        [0.08, 0.00, 0.00],
        [0.08, 0.00, 0.00],
        [0.00, 0.00, 0.08],
        [-0.08, -0.08, -0.03],
    ])

    for i, label in enumerate(MODULE_LABELS):
        module = MODULE_NAMES[i]

        ax.scatter(
            vertices[i, 0],
            vertices[i, 1],
            vertices[i, 2],
            color=MODULE_COLORS[module],
            marker="^",
            s=30,
            edgecolor=MODULE_COLORS[module],
            linewidth=0.3,
        )

        x, y, z = vertices[i] + label_offsets[i]

        ax.text(
            x,
            y,
            z,
            label,
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel(MODULE_LABELS[0])
    ax.set_ylabel(MODULE_LABELS[1])
    ax.set_zlabel(MODULE_LABELS[2])

    ax.view_init(elev=22, azim=35)

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=12,
    )

    plt.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved tetrahedron plot: {output_path}")

    if show:
        plt.show()

    plt.close(fig)


# ==========================================================
# Ternary simplex plots
# ==========================================================

def require_ternary():
    try:
        import ternary
    except ImportError as exc:
        raise ImportError(
            "The package 'python-ternary' is required for simplex plots. "
            "Install it with: pip install python-ternary"
        ) from exc

    return ternary


def plot_simplex_face(W_norm, module_df, face, threshold, output_path, show=False):
    ternary = require_ternary()

    a, b, c = face
    points = np.column_stack([W_norm[:, a], W_norm[:, b], W_norm[:, c]])

    fig, tax = ternary.figure(scale=1.0)
    fig.set_size_inches(7.5, 6.5)

    mixed = module_df["dominant_module_prop"] < threshold

    tax.scatter(
        points[mixed],
        color="gray",
        alpha=0.40,
        s=10,
        edgecolor="gray",
        label=f"Admixed (< {threshold:.2f})",
    )

    for module_index in face:
        module = MODULE_NAMES[module_index]

        idx = (
            (module_df["dominant_module"] == module)
            & (module_df["dominant_module_prop"] >= threshold)
        )

        tax.scatter(
            points[idx],
            color=MODULE_COLORS[module],
            alpha=0.85,
            s=60,
            edgecolor="gray",
            label=MODULE_LABELS[module_index],
        )

    tax.left_axis_label(MODULE_LABELS[b], fontsize=14)
    tax.right_axis_label(MODULE_LABELS[a], fontsize=14)
    tax.bottom_axis_label(MODULE_LABELS[c], fontsize=14, offset=-0.12)

    tax.boundary(linewidth=2)
    tax.legend(loc="upper right", fontsize=12)
    tax.clear_matplotlib_ticks()

    plt.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved simplex plot: {output_path}")

    if show:
        plt.show()

    plt.close(fig)


def plot_selected_simplex_faces(W_norm, module_df, threshold, output_dir, show=False):
    plot_simplex_face(
        W_norm=W_norm,
        module_df=module_df,
        face=(0, 1, 3),
        threshold=threshold,
        output_path=output_dir / f"simplex_sensory_ataxic_autonomic_{threshold:.2f}.pdf",
        show=show,
    )

    plot_simplex_face(
        W_norm=W_norm,
        module_df=module_df,
        face=(0, 1, 2),
        threshold=threshold,
        output_path=output_dir / f"simplex_sensory_ataxic_cognitive_{threshold:.2f}.pdf",
        show=show,
    )


# ==========================================================
# Main
# ==========================================================

def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)
    X = get_feature_matrix(df, FEATURES)

    W_raw, H_raw = fit_nmf(
        X,
        n_components=N_COMPONENTS,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )

    print_top_features(H_raw, FEATURES, top_n=8)

    W, H = reorder_components(
        W_raw=W_raw,
        H_raw=H_raw,
        component_order=args.component_order,
    )

    W_norm = normalize_rows(W)

    module_df = build_patient_module_df(
        W_norm=W_norm,
        threshold=args.threshold,
    )

    print_model_summary(
        X=X,
        W=W,
        H=H,
        module_df=module_df,
        threshold=args.threshold,
    )

    # Save patient-level module summary.
    out_df = pd.concat(
        [df.reset_index(drop=True), module_df],
        axis=1,
    )

    module_summary_path = output_dir / "ms_patient_module_simplex_summary.csv"
    out_df.to_csv(module_summary_path, index=False)
    print(f"Saved patient module summary: {module_summary_path}")

    # Save canonical loading matrix.
    H_df = pd.DataFrame(H, index=MODULE_NAMES, columns=FEATURES)
    loading_path = output_dir / "nmf_module_loading_matrix_canonical.csv"
    H_df.to_csv(loading_path)
    print(f"Saved canonical NMF loading matrix: {loading_path}")

    # Plot tetrahedron.
    plot_tetrahedron(
        W_norm=W_norm,
        module_df=module_df,
        threshold=args.threshold,
        output_path=output_dir / f"ms_phenotype_tetrahedron_{args.threshold:.2f}.pdf",
        show=args.show,
    )

    # Plot selected simplex faces.
    plot_selected_simplex_faces(
        W_norm=W_norm,
        module_df=module_df,
        threshold=args.threshold,
        output_dir=output_dir,
        show=args.show,
    )


if __name__ == "__main__":
    main()