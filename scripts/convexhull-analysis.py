#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:49:56 2026

@author: danielhier
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Identify convex-hull boundary cases in four-module NMF phenotype space.

This script:
1. Loads a patient-level phenotype feature matrix.
2. Fits a four-module non-negative matrix factorization model.
3. Normalizes patient-level module weights so each row sums to 1.
4. Computes the convex hull of the normalized module weights.
5. Saves the patient cases that lie on the empirical boundary
   of the normalized four-module phenotype space.

Because four normalized module weights sum to 1, patients lie in a
3-dimensional simplex embedded in 4-dimensional space. ConvexHull is
therefore applied to the first three module coordinates; the fourth is
determined by the simplex constraint.

Example:
    python scripts/identify_convex_hull_boundary_cases.py \
        --input data/ms_patient_level_feature_matrix.csv \
        --output results/convex_hull_boundary_cases.csv
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.decomposition import NMF


# ==========================================================
# Defaults
# ==========================================================

DEFAULT_FEATURES = [
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
    "Sensory_Visual_Pain",
    "Ataxic_Spastic_Falls",
    "Cognitive_Psychologic_Fatigue",
    "Autonomic_Bladder_Bowel",
]

N_MODULES = 4
RANDOM_STATE = 0
MAX_ITER = 5000


# ==========================================================
# Command-line arguments
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Identify convex-hull boundary cases from normalized "
            "four-module NMF phenotype weights."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to patient-level CSV file.",
    )

    parser.add_argument(
        "--output",
        default="results/convex_hull_boundary_cases.csv",
        help="Path for output CSV file.",
    )

    parser.add_argument(
        "--id-col",
        default=None,
        help=(
            "Optional patient identifier column to include in output. "
            "If omitted, a study_case_index will be used."
        ),
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for NMF.",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=MAX_ITER,
        help="Maximum NMF iterations.",
    )

    return parser.parse_args()


# ==========================================================
# Data loading
# ==========================================================

def load_feature_matrix(input_path, features):
    """
    Load patient-level phenotype feature matrix.

    Parameters
    ----------
    input_path : str or pathlib.Path
        Path to CSV file.

    features : list of str
        Phenotype feature columns used for NMF.

    Returns
    -------
    df : pandas.DataFrame
        Original dataframe.

    X : numpy.ndarray
        Patient-by-feature matrix.
    """

    df = pd.read_csv(input_path)

    missing = [col for col in features if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = df[features].to_numpy(dtype=float)

    print("Loaded data")
    print("-----------")
    print(f"Rows/patients: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")

    return df, X


# ==========================================================
# NMF and normalization
# ==========================================================

def run_nmf(X, n_modules=4, random_state=0, max_iter=5000):
    """
    Fit NMF model and return patient weights and module loadings.
    """

    model = NMF(
        n_components=n_modules,
        init="nndsvd",
        random_state=random_state,
        max_iter=max_iter,
    )

    W = model.fit_transform(X)
    H = model.components_

    return W, H


def normalize_rows(W):
    """
    Normalize each patient's module weights to sum to 1.
    """

    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1.0

    W_norm = W / row_sums[:, None]

    return W_norm


# ==========================================================
# Convex hull
# ==========================================================

def simplex_coordinates(W_norm):
    """
    Convert normalized four-module weights to independent coordinates.

    Since rows of W_norm sum to 1, the four-dimensional points lie in
    a three-dimensional simplex. Dropping the final coordinate preserves
    the information needed for convex-hull geometry because:

        w4 = 1 - w1 - w2 - w3
    """

    if W_norm.ndim != 2:
        raise ValueError("W_norm must be a 2D array.")

    if W_norm.shape[1] < 2:
        raise ValueError("W_norm must contain at least two modules.")

    return W_norm[:, :-1]


def identify_convex_hull_cases(df, W_norm, module_names, id_col=None):
    """
    Identify patient cases lying on the convex hull.

    Parameters
    ----------
    df : pandas.DataFrame
        Original patient-level dataframe.

    W_norm : numpy.ndarray
        Row-normalized patient-by-module weight matrix.

    module_names : list of str
        Names of the NMF modules.

    id_col : str or None
        Optional ID column to retain in output.

    Returns
    -------
    hull_df : pandas.DataFrame
        Convex-hull boundary cases with normalized module weights.

    hull : scipy.spatial.ConvexHull
        Convex hull object.
    """

    W_hull = simplex_coordinates(W_norm)

    hull = ConvexHull(W_hull)

    vertex_indices = hull.vertices

    rows = []

    for idx in vertex_indices:
        weights = W_norm[idx, :]

        dominant_index = int(np.argmax(weights))
        sorted_indices = np.argsort(weights)[::-1]

        primary_index = int(sorted_indices[0])
        secondary_index = int(sorted_indices[1])

        row = {
            "study_case_index": int(idx),
            "dominant_module": module_names[dominant_index],
            "max_weight": float(weights[primary_index]),
            "secondary_module": module_names[secondary_index],
            "secondary_weight": float(weights[secondary_index]),
            "margin": float(weights[primary_index] - weights[secondary_index]),
            "pure_vertex": bool(np.isclose(weights[primary_index], 1.0)),
        }

        if id_col is not None and id_col in df.columns:
            row[id_col] = df.loc[idx, id_col]

        for module_index, module_name in enumerate(module_names):
            row[f"{module_name}_weight"] = float(weights[module_index])

        rows.append(row)

    hull_df = pd.DataFrame(rows)

    hull_df = hull_df.sort_values(
        by=["max_weight", "margin"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return hull_df, hull


def print_hull_summary(W_norm, hull, hull_df):
    """
    Print convex-hull summary.
    """

    n_patients = W_norm.shape[0]
    n_hull = len(hull.vertices)
    n_pure = int(hull_df["pure_vertex"].sum())

    print("\nConvex-hull summary")
    print("-------------------")
    print(f"Patients: {n_patients}")
    print(f"Original module dimensions: {W_norm.shape[1]}")
    print(f"Hull calculation dimensions: {W_norm.shape[1] - 1}")
    print(f"Convex-hull boundary cases: {n_hull}")
    print(f"Fraction on hull: {n_hull / n_patients:.3f}")
    print(f"Pure occupied vertices: {n_pure}")


# ==========================================================
# Main
# ==========================================================

def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df, X = load_feature_matrix(
        input_path=input_path,
        features=DEFAULT_FEATURES,
    )

    W, H = run_nmf(
        X=X,
        n_modules=N_MODULES,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )

    W_norm = normalize_rows(W)

    hull_df, hull = identify_convex_hull_cases(
        df=df,
        W_norm=W_norm,
        module_names=MODULE_NAMES,
        id_col=args.id_col,
    )

    print_hull_summary(
        W_norm=W_norm,
        hull=hull,
        hull_df=hull_df,
    )

    hull_df.to_csv(output_path, index=False)

    print("\nSaved convex-hull boundary cases:")
    print(output_path)


if __name__ == "__main__":
    main()