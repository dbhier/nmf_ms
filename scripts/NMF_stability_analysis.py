#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NMF Stability Analysis for MS Phenotype Modules
-----------------------------------------------

Purpose:
    Assess stability of the selected four-module NMF solution using
    repeated 90% patient subsampling.

Pipeline:
    1. Load patient-level phenotype matrix.
    2. Fit reference four-module NMF model on the full cohort.
    3. Repeatedly sample 90% of patients without replacement.
    4. Refit four-module NMF to each subsample.
    5. Match subsample modules to full-cohort modules using maximum
       Pearson correlation of feature-loading vectors.
    6. Save long-form and summary stability results.

Created for:
    Phenotypic Diversity in Multiple Sclerosis can be Represented
    by Four Additive Symptom Modules

Author:
    Daniel B. Hier
    
Libraries:
    pip install pandas numpy scipy scikit-learn    
"""

# ==========================================================
# Imports
# ==========================================================

from pathlib import Path

import numpy as np
import pandas as pd
import argparse
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF


# ==========================================================
# Configuration
# ==========================================================

INPUT_PATH = Path("/Users/danielhier/Desktop/MS_text")
OUTPUT_DIR = Path("/Users/danielhier/Desktop/MS_text")

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
    "Sensory_Visual_Pain",
    "Ataxic_Spastic_Falls",
    "Cognitive_Psychologic_Fatigue",
    "Autonomic_Bladder_Bowel",
]

N_MODULES = 4
REFERENCE_RANDOM_STATE = 0

N_TRIALS = 100
SAMPLE_FRACTION = 0.90
BASE_RANDOM_STATE = 1000

LONG_OUTPUT = OUTPUT_DIR / "nmf_90_percent_subsampling_stability_long.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "nmf_90_percent_subsampling_stability_summary.csv"
REFERENCE_LOADINGS_OUTPUT = OUTPUT_DIR / "nmf_reference_module_loadings.csv"


# ==========================================================
# Data Loading
# ==========================================================

def load_feature_matrix(path, features):
    """
    Load the patient-level phenotype feature matrix.

    Parameters
    ----------
    path : pathlib.Path
        Path to CSV file.

    features : list of str
        Names of phenotype feature columns.

    Returns
    -------
    df : pandas.DataFrame
        Full input dataframe.

    X : numpy.ndarray
        Patient-by-feature matrix.
    """

    df = pd.read_csv(path)

    missing = [feature for feature in features if feature not in df.columns]

    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[features].to_numpy(dtype=float)

    print("Loaded dataset")
    print("--------------")
    print(f"Input file: {path}")
    print(f"Patients: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")

    return df, X


# ==========================================================
# NMF
# ==========================================================

def run_nmf(X, n_modules=4, random_state=0):
    """
    Fit a non-negative matrix factorization model.

    Parameters
    ----------
    X : numpy.ndarray
        Patient-by-feature matrix.

    n_modules : int
        Number of NMF components.

    random_state : int
        Random seed.

    Returns
    -------
    W : numpy.ndarray
        Patient-by-module weight matrix.

    H : numpy.ndarray
        Module-by-feature loading matrix.
    """

    model = NMF(
        n_components=n_modules,
        init="nndsvd",
        random_state=random_state,
        max_iter=5000,
    )

    W = model.fit_transform(X)
    H = model.components_

    return W, H


def make_loading_table(H, features, module_names):
    """
    Convert NMF loading matrix into a readable dataframe.

    Parameters
    ----------
    H : numpy.ndarray
        Module-by-feature loading matrix.

    features : list of str
        Feature names.

    module_names : list of str
        Module names.

    Returns
    -------
    loading_df : pandas.DataFrame
        Long-form module loading table.
    """

    rows = []

    for module_index, module_name in enumerate(module_names):
        for feature_index, feature_name in enumerate(features):
            rows.append(
                {
                    "module_index": module_index,
                    "module": module_name,
                    "feature_index": feature_index,
                    "feature": feature_name,
                    "loading": H[module_index, feature_index],
                }
            )

    loading_df = pd.DataFrame(rows)

    return loading_df


# ==========================================================
# Module Matching
# ==========================================================

def module_correlation_matrix(H_ref, H_test):
    """
    Compute Pearson correlations between reference and test modules.

    Parameters
    ----------
    H_ref : numpy.ndarray
        Reference module-by-feature loading matrix.

    H_test : numpy.ndarray
        Test module-by-feature loading matrix.

    Returns
    -------
    corr : numpy.ndarray
        Matrix where corr[i, j] is the Pearson correlation between
        reference module i and test module j.
    """

    n_ref = H_ref.shape[0]
    n_test = H_test.shape[0]

    corr = np.zeros((n_ref, n_test))

    for i in range(n_ref):
        for j in range(n_test):
            corr[i, j] = np.corrcoef(H_ref[i, :], H_test[j, :])[0, 1]

    return corr


def match_modules_by_correlation(H_ref, H_test):
    """
    Match unordered NMF modules using one-to-one maximum correlation.

    NMF components can appear in arbitrary order. This function uses
    the Hungarian algorithm to find the one-to-one assignment that
    maximizes total Pearson correlation between module-loading vectors.

    Parameters
    ----------
    H_ref : numpy.ndarray
        Reference module-by-feature loading matrix.

    H_test : numpy.ndarray
        Test module-by-feature loading matrix.

    Returns
    -------
    matched : pandas.DataFrame
        One row per reference module.
    """

    corr = module_correlation_matrix(H_ref, H_test)

    # linear_sum_assignment minimizes cost, so use negative correlation.
    row_indices, col_indices = linear_sum_assignment(-corr)

    rows = []

    for ref_idx, test_idx in zip(row_indices, col_indices):
        rows.append(
            {
                "reference_module_index": int(ref_idx),
                "test_module_index": int(test_idx),
                "matched_correlation": float(corr[ref_idx, test_idx]),
            }
        )

    matched = (
        pd.DataFrame(rows)
        .sort_values("reference_module_index")
        .reset_index(drop=True)
    )

    return matched


# ==========================================================
# Stability Analysis
# ==========================================================

def nmf_subsampling_stability(
    X,
    H_ref,
    module_names,
    n_trials=100,
    sample_fraction=0.90,
    n_modules=4,
    base_random_state=1000,
):
    """
    Assess NMF module stability using repeated patient subsampling.

    In each trial, a random fraction of patients is sampled without
    replacement. NMF is refit to the subsample, and the resulting
    module-loading vectors are matched to the reference full-cohort
    modules.

    Stability is measured as the Pearson correlation between each
    full-cohort module's 17-dimensional feature-loading vector and
    the matched subsample module's feature-loading vector.

    Parameters
    ----------
    X : numpy.ndarray
        Full patient-by-feature matrix.

    H_ref : numpy.ndarray
        Reference full-cohort module-by-feature loading matrix.

    module_names : list of str
        Names of the reference modules.

    n_trials : int
        Number of subsampling trials.

    sample_fraction : float
        Fraction of patients retained in each trial.

    n_modules : int
        Number of NMF modules.

    base_random_state : int
        Starting random seed for reproducibility.

    Returns
    -------
    stability_long : pandas.DataFrame
        One row per module per trial.

    stability_summary : pandas.DataFrame
        Summary statistics by reference module.
    """

    n_patients = X.shape[0]
    n_sample = int(round(sample_fraction * n_patients))

    if n_sample <= n_modules:
        raise ValueError("Subsample size must be larger than number of modules.")

    rng = np.random.default_rng(base_random_state)

    all_rows = []

    print("\nNMF subsampling stability analysis")
    print("----------------------------------")
    print(f"Trials: {n_trials}")
    print(f"Patients per trial: {n_sample} of {n_patients}")
    print(f"Sample fraction: {sample_fraction}")

    for trial in range(n_trials):

        sample_idx = rng.choice(
            n_patients,
            size=n_sample,
            replace=False,
        )

        X_sub = X[sample_idx, :]

        _, H_sub = run_nmf(
            X_sub,
            n_modules=n_modules,
            random_state=base_random_state + trial,
        )

        matched = match_modules_by_correlation(H_ref, H_sub)

        for _, row in matched.iterrows():

            ref_idx = int(row["reference_module_index"])

            all_rows.append(
                {
                    "trial": trial + 1,
                    "sample_fraction": sample_fraction,
                    "n_patients_full": n_patients,
                    "n_patients_subsample": n_sample,
                    "reference_module_index": ref_idx,
                    "reference_module": module_names[ref_idx],
                    "test_module_index": int(row["test_module_index"]),
                    "matched_correlation": float(row["matched_correlation"]),
                }
            )

    stability_long = pd.DataFrame(all_rows)

    stability_summary = (
        stability_long
        .groupby("reference_module", as_index=False)
        .agg(
            median_r=("matched_correlation", "median"),
            q1=("matched_correlation", lambda x: x.quantile(0.25)),
            q3=("matched_correlation", lambda x: x.quantile(0.75)),
            min_r=("matched_correlation", "min"),
            max_r=("matched_correlation", "max"),
            mean_r=("matched_correlation", "mean"),
            sd_r=("matched_correlation", "std"),
        )
    )

    return stability_long, stability_summary


def main():

    parser = argparse.ArgumentParser(
        description="Run NMF stability analysis for MS phenotype modules."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file"
    )

    parser.add_argument(
        "--output_dir",
        default="/Users/danielhier/Desktop/MS_text",
        help="Directory for output CSV files"
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of subsampling trials"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    long_output = output_dir / "nmf_90_percent_subsampling_stability_long.csv"
    summary_output = output_dir / "nmf_90_percent_subsampling_stability_summary.csv"
    reference_loadings_output = output_dir / "nmf_reference_module_loadings.csv"

    _, X = load_feature_matrix(input_path, FEATURES)

    print("\nFitting reference NMF model")
    print("---------------------------")

    _, H_ref = run_nmf(
        X,
        n_modules=N_MODULES,
        random_state=REFERENCE_RANDOM_STATE,
    )

    reference_loadings = make_loading_table(
        H=H_ref,
        features=FEATURES,
        module_names=MODULE_NAMES,
    )

    reference_loadings.to_csv(reference_loadings_output, index=False)

    stability_long, stability_summary = nmf_subsampling_stability(
        X=X,
        H_ref=H_ref,
        module_names=MODULE_NAMES,
        n_trials=args.trials,
        sample_fraction=SAMPLE_FRACTION,
        n_modules=N_MODULES,
        base_random_state=BASE_RANDOM_STATE,
    )

    stability_long.to_csv(long_output, index=False)
    stability_summary.to_csv(summary_output, index=False)

    print("\nNMF stability summary")
    print("---------------------")
    print(stability_summary)

    print("\nSaved files")
    print("-----------")
    print(long_output)
    print(summary_output)
    print(reference_loadings_output)


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":
    main()