#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 14:35:26 2026

@author: danielhier
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core NMF analysis of MS phenotype modules.

Purpose:
    1. Run NMF for k = 2, 3, 4, 5.
    2. Name modules M1, M2, etc.
    3. Plot sorted module-feature heatmaps.
    4. Save loading matrices and goodness-of-fit metrics.
    5. No clustering.

Example:
    python core_nmf_analysis.py \
        --input data/ms_patient_summary.csv \
        --output-dir results/nmf_k2_to_k5
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF


# ==========================================================
# Configuration
# ==========================================================

FEATURES = [
    "psychologic", "dysarthria", "dysphagia", "spasticity", "hyperreflexia",
    "vision", "weakness", "sensory", "ataxia", "tremor", "bladder", "bowel",
    "cognitive", "falls", "fatigue", "gait", "pain"
]

K_VALUES = [2, 3, 4, 5]
RANDOM_STATE = 0
MAX_ITER = 5000


# ==========================================================
# Command-line arguments
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run core NMF analysis for MS phenotype modules."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input patient-level CSV file."
    )

    parser.add_argument(
        "--output-dir",
        default="nmf_k2_to_k5",
        help="Directory where output files will be saved."
    )

    return parser.parse_args()


# ==========================================================
# Helper functions
# ==========================================================

def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} patients")
    return df


def check_features(df: pd.DataFrame, features: list[str]) -> None:
    missing = [feature for feature in features if feature not in df.columns]

    if missing:
        raise ValueError(f"Missing feature columns: {missing}")


def fit_nmf_model(X: np.ndarray, n_modules: int) -> tuple[NMF, np.ndarray, np.ndarray]:
    nmf = NMF(
        n_components=n_modules,
        init="nndsvd",
        random_state=RANDOM_STATE,
        max_iter=MAX_ITER
    )

    W = nmf.fit_transform(X)
    H = nmf.components_

    return nmf, W, H


def build_loading_table(
    H: np.ndarray,
    features: list[str],
    module_names: list[str]
) -> pd.DataFrame:
    return pd.DataFrame(H, index=module_names, columns=features)


def sort_features_by_module_loading(H_df: pd.DataFrame) -> list[str]:
    H = H_df.to_numpy()
    features = list(H_df.columns)

    dominant_module_index = np.argmax(H, axis=0)
    dominant_loading = np.max(H, axis=0)

    feature_info = pd.DataFrame({
        "feature": features,
        "dominant_module_index": dominant_module_index,
        "dominant_loading": dominant_loading
    })

    feature_info = feature_info.sort_values(
        by=["dominant_module_index", "dominant_loading"],
        ascending=[True, False]
    )

    return feature_info["feature"].tolist()


def compute_reconstruction_metrics(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    module_names: list[str]
) -> dict:
    reconstruction = W @ H

    residual_norm = np.linalg.norm(X - reconstruction, ord="fro")
    x_norm = np.linalg.norm(X, ord="fro")
    relative_error = residual_norm / x_norm

    approximate_variance_captured = 1 - relative_error ** 2
    total_sum_squares = np.sum(X ** 2)

    metrics = {
        "k": len(module_names),
        "reconstruction_error": residual_norm,
        "relative_reconstruction_error": relative_error,
        "approximate_variance_captured": approximate_variance_captured
    }

    for i, module_name in enumerate(module_names):
        W_i = W[:, i:i + 1]
        H_i = H[i:i + 1, :]
        X_i = W_i @ H_i

        metrics[f"{module_name}_approx_variance"] = (
            np.sum(X_i ** 2) / total_sum_squares
        )

    return metrics


def plot_loading_heatmap(
    H_df: pd.DataFrame,
    sorted_features: list[str],
    output_path: Path,
    title: str
) -> None:
    H_sorted = H_df[sorted_features]
    values = H_sorted.to_numpy()

    plt.figure(figsize=(12, 5))
    ax = plt.gca()

    image = ax.imshow(
        values,
        cmap="viridis",
        aspect="auto"
    )

    plt.colorbar(image, label="Loading")

    ax.set_yticks(np.arange(H_sorted.shape[0]))
    ax.set_yticklabels(H_sorted.index)

    ax.set_xticks(np.arange(H_sorted.shape[1]))
    ax.set_xticklabels(H_sorted.columns, rotation=90)

    max_value = np.nanmax(values)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            text_color = "white" if value < 0.50 * max_value else "black"

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight="bold"
            )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Phenotype feature")
    ax.set_ylabel("NMF module")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved heatmap: {output_path}")


def print_top_loadings(H_df: pd.DataFrame, top_n: int = 5) -> None:
    print(f"\nTop {top_n} loadings for each module")

    for module_name in H_df.index:
        top = H_df.loc[module_name].sort_values(ascending=False).head(top_n)
        print(f"\n{module_name}")
        print(top.round(3))


# ==========================================================
# Main
# ==========================================================

def main() -> None:
    args = parse_args()

    data_path = Path(args.input)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    check_features(df, FEATURES)

    X = df[FEATURES].to_numpy(dtype=float)

    all_metrics = []

    for k in K_VALUES:
        print("\n" + "=" * 60)
        print(f"Running NMF with k = {k}")
        print("=" * 60)

        module_names = [f"M{i + 1}" for i in range(k)]

        _, W, H = fit_nmf_model(X, k)

        H_df = build_loading_table(H, FEATURES, module_names)
        sorted_features = sort_features_by_module_loading(H_df)

        loading_file = output_dir / f"nmf_k{k}_loading_matrix.csv"
        H_df.to_csv(loading_file)
        print(f"Saved loading matrix: {loading_file}")

        sorted_loading_file = output_dir / f"nmf_k{k}_loading_matrix_sorted.csv"
        H_df[sorted_features].to_csv(sorted_loading_file)
        print(f"Saved sorted loading matrix: {sorted_loading_file}")

        W_df = pd.DataFrame(W, columns=module_names)
        W_file = output_dir / f"nmf_k{k}_patient_module_weights.csv"
        W_df.to_csv(W_file, index=False)
        print(f"Saved patient module weights: {W_file}")

        print_top_loadings(H_df, top_n=5)

        metrics = compute_reconstruction_metrics(X, W, H, module_names)
        all_metrics.append(metrics)

        print("\nGoodness of fit")
        print(
            f"Relative reconstruction error: "
            f"{metrics['relative_reconstruction_error']:.3f}"
        )
        print(
            f"Approximate variance captured: "
            f"{metrics['approximate_variance_captured']:.3f}"
        )

        heatmap_file = output_dir / f"nmf_k{k}_heatmap.png"

        plot_loading_heatmap(
            H_df=H_df,
            sorted_features=sorted_features,
            output_path=heatmap_file,
            title=f"NMF phenotype modules, k = {k}"
        )

    metrics_df = pd.DataFrame(all_metrics)

    metrics_file = output_dir / "nmf_k2_to_k5_goodness_of_fit.csv"
    metrics_df.to_csv(metrics_file, index=False)

    print("\n" + "=" * 60)
    print("Summary goodness-of-fit table")
    print("=" * 60)
    print(metrics_df.round(3))

    print(f"\nSaved goodness-of-fit table: {metrics_file}")
    print("\nDone.")


if __name__ == "__main__":
    main()