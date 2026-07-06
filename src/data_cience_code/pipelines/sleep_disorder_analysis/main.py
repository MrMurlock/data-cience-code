"""
Main pipeline for Sleep Disorder dataset analysis.

This module orchestrates the complete analysis pipeline for the Sleep Disorder dataset,
including EDA, clustering analysis, and all visualizations.
"""

from .eda_pipeline import run_eda
from .clustering_pipeline import run_full_clustering_analysis


def run(k_values: list[int] = [4, 6, 10, 14]) -> dict:
    """
    Run the complete analysis pipeline for Sleep Disorder dataset.

    Args:
        k_values: List of K values for clustering analysis

    Returns:
        Dictionary with all results from EDA and clustering
    """
    print("=" * 70)
    print("=== SLEEP DISORDER DATASET - COMPLETE ANALYSIS PIPELINE ===")
    print("=" * 70)

    # Phase 1: EDA
    print("\n" + "=" * 70)
    print("PHASE 1: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    eda_results = run_eda()

    # Phase 2: Clustering Analysis
    print("\n" + "=" * 70)
    print("PHASE 2: CLUSTERING ANALYSIS")
    print("=" * 70)
    clustering_results = run_full_clustering_analysis(k_values)

    # Combine results
    all_results = {
        "eda": eda_results,
        "clustering": clustering_results,
    }

    print("\n" + "=" * 70)
    print("=== COMPLETE ANALYSIS PIPELINE FINISHED ===")
    print("=" * 70)
    print("\nAll results have been saved to:")
    print("- src/dataset/processed/ (processed data and models)")
    print("- src/output/figures/ (visualizations)")
    print("- src/output/reports/ (metrics and analysis)")

    return all_results


if __name__ == "__main__":
    # Run complete pipeline with default K values
    results = run([4, 6, 10, 14])
