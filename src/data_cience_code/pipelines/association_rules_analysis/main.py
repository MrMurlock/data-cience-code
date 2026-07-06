"""Association Rules Analysis Pipeline.

Compares Apriori vs FP-Growth on the Sleep Disorder dataset.
Usage:
    PYTHONPATH=src poetry run python -m \\
        data_cience_code.pipelines.association_rules_analysis.main
"""

from data_cience_code.pipelines.association_rules_analysis import (
    association_rules_pipeline,
    interpretation_report,
    preparation_pipeline,
    visualization,
)


def run():
    """Run complete association rules pipeline."""
    print("=" * 60)
    print("  ASSOCIATION RULES ANALYSIS PIPELINE")
    print("  Apriori vs FP-Growth on Sleep Disorder Dataset")
    print("=" * 60)
    print()

    # Phase 1: Data preparation
    encoded_df = preparation_pipeline.run()

    # Phase 2: Association rules mining
    results = association_rules_pipeline.run(encoded_df)

    # Phase 3a: General visualizations
    visualization.run(results)

    # Phase 3b: Filtered visualizations (by disorder, occupation)
    filtered_viz_paths = visualization.run_filtered(results)

    # Phase 4: Interpretation report
    interpretation_report.run(results, filtered_viz_paths)

    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run()
