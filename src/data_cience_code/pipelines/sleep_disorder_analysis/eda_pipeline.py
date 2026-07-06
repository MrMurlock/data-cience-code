"""
EDA pipeline for Sleep Disorder dataset.

This module executes exploratory data analysis specific to the Sleep Disorder dataset,
using generic functions from data/ and eda/ modules.
"""

from data_cience_code.data.load import load_dataset
from data_cience_code.data.validate import validate
from data_cience_code.data.preprocess import clean_pipeline
from data_cience_code.eda.exploratory import describe_numeric, describe_categorical
from data_cience_code.eda import viz_bivariate
from data_cience_code.eda import viz_cleaned
from data_cience_code.eda import viz_raw
import data_cience_code.config.config as config


def run_eda() -> dict:
    """Run the EDA pipeline for Sleep Disorder dataset."""
    df = load_dataset("original.csv")

    print("=== Dataset Head ===")
    print(df.head())

    validation_results = validate(df)
    print("\n=== Dataset Validation ===")
    print(f"Shape: {validation_results['shape']}")
    print(f"Duplicates: {validation_results['duplicates']}")
    print("\n--- Null Counts ---")
    print(validation_results['null_counts'].to_string())
    print("\n--- Data Types ---")
    print(validation_results['dtypes'].to_string())

    num_stats = describe_numeric(df)
    print("\n=== Numerical Statistics ===")
    print(num_stats.to_string())

    cat_stats = describe_categorical(df)
    print("\n=== Categorical Statistics ===")
    print(cat_stats.to_string())

    df_clean = clean_pipeline(df)

    output_path = config.get_dataset_path("processed") + "/cleaned.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"\n✓ Saved cleaned dataset to: {output_path}")

    print("\n=== Cleaned Dataset Head ===")
    print(df_clean.head())

    print("\n" + "=" * 50)
    print("=== Generating Visualizations ===")
    print("=" * 50)

    print("\n>>> RAW visualizations...")
    viz_raw.run_raw()

    print("\n>>> CLEANED visualizations...")
    viz_cleaned.run_cleaned()

    print("\n>>> BIVARIATE visualizations...")
    viz_bivariate.run_bivariate()

    print("\n✓ All visualizations saved to: output/figures/")

    return {"validation": validation_results, "num_stats": num_stats, "cat_stats": cat_stats}


if __name__ == "__main__":
    run_eda()
