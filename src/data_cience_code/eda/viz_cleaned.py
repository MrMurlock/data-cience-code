from data.load import load_dataset
from eda.visualization import plot_boxplot, plot_histogram, plot_pie_chart, plot_correlation_heatmap, plot_ordinal_bar


NUMERIC_COLS = ["Age", "Sleep Duration", "Heart Rate", "Daily Steps", "bp_systolic", "bp_diastolic"]
ORDINAL_COLS = ["Quality of Sleep"]
CATEGORICAL_COLS = ["Gender", "BMI Category", "Sleep Disorder", "age_group", "quality_group"]
PREFIX = "cleaned_"

QUALITY_ORDER = [4, 5, 6, 7, 8, 9]
AGE_GROUP_ORDER = ["Young", "Middle", "Senior"]
QUALITY_GROUP_ORDER = ["Low", "Medium", "High"]


def run_cleaned() -> None:
    """Generate visualizations from cleaned dataset (after cleaning)."""
    df = load_dataset("cleaned.csv", folder="processed")

    print("=== CLEANED Dataset - Boxplots ===")
    for col in NUMERIC_COLS + ORDINAL_COLS:
        plot_boxplot(df, col, save=True, prefix=PREFIX)
        print(f"✓ saved: {PREFIX}box_{col.replace(' ', '_')}.png")

    print("\n=== CLEANED Dataset - Histograms ===")
    for col in NUMERIC_COLS:
        plot_histogram(df, col, save=True, prefix=PREFIX)
        print(f"✓ saved: {PREFIX}hist_{col.replace(' ', '_')}.png")

    print("\n=== CLEANED Dataset - Ordinal Bar Charts ===")
    for col in ORDINAL_COLS:
        plot_ordinal_bar(df, col, save=True, prefix=PREFIX, order=QUALITY_ORDER)
        print(f"✓ saved: {PREFIX}bar_ordinal_{col.replace(' ', '_')}.png")

    print("\n=== CLEANED Dataset - Pie Charts ===")
    for col in CATEGORICAL_COLS:
        plot_pie_chart(df, col, save=True, prefix=PREFIX)
        print(f"✓ saved: {PREFIX}pie_{col.replace(' ', '_')}.png")

    print("\n=== CLEANED Dataset - Correlation Heatmap ===")
    plot_correlation_heatmap(df, save=True, prefix=PREFIX)
    print(f"✓ saved: {PREFIX}correlation_heatmap.png")


if __name__ == "__main__":
    run_cleaned()