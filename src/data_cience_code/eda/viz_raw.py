from data_cience_code.data.load import load_dataset
from data_cience_code.eda.visualization import plot_boxplot, plot_histogram, plot_pie_chart, plot_ordinal_bar


NUMERIC_COLS = ["Age", "Sleep Duration", "Heart Rate", "Daily Steps"]
ORDINAL_COLS = ["Quality of Sleep"]
CATEGORICAL_COLS = ["Gender", "BMI Category", "Sleep Disorder"]
PREFIX = "raw_"

QUALITY_ORDER = [4, 5, 6, 7, 8, 9]


def run_raw() -> None:
    """Generate visualizations from raw dataset (before cleaning)."""
    df = load_dataset("original.csv")

    print("=== RAW Dataset - Boxplots ===")
    for col in NUMERIC_COLS + ORDINAL_COLS:
        plot_boxplot(df, col, save=True, prefix=PREFIX)
        print(f"✓ saved: {PREFIX}box_{col.replace(' ', '_')}.png")

    print("\n=== RAW Dataset - Histograms ===")
    for col in NUMERIC_COLS:
        plot_histogram(df, col, save=True, prefix=PREFIX)
        print(f"✓ saved: {PREFIX}hist_{col.replace(' ', '_')}.png")

    print("\n=== RAW Dataset - Ordinal Bar Charts ===")
    for col in ORDINAL_COLS:
        plot_ordinal_bar(df, col, save=True, prefix=PREFIX, order=QUALITY_ORDER)
        print(f"✓ saved: {PREFIX}bar_ordinal_{col.replace(' ', '_')}.png")

    print("\n=== RAW Dataset - Pie Charts ===")
    for col in CATEGORICAL_COLS:
        plot_pie_chart(df, col, save=True, prefix=PREFIX)
        print(f"✓ saved: {PREFIX}pie_{col.replace(' ', '_')}.png")


if __name__ == "__main__":
    run_raw()