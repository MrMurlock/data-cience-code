from data_cience_code.data.load import load_dataset
from data_cience_code.eda.visualization import plot_scatter, plot_boxplot_grouped, plot_grouped_bar


def run_bivariate() -> None:
    """Generate bivariate/trivariate visualizations from cleaned dataset."""
    df = load_dataset("cleaned.csv", folder="processed")
    prefix = "cleaned_"

    print("=== BIVARIATE/TRIVARIATE Charts ===")

    print("\n1. Scatter: Sleep Duration vs Quality of Sleep")
    plot_scatter(df, x="Sleep Duration", y="Quality of Sleep", save=True, prefix=prefix)
    print(f"✓ saved: {prefix}scatter_duration_quality.png")

    print("\n2. Boxplot: Sleep Disorder vs Sleep Duration")
    plot_boxplot_grouped(df, x="Sleep Disorder", y="Sleep Duration", save=True, prefix=prefix)
    print(f"✓ saved: {prefix}box_disorder_duration.png")

    print("\n3. Scatter: Age vs Sleep Duration hue=Sleep Disorder")
    plot_scatter(df, x="Age", y="Sleep Duration", hue="Sleep Disorder", save=True, prefix=prefix)
    print(f"✓ saved: {prefix}scatter_age_disorder.png")

    print("\n4. Grouped Bar: Sleep Disorder by Gender")
    plot_grouped_bar(df, x="Sleep Disorder", hue="Gender", save=True, prefix=prefix)
    print(f"✓ saved: {prefix}bar_disorder_gender.png")


if __name__ == "__main__":
    run_bivariate()