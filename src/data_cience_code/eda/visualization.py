import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import data_cience_code.config.config as config


plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 6)


def plot_histogram(df: pd.DataFrame, column: str, save: bool = False, prefix: str = "") -> None:
    """Plot histogram for a numeric column."""
    fig, ax = plt.subplots()
    df[column].hist(bins=30, ax=ax)
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    if save:
        filename = f"{prefix}hist_{column}.png".replace(" ", "_")
        fig.savefig(config.get_output_path("figures", subfolder="eda") + f"/{filename}", dpi=150)
    plt.close()


def plot_boxplot(df: pd.DataFrame, column: str, save: bool = False, prefix: str = "") -> None:
    """Plot boxplot for a numeric column using seaborn."""
    fig, ax = plt.subplots()
    sns.boxplot(data=df, y=column, ax=ax)
    ax.set_title(f"Boxplot of {column}")
    plt.tight_layout()
    if save:
        filename = f"{prefix}box_{column}.png".replace(" ", "_")
        fig.savefig(config.get_output_path("figures", subfolder="eda") + f"/{filename}", dpi=150)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, save: bool = False, prefix: str = "") -> None:
    """Plot correlation heatmap for numeric columns."""
    corr = df.select_dtypes(include=["number"]).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    if save:
        filename = f"{prefix}correlation_heatmap.png"
        fig.savefig(config.get_output_path("figures", subfolder="eda") + f"/{filename}", dpi=150)
    plt.close()


def plot_bar_chart(df: pd.DataFrame, column: str, save: bool = False, prefix: str = "") -> None:
    """Plot bar chart for a categorical column."""
    fig, ax = plt.subplots()
    df[column].value_counts().plot(kind="bar", ax=ax)
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save:
        filename = f"{prefix}bar_{column}.png".replace(" ", "_")
        fig.savefig(config.get_output_path("figures", subfolder="eda") + f"/{filename}", dpi=150)
    plt.close()


def plot_ordinal_bar(df: pd.DataFrame, column: str, save: bool = False, prefix: str = "", order: list[int] = None) -> None:
    """Plot bar chart for ordinal categorical column with specified order."""
    fig, ax = plt.subplots()
    if order is not None:
        vc = df[column].value_counts().reindex(order, fill_value=0)
    else:
        vc = df[column].value_counts()
    vc.plot(kind="bar", ax=ax, color=sns.palettes.color_palette("husl", len(vc)))
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    if save:
        filename = f"{prefix}bar_ordinal_{column}.png".replace(" ", "_")
        fig.savefig(config.get_output_path("figures", subfolder="eda") + f"/{filename}", dpi=150)
    plt.close()


def plot_pie_chart(df: pd.DataFrame, column: str, save: bool = False, prefix: str = "") -> None:
    """Plot pie chart for a categorical column."""
    fig, ax = plt.subplots()
    df[column].value_counts().plot(kind="pie", ax=ax, autopct="%1.1f%%")
    ax.set_title(f"Distribution of {column}")
    ax.set_ylabel("")
    plt.tight_layout()
    if save:
        filename = f"{prefix}pie_{column}.png".replace(" ", "_")
        fig.savefig(config.get_output_path("figures", subfolder="eda") + f"/{filename}", dpi=150)
    plt.close()


def plot_scatter(df: pd.DataFrame, x: str, y: str, save: bool = False, prefix: str = "", hue: str = None) -> None:
    """Plot scatter plot with optional hue."""
    fig, ax = plt.subplots()
    if hue:
        sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)
    else:
        sns.scatterplot(data=df, x=x, y=y, ax=ax)
    ax.set_title(f"{y} vs {x}" + (f" by {hue}" if hue else ""))
    plt.tight_layout()
    if save:
        filename = f"{prefix}scatter_{x}_{y}" + (f"_by_{hue}" if hue else "") + ".png"
        filename = filename.replace(" ", "_")
        fig.savefig(config.get_output_path("figures", subfolder="eda") + f"/{filename}", dpi=150)
    plt.close()


def plot_boxplot_grouped(df: pd.DataFrame, x: str, y: str, save: bool = False, prefix: str = "", hue: str = None) -> None:
    """Plot grouped boxplot."""
    fig, ax = plt.subplots()
    if hue:
        sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax)
    else:
        sns.boxplot(data=df, x=x, y=y, ax=ax)
    ax.set_title(f"{y} by {x}" + (f" and {hue}" if hue else ""))
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save:
        filename = f"{prefix}box_{x}_{y}" + (f"_by_{hue}" if hue else "") + ".png"
        filename = filename.replace(" ", "_")
        fig.savefig(config.get_output_path("figures", subfolder="eda") + f"/{filename}", dpi=150)
    plt.close()


def plot_grouped_bar(df: pd.DataFrame, x: str, save: bool = False, prefix: str = "", hue: str = None) -> None:
    """Plot grouped bar chart (crosstab)."""
    fig, ax = plt.subplots()
    if hue:
        ct = pd.crosstab(df[x], df[hue])
        ct.plot(kind="bar", ax=ax, stacked=False)
    else:
        df[x].value_counts().plot(kind="bar", ax=ax)
    ax.set_title(f"{x}" + (f" by {hue}" if hue else ""))
    ax.set_xlabel(x)
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save:
        filename = f"{prefix}bar_{x}_by_{hue}.png".replace(" ", "_")
        fig.savefig(config.get_output_path("figures", subfolder="eda") + f"/{filename}", dpi=150)
    plt.close()