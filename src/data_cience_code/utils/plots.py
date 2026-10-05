import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _style():
    sns.set_theme(style="whitegrid")


def hist_plot(df, col, bins=20, title=None):
    _style()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[col].dropna(), bins=bins, kde=True, ax=ax)
    ax.set_title(title or f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    return fig, ax


def box_plot(df, col, title=None):
    _style()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=df[col].dropna(), ax=ax)
    ax.set_title(title or f"Boxplot of {col}")
    ax.set_xlabel(col)
    return fig, ax


def bar_plot(df, col, title=None, max_cats=30):
    _style()
    counts = df[col].dropna().value_counts().sort_values(ascending=False)
    if len(counts) > max_cats:
        counts = counts.head(max_cats)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.values, y=counts.index, ax=ax)
    ax.set_title(title or f"Distribution of {col}")
    ax.set_xlabel("Count")
    ax.set_ylabel(col)
    return fig, ax


def scatter_plot(df, x_col, y_col, title=None, annotate_outliers=True):
    """Scatter plot bivariado.

    Marca con 'x' roja los puntos fuera de las vallas 1.5*IQR de cualquiera
    de las dos variables, para hacer visible el tratamiento de outliers (§4.1).
    """
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    data = df[[x_col, y_col]].dropna()
    sns.scatterplot(data=data, x=x_col, y=y_col, alpha=0.55, ax=ax)
    if annotate_outliers:
        mask = pd.Series(False, index=data.index)
        for col in (x_col, y_col):
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            mask = mask | (data[col] < q1 - 1.5 * iqr) | (data[col] > q3 + 1.5 * iqr)
        if mask.any():
            sns.scatterplot(
                data=data[mask],
                x=x_col,
                y=y_col,
                color="red",
                marker="x",
                s=80,
                ax=ax,
                legend=False,
            )
    ax.set_title(title or f"{y_col} vs {x_col}")
    return fig, ax


def group_box_plot(df, cat_col, num_col, title=None):
    """Boxplot de una variable numérica agrupada por una categórica (§4.2)."""
    _style()
    order = df[cat_col].value_counts().index.tolist()
    wide = len(order) > 6
    fig, ax = plt.subplots(figsize=(8, 5) if wide else (6.4, 4.4))
    sns.boxplot(data=df, x=cat_col, y=num_col, order=order, ax=ax)
    ax.set_title(title or f"{num_col} by {cat_col}")
    if wide:
        ax.tick_params(axis="x", rotation=30)
    return fig, ax


def corr_heatmap(df, title=None, method: str = "pearson"):
    """Heatmap de la matriz de correlación (pearson o spearman) entre todas las numéricas."""
    _style()
    corr = df.select_dtypes(include=[np.number]).corr(method=method)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title(title or f"Matriz de correlación ({method.capitalize()})")
    return fig, ax


def contingency_heatmap(df, x_col, y_col, title=None):
    """Heatmap de proporciones por fila (%): distribución de y_col dentro de cada x_col."""
    _style()
    prop = pd.crosstab(df[x_col], df[y_col], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(prop, annot=True, fmt=".1f", cmap="Blues", ax=ax)
    ax.set_ylabel(x_col)
    ax.set_xlabel(y_col)
    ax.set_title(title or f"Share of {y_col} within each {x_col} (%)")
    return fig, ax
