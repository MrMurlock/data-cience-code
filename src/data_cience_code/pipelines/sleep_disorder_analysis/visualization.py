"""
Visualization functions specific to Sleep Disorder dataset.

This module contains visualization functions that are specific to the
Sleep Disorder dataset, including overlays with Sleep Disorder categories.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Markers for Sleep Disorder categories
SLEEP_DISORDER_MARKERS = {"None": "o", "Insomnia": "s", "Sleep Apnea": "^"}


def plot_clusters_3d_with_sleep_disorder(
    df_pca: pd.DataFrame,
    labels: np.ndarray,
    sleep_disorder: pd.Series,
    k: int,
    title: str,
) -> plt.Figure:
    """
    Genera scatter plot 3D coloreado por cluster con marcadores por Sleep Disorder.

    Args:
        df_pca: DataFrame con componentes PCA (PC1, PC2, PC3)
        labels: Etiquetas de cluster asignadas
        sleep_disorder: Serie con Sleep Disorder
        k: Número de clusters
        title: Título del gráfico

    Returns:
        Figure matplotlib
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Configurar estilo
    colors = plt.cm.tab10(np.arange(k) / 10)

    for cluster in range(k):
        for disorder in sleep_disorder.unique():
            mask = (labels == cluster) & (sleep_disorder == disorder)
            if mask.sum() > 0:
                ax.scatter(
                    df_pca.loc[mask, "PC1"],
                    df_pca.loc[mask, "PC2"],
                    df_pca.loc[mask, "PC3"],
                    c=[colors[cluster]],
                    marker=SLEEP_DISORDER_MARKERS.get(disorder, "o"),
                    label=f"Cluster {cluster} - {disorder}",
                    alpha=0.6,
                    s=50,
                )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    ax.legend(title="Cluster - Sleep Disorder", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    return fig


def plot_biplot_clusters_3d_with_sleep_disorder(
    df_pca: pd.DataFrame,
    loadings: pd.DataFrame,
    labels: np.ndarray,
    sleep_disorder: pd.Series,
    k: int,
    title: str,
) -> plt.Figure:
    """
    Genera biplot 3D con clusters y Sleep Disorder.

    Args:
        df_pca: DataFrame con componentes PCA (PC1, PC2, PC3)
        loadings: DataFrame con loadings
        labels: Etiquetas de cluster asignadas
        sleep_disorder: Serie con Sleep Disorder
        k: Número de clusters
        title: Título del gráfico

    Returns:
        Figure matplotlib
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Configurar estilo
    colors = plt.cm.tab10(np.arange(k) / 10)

    for cluster in range(k):
        for disorder in sleep_disorder.unique():
            mask = (labels == cluster) & (sleep_disorder == disorder)
            if mask.sum() > 0:
                ax.scatter(
                    df_pca.loc[mask, "PC1"],
                    df_pca.loc[mask, "PC2"],
                    df_pca.loc[mask, "PC3"],
                    c=[colors[cluster]],
                    marker=SLEEP_DISORDER_MARKERS.get(disorder, "o"),
                    label=f"Cluster {cluster} - {disorder}",
                    alpha=0.6,
                    s=50,
                )

    # Agregar loadings como vectores 3D
    scale_factor = np.max(np.abs(df_pca[["PC1", "PC2", "PC3"]])) * 0.8

    for var in loadings.index:
        x = loadings.loc[var, "PC1"] * scale_factor
        y = loadings.loc[var, "PC2"] * scale_factor
        z = loadings.loc[var, "PC3"] * scale_factor

        ax.quiver(
            0,
            0,
            0,
            x,
            y,
            z,
            color="red",
            alpha=0.5,
            arrow_length_ratio=0.1,
        )
        ax.text(x * 1.1, y * 1.1, z * 1.1, var, color="red", fontsize=9)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    ax.legend(title="Cluster - Sleep Disorder", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    return fig
