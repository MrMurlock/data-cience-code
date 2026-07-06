"""
Base classes for clustering visualizations.

This module provides generic visualization functions that can be used
across different clustering algorithms and datasets.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


class ClusterVisualizer:
    """Generic cluster visualization class."""

    @staticmethod
    def plot_clusters_3d(
        df_pca: pd.DataFrame,
        labels: np.ndarray,
        k: int,
        title: str,
        overlay: pd.Series | None = None,
        overlay_markers: dict | None = None,
    ) -> plt.Figure:
        """
        Genera scatter plot 3D coloreado por cluster con overlay opcional.

        Args:
            df_pca: DataFrame con componentes PCA (debe tener PC1, PC2, PC3)
            labels: Etiquetas de cluster asignadas
            k: Número de clusters
            title: Título del gráfico
            overlay: Serie opcional con variable categórica para overlay (e.g., target variable)
            overlay_markers: Diccionario opcional de marcadores para cada valor de overlay
                           Si None, usa marcadores por defecto

        Returns:
            Figure matplotlib
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Configurar estilo
        colors = plt.cm.tab10(np.arange(k) / 10)

        if overlay is not None and overlay_markers is not None:
            # Plot con overlay
            for cluster in range(k):
                for overlay_value in overlay.unique():
                    mask = (labels == cluster) & (overlay == overlay_value)
                    if mask.sum() > 0:
                        ax.scatter(
                            df_pca.loc[mask, "PC1"],
                            df_pca.loc[mask, "PC2"],
                            df_pca.loc[mask, "PC3"],
                            c=[colors[cluster]],
                            marker=overlay_markers.get(overlay_value, "o"),
                            label=f"Cluster {cluster} - {overlay_value}",
                            alpha=0.6,
                            s=50,
                        )
            ax.legend(title="Cluster - Overlay", bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            # Plot sin overlay
            for cluster in range(k):
                mask = labels == cluster
                ax.scatter(
                    df_pca.loc[mask, "PC1"],
                    df_pca.loc[mask, "PC2"],
                    df_pca.loc[mask, "PC3"],
                    c=[colors[cluster]],
                    label=f"Cluster {cluster}",
                    alpha=0.6,
                    s=50,
                )
            ax.legend(title="Clusters")

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title(title)
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_biplot_3d(
        df_pca: pd.DataFrame,
        loadings: pd.DataFrame,
        labels: np.ndarray,
        k: int,
        title: str,
        overlay: pd.Series | None = None,
        overlay_markers: dict | None = None,
    ) -> plt.Figure:
        """
        Genera biplot 3D con clusters y overlay opcional.

        Args:
            df_pca: DataFrame con componentes PCA (PC1, PC2, PC3)
            loadings: DataFrame con loadings
            labels: Etiquetas de cluster asignadas
            k: Número de clusters
            title: Título del gráfico
            overlay: Serie opcional con variable categórica para overlay
            overlay_markers: Diccionario opcional de marcadores para cada valor de overlay

        Returns:
            Figure matplotlib
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Configurar estilo
        colors = plt.cm.tab10(np.arange(k) / 10)

        if overlay is not None and overlay_markers is not None:
            # Plot con overlay
            for cluster in range(k):
                for overlay_value in overlay.unique():
                    mask = (labels == cluster) & (overlay == overlay_value)
                    if mask.sum() > 0:
                        ax.scatter(
                            df_pca.loc[mask, "PC1"],
                            df_pca.loc[mask, "PC2"],
                            df_pca.loc[mask, "PC3"],
                            c=[colors[cluster]],
                            marker=overlay_markers.get(overlay_value, "o"),
                            label=f"Cluster {cluster} - {overlay_value}",
                            alpha=0.6,
                            s=50,
                        )
            ax.legend(title="Cluster - Overlay", bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            # Plot sin overlay
            for cluster in range(k):
                mask = labels == cluster
                ax.scatter(
                    df_pca.loc[mask, "PC1"],
                    df_pca.loc[mask, "PC2"],
                    df_pca.loc[mask, "PC3"],
                    c=[colors[cluster]],
                    label=f"Cluster {cluster}",
                    alpha=0.6,
                    s=50,
                )
            ax.legend(title="Clusters")

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
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()
        return fig
