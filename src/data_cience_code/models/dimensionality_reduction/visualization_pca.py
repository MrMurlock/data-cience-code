from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

import data_cience_code.config.config as config


def plot_biplot_2d(
    df_pca: pd.DataFrame,
    loadings: pd.DataFrame,
    target: pd.Series,
    pca_metrics: dict,
    title: str = "Biplot PCA 2D",
) -> plt.Figure:
    """
    Crea biplot 2D con PC1 vs PC2 mostrando loadings como vectores.

    Args:
        df_pca: DataFrame con componentes PCA (debe tener PC1 y PC2)
        loadings: DataFrame con loadings (variables originales x componentes)
        target: Serie con variable categórica para colorear puntos
        pca_metrics: Diccionario con métricas del PCA
        title: Título del gráfico

    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Configurar estilo
    sns.set_palette("husl")

    # Colorear puntos por Sleep Disorder
    unique_targets = target.unique()
    colors = sns.color_palette("husl", len(unique_targets))
    color_map = dict(zip(unique_targets, colors))

    for t in unique_targets:
        mask = target == t
        ax.scatter(
            df_pca.loc[mask, "PC1"],
            df_pca.loc[mask, "PC2"],
            c=[color_map[t]],
            label=t,
            alpha=0.6,
            s=50,
        )

    # Agregar loadings como vectores
    # Escalar loadings para que sean visibles
    scale_factor = np.max(np.abs(df_pca[["PC1", "PC2"]])) * 0.8

    for var in loadings.index:
        # Usar solo PC1 y PC2 para biplot 2D
        x = loadings.loc[var, "PC1"] * scale_factor
        y = loadings.loc[var, "PC2"] * scale_factor

        ax.arrow(
            0,
            0,
            x,
            y,
            color="red",
            alpha=0.5,
            head_width=scale_factor * 0.05,
            head_length=scale_factor * 0.05,
        )
        ax.text(
            x * 1.1,
            y * 1.1,
            var,
            color="red",
            fontsize=9,
            ha="center",
            va="center",
        )

    # Usar varianza explicada de las métricas
    pc1_var = pca_metrics["explained_variance_ratio"][0]
    pc2_var = pca_metrics["explained_variance_ratio"][1]

    ax.set_xlabel(f"PC1 ({pc1_var:.2%})")
    ax.set_ylabel(f"PC2 ({pc2_var:.2%})")
    ax.set_title(title)
    ax.legend(title="Sleep Disorder")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.axvline(x=0, color="k", linestyle="--", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_biplot_3d(
    df_pca: pd.DataFrame,
    loadings: pd.DataFrame,
    target: pd.Series,
    title: str = "Biplot PCA 3D",
) -> plt.Figure:
    """
    Crea biplot 3D con PC1 vs PC2 vs PC3 mostrando loadings como vectores.

    Args:
        df_pca: DataFrame con componentes PCA (debe tener PC1, PC2, PC3)
        loadings: DataFrame con loadings (variables originales x componentes)
        target: Serie con variable categórica para colorear puntos
        title: Título del gráfico

    Returns:
        Figure matplotlib
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Configurar estilo
    sns.set_palette("husl")

    # Colorear puntos por Sleep Disorder
    unique_targets = target.unique()
    colors = sns.color_palette("husl", len(unique_targets))
    color_map = dict(zip(unique_targets, colors))

    for t in unique_targets:
        mask = target == t
        ax.scatter(
            df_pca.loc[mask, "PC1"],
            df_pca.loc[mask, "PC2"],
            df_pca.loc[mask, "PC3"],
            c=[color_map[t]],
            label=t,
            alpha=0.6,
            s=50,
        )

    # Agregar loadings como vectores 3D
    scale_factor = np.max(np.abs(df_pca[["PC1", "PC2", "PC3"]])) * 0.8

    for var in loadings.index:
        # Usar PC1, PC2, PC3 para biplot 3D
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
        ax.text(
            x * 1.1,
            y * 1.1,
            z * 1.1,
            var,
            color="red",
            fontsize=9,
        )

    ax.set_xlabel(f"PC1")
    ax.set_ylabel(f"PC2")
    ax.set_zlabel(f"PC3")
    ax.set_title(title)
    ax.legend(title="Sleep Disorder")

    # Ajustar ángulo de vista
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    return fig


def run_pca_visualizations() -> dict:
    """
    Ejecuta el pipeline completo de visualizaciones PCA.

    Returns:
        Diccionario con paths de figuras generadas
    """
    # Cargar componentes PCA
    pca_components_path = config.get_dataset_path("processed") + "/pca_components.csv"
    df_pca = pd.read_csv(pca_components_path)

    # Cargar loadings
    loadings_path = config.get_dataset_path("processed") + "/pca_loadings.csv"
    loadings = pd.read_csv(loadings_path, index_col=0)

    # Cargar métricas PCA
    import json

    pca_reports_path = config.get_output_path("reports", "pca")
    pca_metrics_path = pca_reports_path + "/pca_metrics.json"
    with open(pca_metrics_path, "r") as f:
        pca_metrics = json.load(f)

    # Cargar dataset normalizado para obtener Sleep Disorder
    normalized_path = config.get_dataset_path("processed") + "/normalized.csv"
    df_normalized = pd.read_csv(normalized_path)
    target = df_normalized["Sleep Disorder"]

    # Crear directorio de salida
    figures_path = config.get_output_path("figures", "pca")
    Path(figures_path).mkdir(parents=True, exist_ok=True)

    # Configurar estilo
    plt.style.use("seaborn-v0_8-whitegrid")

    # Generar biplot 2D
    fig_2d = plot_biplot_2d(
        df_pca,
        loadings,
        target,
        pca_metrics,
        "Biplot PCA 2D (PC1 vs PC2)",
    )
    biplot_2d_path = figures_path + "/biplot_2d_pca.png"
    fig_2d.savefig(biplot_2d_path, dpi=300, bbox_inches="tight")
    plt.close(fig_2d)

    # Generar biplot 3D
    fig_3d = plot_biplot_3d(df_pca, loadings, target, "Biplot PCA 3D (PC1 vs PC2 vs PC3)")
    biplot_3d_path = figures_path + "/biplot_3d_pca.png"
    fig_3d.savefig(biplot_3d_path, dpi=300, bbox_inches="tight")
    plt.close(fig_3d)

    # Retornar resultados
    results = {
        "biplot_2d_path": biplot_2d_path,
        "biplot_3d_path": biplot_3d_path,
    }

    return results
