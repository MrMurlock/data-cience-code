from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

import config.config as config
from .kmeans import (
    plot_biplot_clusters_3d,
    plot_biplot_clusters_3d_with_sleep_disorder,
    plot_clusters_3d,
    plot_clusters_3d_with_sleep_disorder,
)


def plot_k_distance(df_pca: pd.DataFrame, k: int, title: str) -> plt.Figure:
    """
    Genera gráfico de distancias al k-ésimo vecino más cercano
    para determinar eps óptimo en DBSCAN.

    Args:
        df_pca: DataFrame con componentes PCA
        k: Número de vecinos (generalmente min_samples)
        title: Título del gráfico

    Returns:
        Figure matplotlib con distancias ordenadas
    """
    # Calcular distancias al k-ésimo vecino más cercano
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(df_pca)
    distances, _ = neighbors_fit.kneighbors(df_pca)

    # Obtener distancia al k-ésimo vecino
    k_distances = distances[:, k - 1]
    k_distances_sorted = np.sort(k_distances)[::-1]

    # Generar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(k_distances_sorted) + 1), k_distances_sorted, "b-", linewidth=2)
    ax.set_xlabel("Points sorted by distance", fontsize=12)
    ax.set_ylabel(f"{k}-NN Distance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def find_optimal_eps(
    df_pca: pd.DataFrame, min_samples: int, k_distance_k: int | None = None
) -> float:
    """
    Encuentra eps óptimo usando el método del codo en gráfico k-distance.

    Args:
        df_pca: DataFrame con componentes PCA
        min_samples: Valor de min_samples para DBSCAN
        k_distance_k: k para k-distance (default: min_samples)

    Returns:
        Valor de eps óptimo
    """
    k = k_distance_k if k_distance_k is not None else min_samples

    # Calcular distancias al k-ésimo vecino más cercano
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(df_pca)
    distances, _ = neighbors_fit.kneighbors(df_pca)

    # Obtener distancia al k-ésimo vecino
    k_distances = distances[:, k - 1]
    k_distances_sorted = np.sort(k_distances)[::-1]

    # Método simple: usar el valor en el punto de inflexión
    # Usamos el valor en el percentil 90 como aproximación del codo
    eps = np.percentile(k_distances_sorted, 90)

    return eps


def train_dbscan(
    df_pca: pd.DataFrame, eps: float, min_samples: int
) -> tuple[DBSCAN, np.ndarray]:
    """
    Entrena DBSCAN con parámetros específicos.

    Args:
        df_pca: DataFrame con componentes PCA (PC1, PC2, PC3)
        eps: Radio del vecindario
        min_samples: Mínimo de muestras en vecindario

    Returns:
        Tuple con modelo y etiquetas
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(df_pca)
    return dbscan, labels


def run_dbscan_clustering(
    eps: float, min_samples: int, generate_k_distance: bool = True
) -> dict:
    """
    Ejecuta pipeline completo de DBSCAN clustering y visualización.

    Args:
        eps: Radio del vecindario
        min_samples: Mínimo de muestras en vecindario
        generate_k_distance: Si True, genera gráfico k-distance

    Returns:
        Diccionario con resultados y métricas
    """
    # Cargar componentes PCA
    pca_components_path = config.get_dataset_path("processed") + "/pca_components.csv"
    df_pca = pd.read_csv(pca_components_path)

    # Extraer solo PC1, PC2, PC3
    df_pca_3d = df_pca[["PC1", "PC2", "PC3"]]

    # Cargar loadings
    loadings_path = config.get_dataset_path("processed") + "/pca_loadings.csv"
    loadings = pd.read_csv(loadings_path, index_col=0)

    # Cargar dataset normalizado para obtener Sleep Disorder
    normalized_path = config.get_dataset_path("processed") + "/normalized.csv"
    df_normalized = pd.read_csv(normalized_path)
    sleep_disorder = df_normalized["Sleep Disorder"]

    # Configurar estilo
    plt.style.use("seaborn-v0_8-whitegrid")

    # Crear directorio de salida
    figures_path = config.get_output_path("figures", "clustering/dbscan")
    reports_path = config.get_output_path("reports", "clustering/dbscan/metrics")
    Path(figures_path).mkdir(parents=True, exist_ok=True)
    Path(reports_path).mkdir(parents=True, exist_ok=True)

    results = {}

    # Generar k-distance plot si se solicita
    if generate_k_distance:
        k_distance_k = min_samples
        fig_kdist = plot_k_distance(
            df_pca_3d,
            k_distance_k,
            f"K-Distance Graph (k={k_distance_k}) for DBSCAN",
        )
        kdist_path = figures_path + "/k_distance.png"
        fig_kdist.savefig(kdist_path, dpi=300, bbox_inches="tight")
        plt.close(fig_kdist)
        results["k_distance_plot_path"] = kdist_path

    # Entrenar DBSCAN
    dbscan_model, labels = train_dbscan(df_pca_3d, eps, min_samples)

    # Calcular estadísticas de clustering
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    # Calcular silhouette score (solo para puntos no noise)
    if n_clusters > 1 and n_noise < len(labels):
        mask = labels != -1
        if mask.sum() > 1:
            silhouette = silhouette_score(df_pca_3d.loc[mask], labels[mask])
        else:
            silhouette = None
    else:
        silhouette = None

    # Guardar métricas en CSV
    metrics_df = pd.DataFrame(
        {
            "eps": [eps],
            "min_samples": [min_samples],
            "silhouette_score": [silhouette],
            "n_clusters": [n_clusters],
            "n_noise": [n_noise],
        }
    )
    metrics_path = reports_path + "/silhouette_scores.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Guardar labels en CSV
    labels_df = pd.DataFrame({"cluster": labels})
    labels_path = config.get_dataset_path("processed") + "/dbscan_labels.csv"
    labels_df.to_csv(labels_path, index=False)
    results["labels_path"] = labels_path

    # Generar visualizaciones si hay clusters
    if n_clusters > 0:
        # Ajustar etiquetas para visualización (noise -> -1)
        # Usar colores consistentes con k-means y jerárquico
        k_visual = n_clusters + 1  # +1 para noise
        colors = plt.cm.tab10(np.arange(k_visual) / 10)

        # Scatter 3D
        fig_scatter = plot_clusters_3d(
            df_pca_3d, labels, k_visual, f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})"
        )
        scatter_path = figures_path + "/scatter_3d_clusters.png"
        fig_scatter.savefig(scatter_path, dpi=300, bbox_inches="tight")
        plt.close(fig_scatter)

        # Scatter 3D + Sleep Disorder
        fig_scatter_sd = plot_clusters_3d_with_sleep_disorder(
            df_pca_3d,
            labels,
            sleep_disorder,
            k_visual,
            f"DBSCAN Clustering (eps={eps}, min_samples={min_samples}) with Sleep Disorder",
        )
        scatter_sd_path = figures_path + "/scatter_3d_clusters_sleep_disorder.png"
        fig_scatter_sd.savefig(scatter_sd_path, dpi=300, bbox_inches="tight")
        plt.close(fig_scatter_sd)

        # Biplot 3D
        fig_biplot = plot_biplot_clusters_3d(
            df_pca_3d,
            loadings,
            labels,
            k_visual,
            f"Biplot DBSCAN Clustering (eps={eps}, min_samples={min_samples})",
        )
        biplot_path = figures_path + "/biplot_3d_clusters.png"
        fig_biplot.savefig(biplot_path, dpi=300, bbox_inches="tight")
        plt.close(fig_biplot)

        # Biplot 3D + Sleep Disorder
        fig_biplot_sd = plot_biplot_clusters_3d_with_sleep_disorder(
            df_pca_3d,
            loadings,
            labels,
            sleep_disorder,
            k_visual,
            f"Biplot DBSCAN Clustering (eps={eps}, min_samples={min_samples}) with Sleep Disorder",
        )
        biplot_sd_path = figures_path + "/biplot_3d_clusters_sleep_disorder.png"
        fig_biplot_sd.savefig(biplot_sd_path, dpi=300, bbox_inches="tight")
        plt.close(fig_biplot_sd)

        results["scatter_path"] = scatter_path
        results["scatter_sd_path"] = scatter_sd_path
        results["biplot_path"] = biplot_path
        results["biplot_sd_path"] = biplot_sd_path

    # Retornar resultados
    results.update(
        {
            "eps": eps,
            "min_samples": min_samples,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "silhouette_score": silhouette,
            "metrics_path": metrics_path,
        }
    )

    return results
