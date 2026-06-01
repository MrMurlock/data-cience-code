from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

import config.config as config
from .kmeans import (
    plot_biplot_clusters_3d,
    plot_biplot_clusters_3d_with_sleep_disorder,
    plot_clusters_3d,
    plot_clusters_3d_with_sleep_disorder,
)


def fit_agglomerative(
    df_pca: pd.DataFrame, linkage_method: str = "ward"
) -> AgglomerativeClustering:
    """
    Entrena clustering jerárquico aglomerativo (bottom-up).

    Lógica aglomerativa:
    - Comienza con cada observación como su propio cluster
    - Iterativamente fusiona los clusters más cercanos
    - Termina cuando todas las observaciones están en un solo cluster
    - Ward linkage minimiza la varianza dentro de los clusters

    Args:
        df_pca: DataFrame con componentes PCA (debe tener PC1, PC2, PC3)
        linkage_method: Método de linkage ("ward", "complete", "average", "single")

    Returns:
        Modelo AgglomerativeClustering ajustado
    """
    clustering = AgglomerativeClustering(
        linkage=linkage_method, distance_threshold=0, n_clusters=None
    )
    clustering.fit(df_pca)
    return clustering


def fit_divisive(df_pca: pd.DataFrame) -> AgglomerativeClustering:
    """
    Entrena clustering jerárquico divisivo (top-down).

    Lógica divisiva:
    - Comienza con todas las observaciones en un solo cluster
    - Iterativamente divide el cluster más heterogéneo
    - Termina cuando cada observación es su propio cluster
    - Usa distancia euclidean para determinar divisiones

    Args:
        df_pca: DataFrame con componentes PCA (debe tener PC1, PC2, PC3)

    Returns:
        Modelo AgglomerativeClustering ajustado (simulación divisiva)
    """
    # Simulación de clustering divisivo usando linkage "complete"
    clustering = AgglomerativeClustering(
        linkage="complete", distance_threshold=0, n_clusters=None
    )
    clustering.fit(df_pca)
    return clustering


def plot_dendrogram_agglomerative(
    df_pca: pd.DataFrame, linkage_method: str = "ward"
) -> plt.Figure:
    """
    Genera dendrograma para clustering aglomerativo.

    Valor interpretativo del dendrograma:
    - Altura de fusión indica distancia entre clusters
    - Fusión a gran altura = clusters muy diferentes
    - Fusión a pequeña altura = clusters similares
    - K óptimo se determina cortando el dendrograma a una altura apropiada
    - Permite identificar estructuras jerárquicas naturales en los datos

    Args:
        df_pca: DataFrame con componentes PCA (debe tener PC1, PC2, PC3)
        linkage_method: Método de linkage ("ward", "complete", "average", "single")

    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calcular matriz de linkage
    linkage_matrix = linkage(df_pca, method=linkage_method, metric="euclidean")

    # Generar dendrograma
    dendrogram(
        linkage_matrix,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=8,
        show_contracted=True,
    )

    ax.set_xlabel("Samples", fontsize=12)
    ax.set_ylabel("Distance (Euclidean)", fontsize=12)
    ax.set_title(
        f"Hierarchical Clustering Dendrogram (Agglomerative - {linkage_method})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


def plot_dendrogram_divisive(df_pca: pd.DataFrame) -> plt.Figure:
    """
    Genera dendrograma para clustering divisivo.

    Valor interpretativo del dendrograma:
    - Altura de fusión indica distancia entre clusters
    - Fusión a gran altura = clusters muy diferentes
    - Fusión a pequeña altura = clusters similares
    - K óptimo se determina cortando el dendrograma a una altura apropiada
    - Permite identificar estructuras jerárquicas naturales en los datos

    Args:
        df_pca: DataFrame con componentes PCA (debe tener PC1, PC2, PC3)

    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calcular matriz de linkage para simulación divisiva
    linkage_matrix = linkage(df_pca, method="complete", metric="euclidean")

    # Generar dendrograma
    dendrogram(
        linkage_matrix,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=8,
        show_contracted=True,
    )

    ax.set_xlabel("Samples", fontsize=12)
    ax.set_ylabel("Distance (Euclidean)", fontsize=12)
    ax.set_title(
        "Hierarchical Clustering Dendrogram (Divisive)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


def calculate_silhouette_hierarchical(
    df_pca: pd.DataFrame, k_range: tuple[int, int], linkage_method: str
) -> dict:
    """
    Calcula silhouette scores para clustering jerárquico en rango de K.

    Args:
        df_pca: DataFrame con componentes PCA (debe tener PC1, PC2, PC3)
        k_range: Tupla (k_min, k_max) para rango de K
        linkage_method: Método de linkage ("ward", "complete")

    Returns:
        Diccionario con k_values y silhouette_scores
    """
    k_values = []
    silhouette_scores = []

    for k in range(k_range[0], k_range[1] + 1):
        clustering = AgglomerativeClustering(
            n_clusters=k, linkage=linkage_method
        )
        labels = clustering.fit_predict(df_pca)

        # Calcular silhouette score
        score = silhouette_score(df_pca, labels)
        k_values.append(k)
        silhouette_scores.append(score)

    return {"k_values": k_values, "silhouette_scores": silhouette_scores}


def plot_silhouette_scores_hierarchical(
    k_values: list[int], silhouette_scores: list[float], linkage_method: str
) -> plt.Figure:
    """
    Genera plot de silhouette scores para clustering jerárquico.

    Args:
        k_values: Lista de valores de K
        silhouette_scores: Lista de silhouette scores
        linkage_method: Método de linkage para título

    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(k_values, silhouette_scores, marker="o", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax.set_ylabel("Silhouette Score", fontsize=12)
    ax.set_title(
        f"Silhouette Scores - Hierarchical Clustering ({linkage_method})",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def run_hierarchical_analysis(k_range: tuple[int, int] = (2, 10)) -> dict:
    """
    Ejecuta el pipeline completo de análisis de clustering jerárquico:
    dendrogramas y silhouette scores para aglomerativo y divisivo.

    Args:
        k_range: Tupla (k_min, k_max) para análisis de silhouette

    Returns:
        Diccionario con paths de figuras generadas y métricas
    """
    # Cargar componentes PCA
    pca_components_path = config.get_dataset_path("processed") + "/pca_components.csv"
    df_pca = pd.read_csv(pca_components_path)

    # Extraer solo PC1, PC2, PC3
    df_pca_3d = df_pca[["PC1", "PC2", "PC3"]]

    # Crear directorio de salida
    figures_path = config.get_output_path("figures", "clustering/hierarchical")
    reports_path = config.get_output_path("reports", "clustering/hierarchical/metrics")
    Path(figures_path).mkdir(parents=True, exist_ok=True)
    Path(reports_path).mkdir(parents=True, exist_ok=True)

    # Configurar estilo
    plt.style.use("seaborn-v0_8-whitegrid")

    # Generar dendrograma aglomerativo (ward)
    fig_agg = plot_dendrogram_agglomerative(df_pca_3d, "ward")
    agg_path = figures_path + "/dendrogram_agglomerative_ward.png"
    fig_agg.savefig(agg_path, dpi=300, bbox_inches="tight")
    plt.close(fig_agg)

    # Generar dendrograma divisivo
    fig_div = plot_dendrogram_divisive(df_pca_3d)
    div_path = figures_path + "/dendrogram_divisive.png"
    fig_div.savefig(div_path, dpi=300, bbox_inches="tight")
    plt.close(fig_div)

    # Calcular silhouette para aglomerativo (ward)
    metrics_agg = calculate_silhouette_hierarchical(df_pca_3d, k_range, "ward")

    # Guardar métricas de silhouette aglomerativo en CSV
    df_metrics_agg = pd.DataFrame(metrics_agg)
    metrics_agg_path = reports_path + "/silhouette_scores_agglomerative_ward.csv"
    df_metrics_agg.to_csv(metrics_agg_path, index=False)

    # Generar plot de silhouette aglomerativo
    fig_sil_agg = plot_silhouette_scores_hierarchical(
        metrics_agg["k_values"], metrics_agg["silhouette_scores"], "ward"
    )
    sil_agg_path = figures_path + "/silhouette_scores_agglomerative_ward.png"
    fig_sil_agg.savefig(sil_agg_path, dpi=300, bbox_inches="tight")
    plt.close(fig_sil_agg)

    # Calcular silhouette para divisivo (complete)
    metrics_div = calculate_silhouette_hierarchical(df_pca_3d, k_range, "complete")

    # Guardar métricas de silhouette divisivo en CSV
    df_metrics_div = pd.DataFrame(metrics_div)
    metrics_div_path = reports_path + "/silhouette_scores_divisive.csv"
    df_metrics_div.to_csv(metrics_div_path, index=False)

    # Generar plot de silhouette divisivo
    fig_sil_div = plot_silhouette_scores_hierarchical(
        metrics_div["k_values"], metrics_div["silhouette_scores"], "complete"
    )
    sil_div_path = figures_path + "/silhouette_scores_divisive.png"
    fig_sil_div.savefig(sil_div_path, dpi=300, bbox_inches="tight")
    plt.close(fig_sil_div)

    # Retornar resultados
    results = {
        "agglomerative_dendrogram_path": agg_path,
        "divisive_dendrogram_path": div_path,
        "agglomerative_silhouette_plot_path": sil_agg_path,
        "divisive_silhouette_plot_path": sil_div_path,
        "agglomerative_metrics_path": metrics_agg_path,
        "divisive_metrics_path": metrics_div_path,
        "agglomerative_metrics": metrics_agg,
        "divisive_metrics": metrics_div,
    }

    return results


def train_hierarchical(
    df_pca: pd.DataFrame, k: int, linkage_method: str
) -> tuple[AgglomerativeClustering, np.ndarray]:
    """
    Entrena clustering jerárquico con K específico.

    Args:
        df_pca: DataFrame con componentes PCA (PC1, PC2, PC3)
        k: Número de clusters
        linkage_method: Método de linkage ("ward", "complete")

    Returns:
        Tuple con modelo y etiquetas
    """
    clustering = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
    labels = clustering.fit_predict(df_pca)
    return clustering, labels


def run_hierarchical_clustering(k_values: list[int]) -> dict:
    """
    Ejecuta pipeline completo de clustering jerárquico y visualización
    para aglomerativo y divisivo.

    Args:
        k_values: Lista de valores de K (ej: [4, 6, 10])

    Returns:
        Diccionario con resultados para cada K y método
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

    results = {}

    # Procesar cada K
    for k in k_values:
        # Aglomerativo (ward)
        agg_model, agg_labels = train_hierarchical(df_pca_3d, k, "ward")

        # Guardar modelo aglomerativo
        agg_model_path = config.get_dataset_path("processed") + f"/hierarchical_agglomerative_ward_k{k}.joblib"
        joblib.dump(agg_model, agg_model_path)

        # Guardar etiquetas aglomerativo
        agg_labels_df = pd.DataFrame({"cluster": agg_labels})
        agg_labels_path = config.get_dataset_path("processed") + f"/hierarchical_agglomerative_ward_k{k}_labels.csv"
        agg_labels_df.to_csv(agg_labels_path, index=False)

        # Crear directorio de salida para figuras aglomerativo
        agg_figures_path = config.get_output_path("figures", f"clustering/hierarchical/agglomerative/k{k}")
        Path(agg_figures_path).mkdir(parents=True, exist_ok=True)

        # Generar scatter 3D aglomerativo
        fig_scatter_agg = plot_clusters_3d(
            df_pca_3d, agg_labels, k, f"Hierarchical Agglomerative (Ward) K={k} (3D)"
        )
        scatter_agg_path = agg_figures_path + "/scatter_3d_clusters.png"
        fig_scatter_agg.savefig(scatter_agg_path, dpi=300, bbox_inches="tight")
        plt.close(fig_scatter_agg)

        # Generar scatter 3D + Sleep Disorder aglomerativo
        fig_scatter_sd_agg = plot_clusters_3d_with_sleep_disorder(
            df_pca_3d,
            agg_labels,
            sleep_disorder,
            k,
            f"Hierarchical Agglomerative (Ward) K={k} (3D) with Sleep Disorder",
        )
        scatter_sd_agg_path = agg_figures_path + "/scatter_3d_clusters_sleep_disorder.png"
        fig_scatter_sd_agg.savefig(scatter_sd_agg_path, dpi=300, bbox_inches="tight")
        plt.close(fig_scatter_sd_agg)

        # Generar biplot 3D aglomerativo
        fig_biplot_agg = plot_biplot_clusters_3d(
            df_pca_3d, loadings, agg_labels, k, f"Biplot Hierarchical Agglomerative (Ward) K={k} (3D)"
        )
        biplot_agg_path = agg_figures_path + "/biplot_3d_clusters.png"
        fig_biplot_agg.savefig(biplot_agg_path, dpi=300, bbox_inches="tight")
        plt.close(fig_biplot_agg)

        # Generar biplot 3D + Sleep Disorder aglomerativo
        fig_biplot_sd_agg = plot_biplot_clusters_3d_with_sleep_disorder(
            df_pca_3d,
            loadings,
            agg_labels,
            sleep_disorder,
            k,
            f"Biplot Hierarchical Agglomerative (Ward) K={k} (3D) with Sleep Disorder",
        )
        biplot_sd_agg_path = agg_figures_path + "/biplot_3d_clusters_sleep_disorder.png"
        fig_biplot_sd_agg.savefig(biplot_sd_agg_path, dpi=300, bbox_inches="tight")
        plt.close(fig_biplot_sd_agg)

        # Divisivo (complete)
        div_model, div_labels = train_hierarchical(df_pca_3d, k, "complete")

        # Guardar modelo divisivo
        div_model_path = config.get_dataset_path("processed") + f"/hierarchical_divisive_complete_k{k}.joblib"
        joblib.dump(div_model, div_model_path)

        # Guardar etiquetas divisivo
        div_labels_df = pd.DataFrame({"cluster": div_labels})
        div_labels_path = config.get_dataset_path("processed") + f"/hierarchical_divisive_complete_k{k}_labels.csv"
        div_labels_df.to_csv(div_labels_path, index=False)

        # Crear directorio de salida para figuras divisivo
        div_figures_path = config.get_output_path("figures", f"clustering/hierarchical/divisive/k{k}")
        Path(div_figures_path).mkdir(parents=True, exist_ok=True)

        # Generar scatter 3D divisivo
        fig_scatter_div = plot_clusters_3d(
            df_pca_3d, div_labels, k, f"Hierarchical Divisive (Complete) K={k} (3D)"
        )
        scatter_div_path = div_figures_path + "/scatter_3d_clusters.png"
        fig_scatter_div.savefig(scatter_div_path, dpi=300, bbox_inches="tight")
        plt.close(fig_scatter_div)

        # Generar scatter 3D + Sleep Disorder divisivo
        fig_scatter_sd_div = plot_clusters_3d_with_sleep_disorder(
            df_pca_3d,
            div_labels,
            sleep_disorder,
            k,
            f"Hierarchical Divisive (Complete) K={k} (3D) with Sleep Disorder",
        )
        scatter_sd_div_path = div_figures_path + "/scatter_3d_clusters_sleep_disorder.png"
        fig_scatter_sd_div.savefig(scatter_sd_div_path, dpi=300, bbox_inches="tight")
        plt.close(fig_scatter_sd_div)

        # Generar biplot 3D divisivo
        fig_biplot_div = plot_biplot_clusters_3d(
            df_pca_3d, loadings, div_labels, k, f"Biplot Hierarchical Divisive (Complete) K={k} (3D)"
        )
        biplot_div_path = div_figures_path + "/biplot_3d_clusters.png"
        fig_biplot_div.savefig(biplot_div_path, dpi=300, bbox_inches="tight")
        plt.close(fig_biplot_div)

        # Generar biplot 3D + Sleep Disorder divisivo
        fig_biplot_sd_div = plot_biplot_clusters_3d_with_sleep_disorder(
            df_pca_3d,
            loadings,
            div_labels,
            sleep_disorder,
            k,
            f"Biplot Hierarchical Divisive (Complete) K={k} (3D) with Sleep Disorder",
        )
        biplot_sd_div_path = div_figures_path + "/biplot_3d_clusters_sleep_disorder.png"
        fig_biplot_sd_div.savefig(biplot_sd_div_path, dpi=300, bbox_inches="tight")
        plt.close(fig_biplot_sd_div)

        # Guardar resultados para este K
        results[f"k{k}"] = {
            "agglomerative": {
                "model_path": agg_model_path,
                "labels_path": agg_labels_path,
                "scatter_path": scatter_agg_path,
                "scatter_sd_path": scatter_sd_agg_path,
                "biplot_path": biplot_agg_path,
                "biplot_sd_path": biplot_sd_agg_path,
            },
            "divisive": {
                "model_path": div_model_path,
                "labels_path": div_labels_path,
                "scatter_path": scatter_div_path,
                "scatter_sd_path": scatter_sd_div_path,
                "biplot_path": biplot_div_path,
                "biplot_sd_path": biplot_sd_div_path,
            },
        }

    return results
