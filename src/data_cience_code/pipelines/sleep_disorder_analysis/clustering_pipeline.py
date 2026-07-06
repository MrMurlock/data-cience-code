"""
Clustering pipeline for Sleep Disorder dataset.

This module integrates all clustering analysis steps specific to the Sleep Disorder dataset,
using generic functions from models/ and dataset-specific visualizations.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import data_cience_code.config.config as config
from data_cience_code.models.clustering import (
    find_optimal_eps,
    plot_k_distance,
    run_hierarchical_analysis,
    train_dbscan,
    train_hierarchical,
    train_kmeans,
)
from data_cience_code.models.clustering.cluster_analysis import run_cluster_descriptive_analysis
from data_cience_code.models.clustering.comparison import generate_comparison_report
from data_cience_code.models.clustering.kmeans import (
    plot_biplot_clusters_3d,
    plot_clusters_3d,
    run_kmeans_discovery,
)
from data_cience_code.models.dimensionality_reduction import run_pca_visualizations
from .visualization import (
    plot_biplot_clusters_3d_with_sleep_disorder,
    plot_clusters_3d_with_sleep_disorder,
)


def run_kmeans_clustering_pipeline(k_values: list[int]) -> dict:
    """
    Ejecuta pipeline completo de clustering k-means y visualización.

    Args:
        k_values: Lista de valores de K para entrenar (ej: [4, 6])

    Returns:
        Diccionario con resultados para cada K
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

    for k in k_values:
        # Entrenar k-means
        kmeans_model, labels = train_kmeans(df_pca_3d, k)

        # Guardar modelo
        model_path = config.get_dataset_path("processed") + f"/kmeans_k{k}.joblib"
        joblib.dump(kmeans_model, model_path)

        # Guardar etiquetas
        labels_df = pd.DataFrame({"cluster": labels})
        labels_path = config.get_dataset_path("processed") + f"/kmeans_k{k}_labels.csv"
        labels_df.to_csv(labels_path, index=False)

        # Crear subdirectorio de salida
        figures_path = config.get_output_path("figures", f"clustering/kmeans/k{k}")
        Path(figures_path).mkdir(parents=True, exist_ok=True)

        # Generar scatter 3D por cluster
        fig_scatter = plot_clusters_3d(
            df_pca_3d, labels, k, f"K-Means Clustering K={k} (3D)"
        )
        scatter_path = figures_path + "/scatter_3d_clusters.png"
        fig_scatter.savefig(scatter_path, dpi=300, bbox_inches="tight")
        plt.close(fig_scatter)

        # Generar scatter 3D cluster + Sleep Disorder
        fig_scatter_sd = plot_clusters_3d_with_sleep_disorder(
            df_pca_3d,
            labels,
            sleep_disorder,
            k,
            f"K-Means Clustering K={k} (3D) with Sleep Disorder",
        )
        scatter_sd_path = figures_path + "/scatter_3d_clusters_sleep_disorder.png"
        fig_scatter_sd.savefig(scatter_sd_path, dpi=300, bbox_inches="tight")
        plt.close(fig_scatter_sd)

        # Generar biplot 3D por cluster
        fig_biplot = plot_biplot_clusters_3d(
            df_pca_3d, loadings, labels, k, f"Biplot K-Means Clustering K={k} (3D)"
        )
        biplot_path = figures_path + "/biplot_3d_clusters.png"
        fig_biplot.savefig(biplot_path, dpi=300, bbox_inches="tight")
        plt.close(fig_biplot)

        # Generar biplot 3D cluster + Sleep Disorder
        fig_biplot_sd = plot_biplot_clusters_3d_with_sleep_disorder(
            df_pca_3d,
            loadings,
            labels,
            sleep_disorder,
            k,
            f"Biplot K-Means Clustering K={k} (3D) with Sleep Disorder",
        )
        biplot_sd_path = figures_path + "/biplot_3d_clusters_sleep_disorder.png"
        fig_biplot_sd.savefig(biplot_sd_path, dpi=300, bbox_inches="tight")
        plt.close(fig_biplot_sd)

        # Guardar resultados para este K
        results[k] = {
            "model_path": model_path,
            "labels_path": labels_path,
            "scatter_path": scatter_path,
            "scatter_sd_path": scatter_sd_path,
            "biplot_path": biplot_path,
            "biplot_sd_path": biplot_sd_path,
        }

    return results


def run_hierarchical_clustering_pipeline(k_values: list[int]) -> dict:
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


def run_dbscan_clustering_pipeline(eps: float, min_samples: int, generate_k_distance: bool = True) -> dict:
    """
    Ejecuta pipeline completo de DBSCAN clustering y visualización.

    Args:
        eps: Radio del vecindario
        min_samples: Mínimo de muestras en vecindario
        generate_k_distance: Si True, genera gráfico k-distance

    Returns:
        Diccionario con resultados y métricas
    """
    from sklearn.metrics import silhouette_score

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


def run_full_clustering_analysis(k_values: list[int] = [4, 6, 10, 14]) -> dict:
    """
    Ejecuta análisis completo de clustering para el dataset Sleep Disorder.

    Incluye:
    1. Descubrimiento de K óptimo para K-means
    2. Clustering K-means
    3. Análisis jerárquico (dendrogramas + silhouette)
    4. Clustering jerárquico
    5. DBSCAN clustering
    6. Análisis descriptivo de clusters
    7. Comparación de métodos
    8. Visualizaciones PCA

    Args:
        k_values: Lista de valores de K para clustering

    Returns:
        Diccionario con todos los resultados
    """
    print("=" * 60)
    print("=== FULL CLUSTERING ANALYSIS FOR SLEEP DISORDER DATASET ===")
    print("=" * 60)

    results = {}

    # 1. K-means discovery
    print("\n>>> Step 1: K-means K discovery...")
    kmeans_discovery = run_kmeans_discovery()
    results["kmeans_discovery"] = kmeans_discovery
    print(f"✓ K-means discovery completed")

    # 2. K-means clustering
    print("\n>>> Step 2: K-means clustering...")
    kmeans_results = run_kmeans_clustering_pipeline(k_values)
    results["kmeans_clustering"] = kmeans_results
    print(f"✓ K-means clustering completed for K={k_values}")

    # 3. Hierarchical analysis (dendrograms)
    print("\n>>> Step 3: Hierarchical analysis (dendrograms)...")
    hierarchical_analysis = run_hierarchical_analysis((2, 10))
    results["hierarchical_analysis"] = hierarchical_analysis
    print(f"✓ Hierarchical analysis completed")

    # 4. Hierarchical clustering
    print("\n>>> Step 4: Hierarchical clustering...")
    hierarchical_results = run_hierarchical_clustering_pipeline(k_values)
    results["hierarchical_clustering"] = hierarchical_results
    print(f"✓ Hierarchical clustering completed for K={k_values}")

    # 5. DBSCAN clustering
    print("\n>>> Step 5: DBSCAN clustering...")
    # Encontrar eps óptimo
    pca_components_path = config.get_dataset_path("processed") + "/pca_components.csv"
    df_pca = pd.read_csv(pca_components_path)
    df_pca_3d = df_pca[["PC1", "PC2", "PC3"]]
    eps_optimal = find_optimal_eps(df_pca_3d, min_samples=5)
    print(f"  Optimal eps found: {eps_optimal:.4f}")
    
    dbscan_results = run_dbscan_clustering_pipeline(eps_optimal, min_samples=5)
    results["dbscan_clustering"] = dbscan_results
    print(f"✓ DBSCAN clustering completed")

    # 6. Descriptive analysis
    print("\n>>> Step 6: Cluster descriptive analysis...")
    descriptive_results = run_cluster_descriptive_analysis()
    results["descriptive_analysis"] = descriptive_results
    print(f"✓ Descriptive analysis completed")

    # 7. Comparison
    print("\n>>> Step 7: Clustering comparison...")
    comparison_results = generate_comparison_report()
    results["comparison"] = comparison_results
    print(f"✓ Comparison completed")

    # 8. PCA visualizations
    print("\n>>> Step 8: PCA visualizations...")
    pca_viz_results = run_pca_visualizations()
    results["pca_visualizations"] = pca_viz_results
    print(f"✓ PCA visualizations completed")

    print("\n" + "=" * 60)
    print("=== FULL CLUSTERING ANALYSIS COMPLETED ===")
    print("=" * 60)

    return results


if __name__ == "__main__":
    # Ejecutar análisis completo
    results = run_full_clustering_analysis([4, 6, 10, 14])

    print("\n=== SUMMARY ===")
    print("All clustering analyses completed successfully.")
    print("Check output/figures/ and output/reports/ for generated files.")
