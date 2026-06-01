from pathlib import Path

import joblib
import pandas as pd

import config.config as config
from .dbscan import train_dbscan
from .utils import calculate_clustering_metrics


def generate_comparison_report(
    k_values: list[int] = [4, 6, 10],
    dbscan_eps: float = 0.7117,
    dbscan_min_samples: int = 5,
) -> pd.DataFrame:
    """
    Genera tabla comparativa de métricas para todos los métodos.

    Args:
        k_values: Lista de K para k-means y jerárquico
        dbscan_eps: eps para DBSCAN
        dbscan_min_samples: min_samples para DBSCAN

    Returns:
        DataFrame con métricas comparativas
    """
    # Cargar componentes PCA
    pca_components_path = config.get_dataset_path("processed") + "/pca_components.csv"
    df_pca = pd.read_csv(pca_components_path)
    df_pca_3d = df_pca[["PC1", "PC2", "PC3"]]

    results = []

    # K-means
    for k in k_values:
        model_path = config.get_dataset_path("processed") + f"/kmeans_k{k}.joblib"
        labels_path = config.get_dataset_path("processed") + f"/kmeans_k{k}_labels.csv"

        model = joblib.load(model_path)
        labels_df = pd.read_csv(labels_path)
        labels = labels_df["cluster"].values

        metrics = calculate_clustering_metrics(df_pca_3d, labels)

        results.append(
            {
                "algorithm": "k-means",
                "k": k,
                "eps": None,
                "min_samples": None,
                "silhouette_score": metrics["silhouette_score"],
                "dunn_index": metrics["dunn_index"],
                "davies_bouldin_index": metrics["davies_bouldin_index"],
                "n_clusters": k,
                "n_noise": 0,
            }
        )

    # Jerárquico aglomerativo (Ward)
    for k in k_values:
        model_path = config.get_dataset_path("processed") + f"/hierarchical_agglomerative_ward_k{k}.joblib"
        labels_path = config.get_dataset_path("processed") + f"/hierarchical_agglomerative_ward_k{k}_labels.csv"

        model = joblib.load(model_path)
        labels_df = pd.read_csv(labels_path)
        labels = labels_df["cluster"].values

        metrics = calculate_clustering_metrics(df_pca_3d, labels)

        results.append(
            {
                "algorithm": "hierarchical_agglomerative",
                "k": k,
                "eps": None,
                "min_samples": None,
                "silhouette_score": metrics["silhouette_score"],
                "dunn_index": metrics["dunn_index"],
                "davies_bouldin_index": metrics["davies_bouldin_index"],
                "n_clusters": k,
                "n_noise": 0,
            }
        )

    # Jerárquico divisivo (Complete)
    for k in k_values:
        model_path = config.get_dataset_path("processed") + f"/hierarchical_divisive_complete_k{k}.joblib"
        labels_path = config.get_dataset_path("processed") + f"/hierarchical_divisive_complete_k{k}_labels.csv"

        model = joblib.load(model_path)
        labels_df = pd.read_csv(labels_path)
        labels = labels_df["cluster"].values

        metrics = calculate_clustering_metrics(df_pca_3d, labels)

        results.append(
            {
                "algorithm": "hierarchical_divisive",
                "k": k,
                "eps": None,
                "min_samples": None,
                "silhouette_score": metrics["silhouette_score"],
                "dunn_index": metrics["dunn_index"],
                "davies_bouldin_index": metrics["davies_bouldin_index"],
                "n_clusters": k,
                "n_noise": 0,
            }
        )

    # DBSCAN
    dbscan_model, dbscan_labels = train_dbscan(df_pca_3d, dbscan_eps, dbscan_min_samples)
    dbscan_metrics = calculate_clustering_metrics(df_pca_3d, dbscan_labels)

    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)

    results.append(
        {
            "algorithm": "dbscan",
            "k": n_clusters,
            "eps": dbscan_eps,
            "min_samples": dbscan_min_samples,
            "silhouette_score": dbscan_metrics["silhouette_score"],
            "dunn_index": dbscan_metrics["dunn_index"],
            "davies_bouldin_index": dbscan_metrics["davies_bouldin_index"],
            "n_clusters": n_clusters,
            "n_noise": n_noise,
        }
    )

    # Crear DataFrame
    df_comparison = pd.DataFrame(results)

    # Guardar en CSV
    comparison_path = config.get_output_path("reports", "clustering/comparison")
    Path(comparison_path).mkdir(parents=True, exist_ok=True)
    csv_path = comparison_path + "/comparison_metrics.csv"
    df_comparison.to_csv(csv_path, index=False)

    return df_comparison
