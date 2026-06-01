import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import davies_bouldin_score, silhouette_score


def calculate_dunn_index(df_pca: pd.DataFrame, labels: np.ndarray) -> float:
    """
    Calcula Dunn index para clustering.

    Dunn = min(distancia inter-cluster) / max(diámetro intra-cluster)

    Args:
        df_pca: DataFrame con componentes PCA
        labels: Etiquetas de clustering

    Returns:
        Dunn index
    """
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]  # Excluir noise

    if len(unique_labels) < 2:
        return 0.0

    # Calcular distancias inter-cluster (mínima)
    inter_cluster_distances = []
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            cluster_i = df_pca[labels == unique_labels[i]].values
            cluster_j = df_pca[labels == unique_labels[j]].values
            dist = cdist(cluster_i, cluster_j).min()
            inter_cluster_distances.append(dist)

    min_inter_distance = min(inter_cluster_distances) if inter_cluster_distances else 0.0

    # Calcular diámetros intra-cluster (máximo)
    intra_cluster_diameters = []
    for label in unique_labels:
        cluster = df_pca[labels == label].values
        if len(cluster) > 1:
            dist_matrix = cdist(cluster, cluster)
            diameter = dist_matrix.max()
            intra_cluster_diameters.append(diameter)

    max_intra_diameter = max(intra_cluster_diameters) if intra_cluster_diameters else 1.0

    # Dunn index
    if max_intra_diameter == 0:
        return 0.0

    dunn = min_inter_distance / max_intra_diameter
    return dunn


def calculate_clustering_metrics(df_pca: pd.DataFrame, labels: np.ndarray) -> dict:
    """
    Calcula silhouette, Dunn y Davies-Bouldin para clustering.

    Args:
        df_pca: DataFrame con componentes PCA
        labels: Etiquetas de clustering (-1 para noise en DBSCAN)

    Returns:
        Dict con silhouette_score, dunn_index, davies_bouldin_index
    """
    unique_labels = np.unique(labels)
    unique_labels_no_noise = unique_labels[unique_labels != -1]

    # Excluir noise del cálculo
    mask = labels != -1
    df_pca_no_noise = df_pca[mask]
    labels_no_noise = labels[mask]

    metrics = {
        "silhouette_score": None,
        "dunn_index": None,
        "davies_bouldin_index": None,
    }

    # Silhouette score
    if len(unique_labels_no_noise) > 1 and mask.sum() > 1:
        try:
            metrics["silhouette_score"] = silhouette_score(df_pca_no_noise, labels_no_noise)
        except Exception:
            metrics["silhouette_score"] = None

    # Dunn index
    if len(unique_labels_no_noise) >= 2:
        try:
            metrics["dunn_index"] = calculate_dunn_index(df_pca_no_noise, labels_no_noise)
        except Exception:
            metrics["dunn_index"] = None

    # Davies-Bouldin index
    if len(unique_labels_no_noise) > 1 and mask.sum() > 1:
        try:
            metrics["davies_bouldin_index"] = davies_bouldin_score(df_pca_no_noise, labels_no_noise)
        except Exception:
            metrics["davies_bouldin_index"] = None

    return metrics
