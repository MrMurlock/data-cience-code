from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

import data_cience_code.config.config as config
from .kmeans import (
    plot_biplot_clusters_3d,
    plot_clusters_3d,
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


