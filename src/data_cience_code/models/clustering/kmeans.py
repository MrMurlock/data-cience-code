from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

import data_cience_code.config.config as config


def find_optimal_k(
    df_pca: pd.DataFrame, k_range: tuple[int, int] = (2, 10)
) -> dict:
    """
    Ejecuta k-means para cada K en el rango especificado y calcula métricas.

    Args:
        df_pca: DataFrame con componentes PCA (debe tener PC1, PC2, PC3)
        k_range: Tupla (k_min, k_max) para explorar

    Returns:
        Diccionario con:
        - k_values: Lista de valores de K
        - inertia_values: Lista de inertia para cada K
        - silhouette_scores: Lista de silhouette scores para cada K
    """
    k_min, k_max = k_range
    k_values = list(range(k_min, k_max + 1))
    inertia_values = []
    silhouette_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(df_pca)
        inertia_values.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df_pca, kmeans.labels_))

    return {
        "k_values": k_values,
        "inertia_values": inertia_values,
        "silhouette_scores": silhouette_scores,
    }


def plot_elbow_method(
    k_values: list[int], inertia_values: list[float]
) -> plt.Figure:
    """
    Genera gráfico del método del codo.

    Args:
        k_values: Lista de valores de K
        inertia_values: Lista de inertia para cada K

    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(k_values, inertia_values, "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax.set_ylabel("Inertia", fontsize=12)
    ax.set_title("Elbow Method for Optimal K", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)

    plt.tight_layout()
    return fig


def plot_silhouette_scores(
    k_values: list[int], silhouette_scores: list[float]
) -> plt.Figure:
    """
    Genera gráfico de Silhouette Score vs K.

    Args:
        k_values: Lista de valores de K
        silhouette_scores: Lista de silhouette scores para cada K

    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(k_values, silhouette_scores, "go-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax.set_ylabel("Silhouette Score", fontsize=12)
    ax.set_title("Silhouette Score for Optimal K", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)

    # Marcar K con score máximo
    max_idx = np.argmax(silhouette_scores)
    max_k = k_values[max_idx]
    max_score = silhouette_scores[max_idx]
    ax.plot(
        max_k,
        max_score,
        "r*",
        markersize=15,
        label=f"Max: K={max_k} ({max_score:.3f})",
    )
    ax.legend()

    plt.tight_layout()
    return fig


def run_kmeans_discovery(k_range: tuple[int, int] = (2, 10)) -> dict:
    """
    Ejecuta el pipeline completo de descubrimiento del K óptimo para k-means.

    Args:
        k_range: Tupla (k_min, k_max) para explorar

    Returns:
        Diccionario con:
        - metrics: Diccionario con k_values, inertia_values, silhouette_scores
        - elbow_plot_path: Path del gráfico del método del codo
        - silhouette_plot_path: Path del gráfico de Silhouette Score
    """
    # Cargar componentes PCA
    pca_components_path = config.get_dataset_path("processed") + "/pca_components.csv"
    df_pca = pd.read_csv(pca_components_path)

    # Extraer solo PC1, PC2, PC3
    df_pca_3d = df_pca[["PC1", "PC2", "PC3"]]

    # Ejecutar análisis para encontrar K óptimo
    metrics = find_optimal_k(df_pca_3d, k_range)

    # Crear directorio de salida
    figures_path = config.get_output_path("figures", "clustering/kmeans")
    reports_path = config.get_output_path("reports", "clustering/kmeans/metrics")
    Path(figures_path).mkdir(parents=True, exist_ok=True)
    Path(reports_path).mkdir(parents=True, exist_ok=True)

    # Configurar estilo
    plt.style.use("seaborn-v0_8-whitegrid")

    # Generar plot del método del codo
    fig_elbow = plot_elbow_method(metrics["k_values"], metrics["inertia_values"])
    elbow_plot_path = figures_path + "/elbow_method.png"
    fig_elbow.savefig(elbow_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig_elbow)

    # Generar plot de Silhouette Score
    fig_silhouette = plot_silhouette_scores(
        metrics["k_values"], metrics["silhouette_scores"]
    )
    silhouette_plot_path = figures_path + "/silhouette_scores.png"
    fig_silhouette.savefig(silhouette_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig_silhouette)

    # Guardar métricas de inertia en CSV
    df_inertia = pd.DataFrame({
        "k": metrics["k_values"],
        "inertia": metrics["inertia_values"]
    })
    inertia_path = reports_path + "/inertia_values.csv"
    df_inertia.to_csv(inertia_path, index=False)

    # Guardar métricas de silhouette en CSV
    df_silhouette = pd.DataFrame({
        "k": metrics["k_values"],
        "silhouette_score": metrics["silhouette_scores"]
    })
    silhouette_metrics_path = reports_path + "/silhouette_scores.csv"
    df_silhouette.to_csv(silhouette_metrics_path, index=False)

    # Retornar resultados
    results = {
        "metrics": metrics,
        "elbow_plot_path": elbow_plot_path,
        "silhouette_plot_path": silhouette_plot_path,
        "inertia_metrics_path": inertia_path,
        "silhouette_metrics_path": silhouette_metrics_path,
    }

    return results


def train_kmeans(df_pca: pd.DataFrame, k: int) -> tuple[KMeans, np.ndarray]:
    """
    Entrena k-means con K específico.

    Args:
        df_pca: DataFrame con componentes PCA (debe tener PC1, PC2, PC3)
        k: Número de clusters

    Returns:
        Tuple con:
        - Modelo KMeans entrenado
        - Etiquetas de cluster asignadas
    """
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(df_pca)
    return kmeans, kmeans.labels_


def plot_clusters_3d(
    df_pca: pd.DataFrame, labels: np.ndarray, k: int, title: str
) -> plt.Figure:
    """
    Genera scatter plot 3D coloreado por cluster.

    Args:
        df_pca: DataFrame con componentes PCA (PC1, PC2, PC3)
        labels: Etiquetas de cluster asignadas
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

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    ax.legend(title="Clusters")
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    return fig




def plot_biplot_clusters_3d(
    df_pca: pd.DataFrame,
    loadings: pd.DataFrame,
    labels: np.ndarray,
    k: int,
    title: str,
) -> plt.Figure:
    """
    Genera biplot 3D con clusters.

    Args:
        df_pca: DataFrame con componentes PCA (PC1, PC2, PC3)
        loadings: DataFrame con loadings
        labels: Etiquetas de cluster asignadas
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
    ax.legend(title="Clusters")
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    return fig




