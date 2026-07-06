from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import data_cience_code.config.config as config


def load_dbscan_labels_and_data():
    """
    Carga los labels de DBSCAN y el dataset normalizado con variables numéricas.

    Returns:
        Tuple con (labels, df_normalized)
    """
    # Cargar labels de DBSCAN
    labels_path = config.get_dataset_path("processed") + "/dbscan_labels.csv"
    labels_df = pd.read_csv(labels_path)
    labels = labels_df["cluster"].values

    # Cargar dataset normalizado
    normalized_path = config.get_dataset_path("processed") + "/normalized.csv"
    df_normalized = pd.read_csv(normalized_path)

    return labels, df_normalized


def calculate_cluster_descriptive_stats(labels, df_normalized):
    """
    Calcula estadísticas descriptivas por cluster para las variables numéricas.

    Args:
        labels: Array con labels de clusters
        df_normalized: DataFrame con datos normalizados

    Returns:
        DataFrame con estadísticas descriptivas por cluster
    """
    # Variables numéricas a analizar
    numerical_vars = [
        "Age",
        "Sleep Duration",
        "Quality of Sleep",
        "Physical Activity Level",
        "Stress Level",
        "Heart Rate",
        "Daily Steps",
        "bp_systolic",
        "bp_diastolic",
    ]

    # Crear DataFrame con labels
    df_with_labels = df_normalized.copy()
    df_with_labels["cluster"] = labels

    # Filtrar solo variables numéricas y cluster
    df_analysis = df_with_labels[numerical_vars + ["cluster"]]

    # Calcular estadísticas por cluster
    stats_list = []

    for cluster_id in sorted(df_analysis["cluster"].unique()):
        cluster_name = f"Cluster_{cluster_id}" if cluster_id != -1 else "Noise"
        cluster_data = df_analysis[df_analysis["cluster"] == cluster_id][numerical_vars]

        stats = {
            "cluster": cluster_name,
            "n_observations": len(cluster_data),
        }

        for var in numerical_vars:
            stats[f"{var}_mean"] = cluster_data[var].mean()
            stats[f"{var}_median"] = cluster_data[var].median()
            stats[f"{var}_std"] = cluster_data[var].std()
            stats[f"{var}_min"] = cluster_data[var].min()
            stats[f"{var}_max"] = cluster_data[var].max()
            stats[f"{var}_q25"] = cluster_data[var].quantile(0.25)
            stats[f"{var}_q75"] = cluster_data[var].quantile(0.75)

        stats_list.append(stats)

    stats_df = pd.DataFrame(stats_list)
    return stats_df


def plot_boxplots_by_cluster(labels, df_normalized, output_path):
    """
    Genera boxplots para cada variable numérica por cluster.

    Args:
        labels: Array con labels de clusters
        df_normalized: DataFrame con datos normalizados
        output_path: Ruta donde guardar los gráficos
    """
    # Variables numéricas a analizar
    numerical_vars = [
        "Age",
        "Sleep Duration",
        "Quality of Sleep",
        "Physical Activity Level",
        "Stress Level",
        "Heart Rate",
        "Daily Steps",
        "bp_systolic",
        "bp_diastolic",
    ]

    # Crear DataFrame con labels
    df_with_labels = df_normalized.copy()
    df_with_labels["cluster"] = labels

    # Renombrar cluster -1 a "Noise"
    df_with_labels["cluster"] = df_with_labels["cluster"].apply(
        lambda x: f"Cluster_{x}" if x != -1 else "Noise"
    )

    # Configurar estilo
    plt.style.use("seaborn-v0_8-whitegrid")

    # Crear directorio de salida
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Generar boxplot para cada variable
    for var in numerical_vars:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Ordenar clusters por tamaño
        cluster_order = (
            df_with_labels["cluster"]
            .value_counts()
            .sort_values(ascending=False)
            .index.tolist()
        )

        sns.boxplot(data=df_with_labels, x="cluster", y=var, order=cluster_order, ax=ax)
        ax.set_xlabel("Cluster", fontsize=12)
        ax.set_ylabel(var, fontsize=12)
        ax.set_title(f"Distribución de {var} por Cluster", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Guardar gráfico
        plot_path = f"{output_path}/boxplot_{var.replace(' ', '_').replace('/', '_')}.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_violin_plots_by_cluster(labels, df_normalized, output_path):
    """
    Genera violin plots para cada variable numérica por cluster.

    Args:
        labels: Array con labels de clusters
        df_normalized: DataFrame con datos normalizados
        output_path: Ruta donde guardar los gráficos
    """
    # Variables numéricas a analizar
    numerical_vars = [
        "Age",
        "Sleep Duration",
        "Quality of Sleep",
        "Physical Activity Level",
        "Stress Level",
        "Heart Rate",
        "Daily Steps",
        "bp_systolic",
        "bp_diastolic",
    ]

    # Crear DataFrame con labels
    df_with_labels = df_normalized.copy()
    df_with_labels["cluster"] = labels

    # Renombrar cluster -1 a "Noise"
    df_with_labels["cluster"] = df_with_labels["cluster"].apply(
        lambda x: f"Cluster_{x}" if x != -1 else "Noise"
    )

    # Configurar estilo
    plt.style.use("seaborn-v0_8-whitegrid")

    # Crear directorio de salida
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Generar violin plot para cada variable
    for var in numerical_vars:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Ordenar clusters por tamaño
        cluster_order = (
            df_with_labels["cluster"]
            .value_counts()
            .sort_values(ascending=False)
            .index.tolist()
        )

        sns.violinplot(
            data=df_with_labels, x="cluster", y=var, order=cluster_order, ax=ax
        )
        ax.set_xlabel("Cluster", fontsize=12)
        ax.set_ylabel(var, fontsize=12)
        ax.set_title(f"Distribución de {var} por Cluster (Violin Plot)", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Guardar gráfico
        plot_path = f"{output_path}/violin_{var.replace(' ', '_').replace('/', '_')}.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_cluster_means_heatmap(labels, df_normalized, output_path):
    """
    Genera heatmap de medias por cluster.

    Args:
        labels: Array con labels de clusters
        df_normalized: DataFrame con datos normalizados
        output_path: Ruta donde guardar el gráfico
    """
    # Variables numéricas a analizar
    numerical_vars = [
        "Age",
        "Sleep Duration",
        "Quality of Sleep",
        "Physical Activity Level",
        "Stress Level",
        "Heart Rate",
        "Daily Steps",
        "bp_systolic",
        "bp_diastolic",
    ]

    # Crear DataFrame con labels
    df_with_labels = df_normalized.copy()
    df_with_labels["cluster"] = labels

    # Renombrar cluster -1 a "Noise"
    df_with_labels["cluster"] = df_with_labels["cluster"].apply(
        lambda x: f"Cluster_{x}" if x != -1 else "Noise"
    )

    # Calcular medias por cluster
    cluster_means = (
        df_with_labels.groupby("cluster")[numerical_vars].mean().T
    )

    # Configurar estilo
    plt.style.use("seaborn-v0_8-whitegrid")

    # Crear directorio de salida
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Generar heatmap
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(
        cluster_means,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={"label": "Media (Normalizada)"},
        ax=ax,
    )
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Variable", fontsize=12)
    ax.set_title("Medias de Variables Numéricas por Cluster", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Guardar gráfico
    plot_path = f"{output_path}/heatmap_cluster_means.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_radar_chart(labels, df_normalized, output_path):
    """
    Genera radar chart de características por cluster.

    Args:
        labels: Array con labels de clusters
        df_normalized: DataFrame con datos normalizados
        output_path: Ruta donde guardar el gráfico
    """
    # Variables numéricas a analizar
    numerical_vars = [
        "Age",
        "Sleep Duration",
        "Quality of Sleep",
        "Physical Activity Level",
        "Stress Level",
        "Heart Rate",
        "Daily Steps",
        "bp_systolic",
        "bp_diastolic",
    ]

    # Crear DataFrame con labels
    df_with_labels = df_normalized.copy()
    df_with_labels["cluster"] = labels

    # Renombrar cluster -1 a "Noise"
    df_with_labels["cluster"] = df_with_labels["cluster"].apply(
        lambda x: f"Cluster_{x}" if x != -1 else "Noise"
    )

    # Calcular medias por cluster
    cluster_means = df_with_labels.groupby("cluster")[numerical_vars].mean()

    # Filtrar clusters con suficientes observaciones (excluir Noise si es muy pequeño)
    cluster_sizes = df_with_labels["cluster"].value_counts()
    clusters_to_plot = cluster_sizes[cluster_sizes >= 5].index.tolist()

    # Configurar estilo
    plt.style.use("seaborn-v0_8-whitegrid")

    # Crear directorio de salida
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Número de variables
    n_vars = len(numerical_vars)
    angles = [n / float(n_vars) * 2 * np.pi for n in range(n_vars)]
    angles += angles[:1]  # Completar el círculo

    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection="polar"))

    # Colores para clusters
    colors = plt.cm.tab20(np.linspace(0, 1, len(clusters_to_plot)))

    # Plotear cada cluster
    for i, cluster_id in enumerate(clusters_to_plot):
        values = cluster_means.loc[cluster_id].values.tolist()
        values += values[:1]  # Completar el círculo

        ax.plot(angles, values, "o-", linewidth=2, label=cluster_id, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    # Configurar etiquetas
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(numerical_vars, fontsize=10)
    ax.set_ylim(-2, 2)  # Rango para datos normalizados
    ax.set_title("Radar Chart de Características por Cluster", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

    plt.tight_layout()

    # Guardar gráfico
    plot_path = f"{output_path}/radar_chart_clusters.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_cluster_descriptive_analysis():
    """
    Ejecuta el análisis descriptivo completo de clusters DBSCAN.

    Returns:
        Diccionario con rutas de archivos generados
    """
    print("Ejecutando análisis descriptivo de clusters DBSCAN...")

    # Cargar datos
    labels, df_normalized = load_dbscan_labels_and_data()
    print(f"✓ Cargados {len(labels)} observaciones y {len(df_normalized.columns)} variables")

    # Calcular estadísticas descriptivas
    stats_df = calculate_cluster_descriptive_stats(labels, df_normalized)
    print(f"✓ Calculadas estadísticas para {len(stats_df)} clusters")

    # Guardar estadísticas
    reports_path = config.get_output_path("reports", "clustering/dbscan/descriptive")
    Path(reports_path).mkdir(parents=True, exist_ok=True)
    stats_path = reports_path + "/cluster_descriptive_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"✓ Estadísticas guardadas en {stats_path}")

    # Generar gráficos
    figures_path = config.get_output_path("figures", "clustering/dbscan/descriptive")

    print("Generando boxplots...")
    plot_boxplots_by_cluster(labels, df_normalized, figures_path)
    print(f"✓ Boxplots guardados en {figures_path}")

    print("Generando violin plots...")
    plot_violin_plots_by_cluster(labels, df_normalized, figures_path)
    print(f"✓ Violin plots guardados en {figures_path}")

    print("Generando heatmap de medias...")
    plot_cluster_means_heatmap(labels, df_normalized, figures_path)
    print(f"✓ Heatmap guardado en {figures_path}")

    print("Generando radar chart...")
    plot_cluster_radar_chart(labels, df_normalized, figures_path)
    print(f"✓ Radar chart guardado en {figures_path}")

    results = {
        "stats_path": stats_path,
        "boxplots_path": figures_path,
        "violin_plots_path": figures_path,
        "heatmap_path": figures_path + "/heatmap_cluster_means.png",
        "radar_chart_path": figures_path + "/radar_chart_clusters.png",
    }

    print("\n✅ Análisis descriptivo completado exitosamente")
    return results
