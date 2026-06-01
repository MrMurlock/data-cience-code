import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.clustering import find_optimal_eps, run_dbscan_clustering

if __name__ == "__main__":
    print("Ejecutando clustering DBSCAN...")

    # Cargar componentes PCA para encontrar eps óptimo
    import pandas as pd
    import config.config as config

    pca_components_path = config.get_dataset_path("processed") + "/pca_components.csv"
    df_pca = pd.read_csv(pca_components_path)
    df_pca_3d = df_pca[["PC1", "PC2", "PC3"]]

    # Encontrar eps óptimo con min_samples=5
    min_samples = 5
    eps_optimal = find_optimal_eps(df_pca_3d, min_samples)
    print(f"\nEps óptimo encontrado: {eps_optimal:.4f} (min_samples={min_samples})")

    # Ejecutar DBSCAN con eps óptimo
    print(f"\nEjecutando DBSCAN con eps={eps_optimal:.4f}, min_samples={min_samples}...")
    results = run_dbscan_clustering(eps_optimal, min_samples, generate_k_distance=True)

    print("\n=== Resultados del Clustering DBSCAN ===")
    print(f"Eps: {results['eps']:.4f}")
    print(f"Min Samples: {results['min_samples']}")
    print(f"Número de Clusters: {results['n_clusters']}")
    print(f"Puntos de Ruido: {results['n_noise']}")
    print(f"Silhouette Score: {results['silhouette_score']:.4f}" if results['silhouette_score'] else "Silhouette Score: N/A")

    if "k_distance_plot_path" in results:
        print(f"\nGráfico K-Distance: {results['k_distance_plot_path']}")

    if results['n_clusters'] > 0:
        print(f"\nScatter 3D: {results['scatter_path']}")
        print(f"Scatter 3D + Sleep Disorder: {results['scatter_sd_path']}")
        print(f"Biplot 3D: {results['biplot_path']}")
        print(f"Biplot 3D + Sleep Disorder: {results['biplot_sd_path']}")

    print(f"\nMétricas: {results['metrics_path']}")

    print("\n✅ Clustering DBSCAN completado exitosamente")
