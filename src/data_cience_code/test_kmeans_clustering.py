import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.clustering import run_kmeans_clustering

if __name__ == "__main__":
    print("Ejecutando clustering k-means con K=4, K=6, K=10 y K=14...")
    results = run_kmeans_clustering([4, 6, 10, 14])

    print("\n=== Resultados del Clustering ===")
    for k, k_results in results.items():
        print(f"\n--- K={k} ---")
        print(f"Modelo: {k_results['model_path']}")
        print(f"Etiquetas: {k_results['labels_path']}")
        print(f"Scatter 3D: {k_results['scatter_path']}")
        print(f"Scatter 3D + Sleep Disorder: {k_results['scatter_sd_path']}")
        print(f"Biplot 3D: {k_results['biplot_path']}")
        print(f"Biplot 3D + Sleep Disorder: {k_results['biplot_sd_path']}")

    print("\n✅ Clustering y visualizaciones completados exitosamente")
