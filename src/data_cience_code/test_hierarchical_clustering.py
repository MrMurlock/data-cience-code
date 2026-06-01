import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.clustering import run_hierarchical_clustering

if __name__ == "__main__":
    print("Ejecutando clustering jerárquico con K=4, K=6, K=10 y K=14...")
    results = run_hierarchical_clustering([4, 6, 10, 14])

    print("\n=== Resultados del Clustering Jerárquico ===")

    for k_key in results:
        k = k_key.replace("k", "")
        print(f"\n--- K={k} ---")

        # Aglomerativo
        agg = results[k_key]["agglomerative"]
        print(f"Modelo Aglomerativo: {agg['model_path']}")
        print(f"Etiquetas Aglomerativo: {agg['labels_path']}")
        print(f"Scatter 3D: {agg['scatter_path']}")
        print(f"Scatter 3D + Sleep Disorder: {agg['scatter_sd_path']}")
        print(f"Biplot 3D: {agg['biplot_path']}")
        print(f"Biplot 3D + Sleep Disorder: {agg['biplot_sd_path']}")

        # Divisivo
        div = results[k_key]["divisive"]
        print(f"Modelo Divisivo: {div['model_path']}")
        print(f"Etiquetas Divisivo: {div['labels_path']}")
        print(f"Scatter 3D: {div['scatter_path']}")
        print(f"Scatter 3D + Sleep Disorder: {div['scatter_sd_path']}")
        print(f"Biplot 3D: {div['biplot_path']}")
        print(f"Biplot 3D + Sleep Disorder: {div['biplot_sd_path']}")

    print("\n✅ Clustering y visualizaciones jerárquicas completados exitosamente")
