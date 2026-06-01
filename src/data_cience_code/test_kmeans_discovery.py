import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.clustering import run_kmeans_discovery

if __name__ == "__main__":
    print("Ejecutando descubrimiento del K óptimo para k-means...")
    results = run_kmeans_discovery()

    print("\n=== Métricas Calculadas ===")
    print(f"K values: {results['metrics']['k_values']}")
    print(f"\nInertia values:")
    for k, inertia in zip(
        results["metrics"]["k_values"], results["metrics"]["inertia_values"]
    ):
        print(f"  K={k}: {inertia:.2f}")

    print(f"\nSilhouette scores:")
    for k, score in zip(
        results["metrics"]["k_values"], results["metrics"]["silhouette_scores"]
    ):
        print(f"  K={k}: {score:.4f}")

    max_idx = results["metrics"]["silhouette_scores"].index(
        max(results["metrics"]["silhouette_scores"])
    )
    best_k = results["metrics"]["k_values"][max_idx]
    best_score = results["metrics"]["silhouette_scores"][max_idx]
    print(f"\nMejor K según Silhouette Score: K={best_k} ({best_score:.4f})")

    print("\n=== Figuras Generadas ===")
    print(f"Método del codo: {results['elbow_plot_path']}")
    print(f"Silhouette Score: {results['silhouette_plot_path']}")

    print("\n=== Métricas Guardadas ===")
    print(f"Inertia: {results['inertia_metrics_path']}")
    print(f"Silhouette: {results['silhouette_metrics_path']}")

    print("\n✅ Descubrimiento del K óptimo completado exitosamente")
