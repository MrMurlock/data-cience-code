import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.clustering import run_hierarchical_analysis

if __name__ == "__main__":
    print("Ejecutando análisis de clustering jerárquico (dendrogramas + silhouette)...")
    results = run_hierarchical_analysis((2, 10))

    print("\n=== Dendrogramas Generados ===")
    print(f"Dendrograma Aglomerativo (Ward): {results['agglomerative_dendrogram_path']}")
    print(f"Dendrograma Divisivo: {results['divisive_dendrogram_path']}")

    print("\n=== Plots de Silhouette Generados ===")
    print(f"Silhouette Aglomerativo (Ward): {results['agglomerative_silhouette_plot_path']}")
    print(f"Silhouette Divisivo: {results['divisive_silhouette_plot_path']}")

    print("\n=== Métricas Guardadas ===")
    print(f"Métricas Aglomerativo: {results['agglomerative_metrics_path']}")
    print(f"Métricas Divisivo: {results['divisive_metrics_path']}")

    print("\n✅ Análisis jerárquico completado exitosamente")
    print("\nInterpreta los dendrogramas y silhouette scores para determinar el K óptimo:")
    print("- Dendrogramas: Altura de fusión indica distancia entre clusters")
    print("- Silhouette: Valores más altos indican mejor separación entre clusters")
    print("- K óptimo se determina combinando ambas métricas")
