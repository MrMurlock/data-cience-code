import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.clustering.cluster_analysis import run_cluster_descriptive_analysis

if __name__ == "__main__":
    print("Ejecutando análisis descriptivo de clusters DBSCAN...")
    results = run_cluster_descriptive_analysis()

    print("\n=== Archivos Generados ===")
    print(f"Estadísticas: {results['stats_path']}")
    print(f"Boxplots: {results['boxplots_path']}")
    print(f"Violin plots: {results['violin_plots_path']}")
    print(f"Heatmap: {results['heatmap_path']}")
    print(f"Radar chart: {results['radar_chart_path']}")

    print("\n✅ Análisis descriptivo completado exitosamente")
